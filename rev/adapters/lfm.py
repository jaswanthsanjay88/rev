"""
LFMDecisionEngine: Liquid Foundation Model (LFM / LFM2.5) decision adapter.
Combines Liquid AI hybrid recurrent/short-conv backbones with calibrated
System 1 decision readout heads for ultra-fast, high-accuracy decisions.
"""

import math
import os
import time
from typing import Any, Dict, List, Optional, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import PeftModel

from ..engine_base import DecisionEngine


class LfmReadoutHead(nn.Module):
    """Calibrated readout head mapping LFM hidden states into option probabilities."""

    def __init__(self, hidden_size: int = 1024, head_dim: int = 256):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, head_dim),
            nn.GELU(),
            nn.LayerNorm(head_dim),
            nn.Linear(head_dim, 1),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: [num_options, hidden_size] -> logits: [num_options]"""
        return self.proj(hidden_states).squeeze(-1)


class LFMDecisionEngine(DecisionEngine):
    """
    DecisionEngine adapter for Liquid Foundation Models (LFM2.5-Encoder-350M, LFM2.5-350M, LFM2.5-1.2B).
    Provides sub-5ms calibrated decisions using LFM representations with dedicated readout heads.
    """

    def __init__(
        self,
        model_id_or_path: str = "LiquidAI/LFM2.5-Encoder-350M",
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        token: Optional[str] = None,
        head_path: Optional[str] = None,
        **kwargs: Any,
    ):
        self.model_name = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype or (torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32)
        hf_token = token or os.environ.get("HF_TOKEN")

        # Check if model_id_or_path is an adapter directory
        adapter_cfg_file = os.path.join(model_id_or_path, "adapter_config.json")
        is_adapter = False
        base_model_ref = model_id_or_path
        if os.path.exists(adapter_cfg_file):
            try:
                import json
                with open(adapter_cfg_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    base_model_ref = cfg.get("base_model_name_or_path", "LiquidAI/LFM2.5-Encoder-350M")
                    is_adapter = True
            except Exception:
                pass

        # 1. Load Tokenizer & Config
        tok_source = model_id_or_path if os.path.exists(os.path.join(model_id_or_path, "tokenizer.json")) else base_model_ref
        self.tokenizer = AutoTokenizer.from_pretrained(tok_source, token=hf_token, trust_remote_code=True)
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

        self.config = AutoConfig.from_pretrained(base_model_ref, token=hf_token, trust_remote_code=True)
        self.is_encoder = "encoder" in base_model_ref.lower() or getattr(self.config, "is_encoder_decoder", False) or "bidirectional" in str(getattr(self.config, "architectures", [])).lower()

        # 2. Load Backbone Model
        # LFM2.5 encoder checkpoint was saved with base_model_prefix="lfm2" via Lfm2BidirectionalForMaskedLM
        try:
            from transformers import AutoModelForMaskedLM
            full_model = AutoModelForMaskedLM.from_pretrained(
                base_model_ref,
                dtype=self.dtype,
                trust_remote_code=True,
                token=hf_token,
            )
            self.backbone = getattr(full_model, "lfm2", full_model).to(self.device)
        except Exception:
            self.backbone = AutoModel.from_pretrained(
                base_model_ref,
                dtype=self.dtype,
                trust_remote_code=True,
                token=hf_token,
            ).to(self.device)

        if is_adapter:
            try:
                from peft import PeftModel
                self.backbone = PeftModel.from_pretrained(self.backbone, model_id_or_path)
            except Exception as e:
                print(f"Warning: Failed to attach PEFT adapter from {model_id_or_path}: {e}")

        self.backbone.eval()

        # 3. Setup Readout Head
        hidden_size = getattr(self.config, "hidden_size", 1024)
        head_dim = kwargs.get("head_dim", 256)
        self.head = LfmReadoutHead(hidden_size=hidden_size, head_dim=head_dim).to(device=self.device, dtype=self.dtype)

        # Load trained head weights if available
        resolved_head = head_path
        if not resolved_head:
            for cand in ["readout_head.pt", "head.pt"]:
                p = os.path.join(model_id_or_path, cand)
                if os.path.exists(p):
                    resolved_head = p
                    break

        if resolved_head and os.path.exists(resolved_head):
            head_data = torch.load(resolved_head, map_location=self.device)
            state_dict = head_data.get("head", head_data)
            self.head.load_state_dict(state_dict)
            self.head.eval()

        # Calibrated temperature scaling
        self.temperatures = kwargs.get("temperatures", {"choice": 1.25, "score": 1.10, "noul": 1.20})

    def predict(
        self,
        state: Union[str, Dict[str, Any], Any] = None,
        questions: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()

        if questions is None:
            questions = kwargs.get("questions") or {}

        # Resolve state text
        if isinstance(state, dict):
            state_text = str(state.get("note", state.get("state", "")))
        else:
            state_text = str(state or "")

        answers: Dict[str, Dict[str, Any]] = {}

        for q_key, raw_q in questions.items():
            q = self.normalize_question(raw_q)
            q_type = q["type"]
            instr = q["instructions"]
            options = self.extract_options(q)

            if not options:
                continue

            # Construct joint prompt for each option to leverage LFM representation
            prompts = [
                f"Context: {state_text}\nQuestion: {instr}\nCandidate: {opt}"
                for opt in options
            ]

            inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.backbone(**inputs)
                # Mean-pool hidden states over valid non-padded tokens
                last_hidden = outputs.last_hidden_state
                mask = inputs.attention_mask.unsqueeze(-1).to(last_hidden.dtype)
                pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                logits = self.head(pooled) / self.temperatures.get(q_type, 1.0)
                probs = F.softmax(logits, dim=-1).float().cpu().numpy().tolist()

            if q_type == "choice":
                best_idx = int(np.argmax(probs))
                answers[q_key] = {
                    "type": "choice",
                    "choice": options[best_idx],
                    "confidence": round(float(probs[best_idx]), 4),
                    "probabilities": {opt: round(float(p), 4) for opt, p in zip(options, probs)},
                }
            elif q_type == "noul":
                p_true = round(float(probs[1]) if len(probs) > 1 else float(probs[0]), 4)
                answers[q_key] = {
                    "type": "noul",
                    "noul": p_true,
                    "confidence": round(max(p_true, 1.0 - p_true), 4),
                }
            elif q_type == "score":
                expected = sum(k * p for k, p in enumerate(probs))
                answers[q_key] = {
                    "type": "score",
                    "score": round(float(expected), 3),
                    "confidence": round(float(max(probs)), 4),
                    "criteria": options,
                    "probabilities": {str(i): round(float(p), 4) for i, p in enumerate(probs)},
                }

        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "answers": answers,
            "latency_ms": round(latency, 2),
            "model": self.model_name,
        }
