"""
High-level inference runtime for rev encoder decision models.
Provides calibrated, non-autoregressive, sub-10ms decision evaluations.
"""

import json
import os
import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .common import (
    QTYPES,
    TEMP_MAX,
    TEMP_MIN,
    amp_dtype,
    build_sequence,
    clamp_temperature,
    collate_items,
    confidence_from_probs,
    render_options,
    temp_bucket,
)
from .encoder_model import EncoderDecisionModel, build_encoder_model


def _fix_tokenizer_config(path: str):
    """Ensure tokenizer_config.json loads cleanly across all transformers versions."""
    cfg_file = os.path.join(path, "tokenizer", "tokenizer_config.json")
    if not os.path.exists(cfg_file):
        cfg_file = os.path.join(path, "tokenizer_config.json")
        if not os.path.exists(cfg_file):
            return
    try:
        with open(cfg_file) as f:
            tcfg = json.load(f)
        changed = False
        if tcfg.get("tokenizer_class") in (None, "TokenizersBackend"):
            tcfg["tokenizer_class"] = "PreTrainedTokenizerFast"
            tcfg.pop("backend", None)
            tcfg.pop("is_local", None)
            changed = True
        extra = tcfg.get("extra_special_tokens")
        if isinstance(extra, list):
            tcfg["extra_special_tokens"] = {"extra_%d" % i: t for i, t in enumerate(extra)}
            changed = True
        if changed:
            with open(cfg_file, "w") as f:
                json.dump(tcfg, f, indent=2)
    except Exception:
        pass


class Agent:
    """System 1 encoder decision runtime: fast, non-autoregressive, calibrated decisions."""

    def __init__(
        self,
        model_id_or_path: str = "convaiinnovations/laya",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
    ):
        """
        Load an encoder checkpoint.
        `subfolder` can select specific checkpoints (e.g. 'multilingual' or 'typed-decisions').
        """
        from safetensors.torch import load_file
        from transformers import AutoTokenizer

        model_dir = model_id_or_path
        if not os.path.exists(model_dir):
            if model_id_or_path.startswith(("/", "./", "../")) or os.path.isabs(model_id_or_path):
                raise FileNotFoundError(f"Local model path not found: {model_id_or_path!r}")
            from huggingface_hub import snapshot_download

            prefix = f"{subfolder}/" if subfolder else ""
            kw = {
                "token": token or os.environ.get("HF_TOKEN"),
                "allow_patterns": [
                    prefix + name
                    for name in (
                        "rl_agent_config.json",
                        "model.safetensors",
                        "tokenizer/*",
                        "tokenizer_config.json",
                        "vocab.json",
                        "merges.txt",
                        "encoder/*",
                    )
                ],
            }
            model_dir = snapshot_download(model_id_or_path, **kw)

        if subfolder:
            model_dir = os.path.join(model_dir, subfolder)
            if not os.path.isdir(model_dir):
                raise FileNotFoundError(f"Subfolder {subfolder!r} not found in {model_id_or_path!r}.")

        _fix_tokenizer_config(model_dir)

        cfg_path = os.path.join(model_dir, "rl_agent_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                self.cfg = json.load(f)
        else:
            self.cfg = {
                "encoder": "answerdotai/ModernBERT-large",
                "head_layers": 2,
                "max_len": 512,
                "head_max_len": 192,
            }

        weights_path = os.path.join(model_dir, "model.safetensors")
        weights = load_file(weights_path) if os.path.exists(weights_path) else None

        tok_dir = os.path.join(model_dir, "tokenizer")
        if not os.path.exists(tok_dir):
            tok_dir = model_dir
        self.tok = AutoTokenizer.from_pretrained(tok_dir, fix_markdown=False)

        enc_dir = os.path.join(model_dir, "encoder")
        self.model = build_encoder_model(self.cfg, enc_dir if os.path.exists(enc_dir) else None)

        if weights is not None:
            self.model.load_state_dict(weights, strict=False)

        self.model.eval()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)

        self.max_len = int(self.cfg.get("max_len", 512))
        self.head_max_len = int(self.cfg.get("head_max_len", 192))
        self.temperatures = self.cfg.get("temperatures", {})

    @torch.no_grad()
    def system_one(
        self,
        state: Union[str, dict, list],
        questions: Dict[str, Any],
        temperatures: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate all questions for state in a single non-autoregressive forward pass.
        Returns dict containing answers, probabilities, and confidence scores.
        """
        if not questions:
            return {"answers": {}, "actions": {}}

        items = []
        q_order = []
        for qid, q in questions.items():
            q_norm = {
                "t": q.get("type") or q.get("t", "choice"),
                "ins": q.get("instructions") or q.get("ins", ""),
                "crit": q.get("criteria") or q.get("crit", {}),
            }
            qtype_id = QTYPES[q_norm["t"]]
            ids, markers = build_sequence(
                self.tok,
                state,
                q_norm,
                max_len=self.max_len,
                head_max_len=self.head_max_len,
            )
            items.append({
                "ids": ids,
                "markers": markers,
                "qtype": qtype_id,
                "qid": qid,
                "q": q_norm,
            })
            q_order.append((qid, q_norm))

        pad_id = getattr(self.tok, "pad_token_id", None) or 0
        batch = collate_items(items, pad_id=pad_id)

        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        marker_pos = batch["marker_pos"].to(self.device)
        marker_mask = batch["marker_mask"].to(self.device)
        qtype = batch["qtype"].to(self.device)

        logits, act_logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            marker_pos=marker_pos,
            marker_mask=marker_mask,
            qtype=qtype,
        )

        answers = {}
        for i, (qid, q_norm) in enumerate(q_order):
            k = int(marker_mask[i].sum().item())
            row_logits = logits[i, :k].cpu().numpy()

            # Temperature calibration
            bucket = temp_bucket(QTYPES[q_norm["t"]], k)
            t_val = 1.0
            if temperatures and bucket in temperatures:
                t_val = clamp_temperature(temperatures[bucket])
            elif bucket in self.temperatures:
                t_val = clamp_temperature(self.temperatures[bucket])

            calibrated_logits = row_logits / t_val
            exp_l = np.exp(calibrated_logits - np.max(calibrated_logits))
            probs = exp_l / np.sum(exp_l)

            conf = confidence_from_probs(probs, k)
            t = q_norm["t"]

            if t == "choice":
                opts = render_options(q_norm)
                crit = q_norm["crit"]
                keys = list(crit.keys()) if isinstance(crit, dict) else [str(j) for j in range(len(opts))]
                best_idx = int(np.argmax(probs))
                answers[qid] = {
                    "type": "choice",
                    "choice": keys[best_idx] if best_idx < len(keys) else str(best_idx),
                    "confidence": round(conf, 4),
                    "probabilities": {keys[j]: round(float(probs[j]), 4) for j in range(min(len(keys), k))},
                }
            elif t == "score":
                expected_score = float(np.sum(np.arange(k) * probs))
                answers[qid] = {
                    "type": "score",
                    "score": round(expected_score, 3),
                    "confidence": round(conf, 4),
                    "probabilities": {str(j): round(float(probs[j]), 4) for j in range(k)},
                }
            else:  # noul
                # probs[0] is false, probs[1] is true
                p_true = float(probs[1]) if k >= 2 else 0.0
                answers[qid] = {
                    "type": "noul",
                    "noul": round(p_true, 4),
                    "confidence": round(conf, 4),
                }

        # Action head / deferral
        act_probs = torch.softmax(act_logits, dim=-1).cpu().numpy()
        actions = {}
        for i, (qid, _) in enumerate(q_order):
            actions[qid] = {
                "action": int(np.argmax(act_probs[i])),
                "probabilities": [round(float(p), 4) for p in act_probs[i]],
            }

        return {"answers": answers, "actions": actions}

    predict = system_one
