"""
VisionDecisionEngine: Multimodal vision decision engine (SmolVLM-256M).
Powers jaswanthsanjay88/rev-vision with sub-40ms visual decisions.
"""

import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from ..engine_base import DecisionEngine


class VisionDecisionEngine(DecisionEngine):
    """Adapter for multimodal vision models (rev-vision based on SmolVLM-256M)."""

    def __init__(
        self,
        model_id_or_path: str = "jaswanthsanjay88/rev-vision",
        device: Optional[str] = None,
        dtype: Optional[Any] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ):
        import torch
        import torch.nn as nn
        from transformers import AutoProcessor, AutoModelForImageTextToText
        from peft import PeftModel
        from safetensors.torch import load_file
        from huggingface_hub import hf_hub_download

        self.model_name = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            self.dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32
        else:
            self.dtype = dtype

        # 1. Decision Head definition
        class _RevVisionDecisionHead(nn.Module):
            def __init__(self, hidden_size: int = 576, head_hidden: int = 256):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(hidden_size, head_hidden),
                    nn.GELU(),
                    nn.LayerNorm(head_hidden),
                    nn.Linear(head_hidden, 1),
                )

            def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
                return self.net(hidden_states).squeeze(-1)

        # 2. Processor with single-tile configuration (64 visual tokens)
        self.processor = AutoProcessor.from_pretrained(
            model_id_or_path,
            subfolder="processor" if not os.path.exists(model_id_or_path) else None,
            token=token or os.environ.get("HF_TOKEN"),
        )
        if hasattr(self.processor, "image_processor") and hasattr(self.processor.image_processor, "do_image_splitting"):
            self.processor.image_processor.do_image_splitting = False

        # 3. Base SmolVLM + LoRA adapter
        base_id = kwargs.get("base_model", "HuggingFaceTB/SmolVLM-256M-Instruct")
        base_model = AutoModelForImageTextToText.from_pretrained(
            base_id,
            torch_dtype=self.dtype,
            device_map=self.device,
            token=token or os.environ.get("HF_TOKEN"),
        )
        self.model = PeftModel.from_pretrained(
            base_model,
            model_id_or_path,
            subfolder="adapter" if not os.path.exists(model_id_or_path) else None,
            token=token or os.environ.get("HF_TOKEN"),
        )
        self.model.eval()

        # 4. Load decision head weights
        if os.path.isdir(model_id_or_path):
            head_path = os.path.join(model_id_or_path, "head.safetensors")
            if not os.path.exists(head_path):
                head_path = os.path.join(model_id_or_path, "model.safetensors")
        else:
            try:
                head_path = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename="head.safetensors",
                    token=token or os.environ.get("HF_TOKEN"),
                )
            except Exception:
                head_path = hf_hub_download(
                    repo_id=model_id_or_path,
                    filename="model.safetensors",
                    token=token or os.environ.get("HF_TOKEN"),
                )

        hidden_dim = getattr(getattr(base_model.config, "text_config", None), "hidden_size", 576)
        self.head = _RevVisionDecisionHead(hidden_size=hidden_dim).to(device=self.device, dtype=self.dtype)
        self.head.load_state_dict(load_file(head_path))
        self.head.eval()

        # 5. Default temperatures
        self.temperatures = kwargs.get("temperatures", {"choice": 2.2028, "score": 1.3652, "noul": 2.1333})

    def predict(
        self,
        state: Union[str, Dict[str, Any], Any],
        questions: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import torch
        import torch.nn.functional as F
        from PIL import Image

        t0 = time.perf_counter()
        image = None
        note = ""

        if isinstance(state, dict):
            image = state.get("image")
            note = str(state.get("note", state.get("state", "")))
        elif isinstance(state, str):
            note = state

        if isinstance(image, str) and os.path.exists(image):
            image = Image.open(image).convert("RGB")
        elif image is None:
            image = Image.new("RGB", (512, 512), color="white")

        newline_id = self.processor.tokenizer.encode("\n", add_special_tokens=False)[-1]
        answers: Dict[str, Dict[str, Any]] = {}

        for q_key, raw_q in questions.items():
            q = self.normalize_question(raw_q)
            q_type = q["type"]
            instr = q["instructions"]
            options = self.extract_options(q)

            prompt = f"<|im_start|>User:<image>{note}\n{q_type.upper()} question: {instr}<end_of_utterance>\nAssistant: Options:\n"
            for opt in options:
                prompt += f"- {opt}\n"

            inputs = self.processor(text=prompt, images=image, return_tensors="pt")
            inputs = {
                k: v.to(device=self.device, dtype=self.dtype) if v.is_floating_point() else v.to(device=self.device)
                for k, v in inputs.items()
            }
            input_ids = inputs["input_ids"][0]

            newlines = (input_ids == newline_id).nonzero(as_tuple=True)[0]
            terminators = (
                newlines[-len(options) :].tolist()
                if len(newlines) >= len(options)
                else list(range(len(input_ids) - len(options), len(input_ids)))
            )

            with torch.no_grad():
                out = self.model(**inputs, output_hidden_states=True)
                hidden = out.hidden_states[-1][0, terminators, :]
                logits = self.head(hidden.unsqueeze(0))[0] / self.temperatures.get(q_type, 1.0)
                probs = F.softmax(logits, dim=-1).float().cpu().numpy().tolist()

            if q_type == "choice":
                best = int(np.argmax(probs))
                answers[q_key] = {
                    "choice": options[best],
                    "confidence": round(float(probs[best]), 4),
                    "probabilities": {opt: round(float(p), 4) for opt, p in zip(options, probs)},
                }
            elif q_type == "noul":
                p_true = round(float(probs[1]), 4)
                answers[q_key] = {
                    "noul": p_true,
                    "confidence": round(max(p_true, 1.0 - p_true), 4),
                }
            elif q_type == "score":
                expected = sum(k * p for k, p in enumerate(probs))
                answers[q_key] = {
                    "score": round(float(expected), 3),
                    "distribution": [round(float(p), 4) for p in probs],
                    "criteria": options,
                }

        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "answers": answers,
            "latency_ms": round(latency, 2),
            "model": self.model_name,
        }
