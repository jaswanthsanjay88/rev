"""
CausalDecisionEngine: Causal autoregressive backbones (Qwen2.5-0.5B, Qwen3-4B)
adapted into prefill-only decision heads with prefix KV-caching.
"""

import os
import time
from typing import Any, Dict, Optional, Union
from ..engine_base import DecisionEngine


class CausalDecisionEngine(DecisionEngine):
    """Adapter for causal decoder models with PointerHead."""

    def __init__(
        self,
        model_id_or_path: str = "jaswanthsanjay88/rev-0.5b",
        device: Optional[str] = None,
        token: Optional[str] = None,
        use_cache: bool = True,
        **kwargs: Any,
    ):
        import torch
        from transformers import AutoTokenizer
        from peft import PeftModel
        from ..model import DecisionModel
        from ..cache import StateKVCacheManager

        self.model_name = model_id_or_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.use_cache = use_cache
        self.cache = StateKVCacheManager(max_entries=64) if use_cache else None

        # Resolve base model name
        base_name = kwargs.get("base_model")
        if not base_name:
            if "4b" in model_id_or_path.lower():
                base_name = "Qwen/Qwen3-4B-Base"
            else:
                base_name = "Qwen/Qwen2.5-0.5B"

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id_or_path,
            token=token or os.environ.get("HF_TOKEN"),
        )
        if not self.tokenizer.pad_token_id:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # Load DecisionModel
        self.model = DecisionModel.from_pretrained(
            model_id_or_path=model_id_or_path,
            base_model_name=base_name,
            device=self.device,
            token=token or os.environ.get("HF_TOKEN"),
        )
        self.model.eval()

    def predict(
        self,
        state: Union[str, Dict[str, Any], Any],
        questions: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        import hashlib
        from ..api import SystemOneRequest, to_record, to_answers

        t0 = time.perf_counter()
        req = SystemOneRequest(
            state=state,
            questions=questions,
            model=kwargs.get("model", self.model_name),
        )
        rec, meta = to_record(req)

        state_str = str(rec.get("state", ""))
        state_hash = hashlib.sha256(state_str.strip().encode("utf-8")).hexdigest()

        cached_hit = False
        if self.use_cache and self.cache:
            entry = self.cache.get_by_hash(state_hash)
            if entry is not None:
                ps = self.model.probs_cached(self.tokenizer, rec, entry.past_key_values, entry.state_len)
                cached_hit = True
            else:
                pkv, s_len, _ = self.model.compute_state_cache(self.tokenizer, state_str)
                self.cache.put(state_str, pkv, s_len)
                ps = self.model.probs_cached(self.tokenizer, rec, pkv, s_len)
        else:
            from ..model import encode
            enc = encode(self.tokenizer, rec)
            ps = self.model.probs(enc)

        answers = to_answers([p.tolist() if hasattr(p, "tolist") else p for p in ps], meta)
        latency = (time.perf_counter() - t0) * 1000.0

        return {
            "answers": answers,
            "latency_ms": round(latency, 2),
            "model": self.model_name,
            "cached": cached_hit,
        }
