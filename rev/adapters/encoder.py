"""
EncoderDecisionEngine: Bidirectional encoder backbone (ModernBERT-large).
Powers jaswanthsanjay88/rev-decision-model with sub-millisecond local inference.
"""

import time
from typing import Any, Dict, Optional, Union
from ..engine_base import DecisionEngine


class EncoderDecisionEngine(DecisionEngine):
    """Adapter for bidirectional encoder models (ModernBERT-large)."""

    def __init__(
        self,
        model_id_or_path: str = "jaswanthsanjay88/rev-decision-model",
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        **kwargs: Any,
    ):
        from ..agent import Agent

        self.model_name = model_id_or_path
        self._agent = Agent(
            model_id_or_path=model_id_or_path,
            device=device,
            token=token,
            subfolder=subfolder,
        )

    def predict(
        self,
        state: Union[str, Dict[str, Any], Any],
        questions: Dict[str, Dict[str, Any]],
        **kwargs: Any,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        raw_res = self._agent.predict(state=state, questions=questions, **kwargs)
        latency = (time.perf_counter() - t0) * 1000.0

        # Ensure consistent structure
        answers = raw_res.get("answers", raw_res)
        return {
            "answers": answers,
            "latency_ms": round(raw_res.get("latency_ms", latency), 2),
            "model": self.model_name,
        }
