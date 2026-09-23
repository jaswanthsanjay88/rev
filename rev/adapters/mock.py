"""
MockDecisionEngine: Fast, deterministic semantic decision engine
requiring zero neural weights, PyTorch, or GPU.
"""

import hashlib
import math
import time
from typing import Any, Dict, List, Union
from ..engine_base import DecisionEngine


class MockDecisionEngine(DecisionEngine):
    """Zero-dependency heuristic decision engine for testing and lightweight pipelines."""

    def __init__(self, model_name: str = "rev-mock"):
        self.model_name = model_name

    def predict(
        self,
        state: Union[str, Dict[str, Any], Any] = None,
        questions: Dict[str, Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        if questions is None:
            questions = kwargs.get("questions") or {}
        has_image = False
        if isinstance(state, dict):
            has_image = bool(state.get("image") or state.get("image_b64") or kwargs.get("image"))
            state_str = str(state.get("note", state.get("state", ""))).lower()
        else:
            has_image = bool(kwargs.get("image"))
            state_str = str(state or "").lower()

        answers: Dict[str, Dict[str, Any]] = {}

        for q_id, raw_q in questions.items():
            q = self.normalize_question(raw_q)
            q_type = q["type"]
            instr = q["instructions"].lower()
            options = self.extract_options(q)

            if q_type == "choice":
                # Compute overlap scores
                scores = []
                for opt in options:
                    opt_lower = opt.lower()
                    overlap = sum(1 for w in opt_lower.split() if len(w) > 2 and w in state_str)
                    if has_image and any(w in instr for w in ["receipt", "restaurant", "food", "bill", "menu", "dining"]):
                        if opt_lower.startswith("yes") or "restaurant" in opt_lower or "food" in opt_lower:
                            overlap += 4.0
                        elif opt_lower.startswith("no") or "not" in opt_lower:
                            overlap -= 1.0
                    # Deterministic hash component
                    h = int(hashlib.md5(f"{state_str[:200]}:{instr}:{opt_lower}".encode()).hexdigest()[:6], 16) / 0xFFFFFF
                    scores.append(max(0.01, 1.0 + overlap * 3.0 + h * 0.5))

                exp_scores = [math.exp(s / 1.0) for s in scores]
                total = sum(exp_scores) or 1.0
                probs = [s / total for s in exp_scores]
                best_idx = int(max(range(len(probs)), key=lambda i: probs[i]))

                answers[q_id] = {
                    "choice": options[best_idx],
                    "confidence": round(float(probs[best_idx]), 4),
                    "probabilities": {opt: round(float(p), 4) for opt, p in zip(options, probs)},
                }

            elif q_type == "noul":
                urgent_keywords = ["urgent", "immediately", "critical", "emergency", "danger", "hazard", "cancel", "fail", "true", "yes"]
                matches = sum(1 for w in urgent_keywords if w in state_str or w in instr)
                p_true = min(0.99, 0.65 + matches * 0.12) if matches > 0 else 0.08
                answers[q_id] = {
                    "noul": round(p_true, 4),
                    "confidence": round(max(p_true, 1.0 - p_true), 4),
                }

            elif q_type == "score":
                n_levels = len(options) if options else 3
                # Bell-shaped curve around middle level
                center = (n_levels - 1) / 2.0
                raw_p = [math.exp(-((i - center) ** 2) / 1.5) for i in range(n_levels)]
                total_p = sum(raw_p) or 1.0
                level_probs = [p / total_p for p in raw_p]
                expected_score = sum(i * p for i, p in enumerate(level_probs))

                answers[q_id] = {
                    "score": round(expected_score, 3),
                    "distribution": [round(p, 4) for p in level_probs],
                    "criteria": options,
                }

        latency = (time.perf_counter() - t0) * 1000.0
        return {
            "answers": answers,
            "latency_ms": round(latency, 2),
            "model": self.model_name,
        }
