"""
Base DecisionEngine interface and common contract for rev.
Ensures all backbones (encoders, causal LLMs, VLMs, and remote APIs)
return identical typed results.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
import time


class DecisionEngine(ABC):
    """
    Abstract base class for all rev decision backbones.
    Guarantees the TypeSafe System 1 Contract across all models.
    """

    model_name: str = "rev-engine"

    @abstractmethod
    def predict(
        self,
        state: Union[str, Dict[str, Any], Any] = None,
        questions: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Evaluate a single state against multiple structured questions in a single pass.

        Args:
            state: Context string, dictionary (e.g. {"image": ..., "note": ...}), or structured record.
            questions: Dictionary mapping question IDs to question specifications.
                       Each spec has:
                         - type: 'choice' | 'noul' | 'score'
                         - instructions: Question text
                         - criteria: Options (for choice/score) or boolean criteria (for noul)
            **kwargs: Backbone-specific runtime options (e.g. temperature overrides, batch size)

        Returns:
            Dict containing:
              - 'answers': Dict[str, Dict[str, Any]] with typed decision results
              - 'latency_ms': Elapsed evaluation time in milliseconds
              - 'model': Identifier of the executing backbone
        """
        pass

    def __call__(
        self,
        state: Union[str, Dict[str, Any], Any] = None,
        questions: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convenience alias allowing `model(state, questions)`."""
        return self.predict(state, questions, **kwargs)

    @staticmethod
    def normalize_question(q: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize question specification keys across conventions (t/type, ins/instructions, crit/criteria)."""
        q_type = q.get("type") or q.get("t", "choice")
        instructions = q.get("instructions") or q.get("ins", "")
        criteria = q.get("criteria") if "criteria" in q else q.get("crit", [])
        return {
            "type": q_type,
            "instructions": str(instructions),
            "criteria": criteria,
        }

    @staticmethod
    def extract_options(q: Dict[str, Any]) -> List[str]:
        """Extract ordered option label list from a question spec."""
        q_type = q.get("type", "choice")
        criteria = q.get("criteria", [])
        if q_type == "noul":
            return ["false", "true"]
        if isinstance(criteria, dict):
            return list(criteria.keys())
        if isinstance(criteria, list):
            return [str(c) for c in criteria]
        return ["option_a", "option_b"]
