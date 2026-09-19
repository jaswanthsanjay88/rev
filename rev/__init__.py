"""
rev: Prefill-only decision model with TypeSafe System One API.
"""

__version__ = "0.1.0"

try:
    from .model import DecisionModel, PointerHead, encode, user_tokens, branch_mask_batch
except ImportError:
    pass

from .api import (
    Noul,
    Choice,
    Score,
    SystemOneRequest,
    to_record,
    to_answers,
    render,
    choice_confidence,
    score_confidence,
)

__all__ = [
    "Noul",
    "Choice",
    "Score",
    "SystemOneRequest",
    "to_record",
    "to_answers",
    "render",
    "choice_confidence",
    "score_confidence",
]
