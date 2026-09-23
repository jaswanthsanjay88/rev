"""
Backbone adapters for rev:
- VisionDecisionEngine: SmolVLM multimodal decision engine (rev-vision)
- EncoderDecisionEngine: ModernBERT bidirectional text decision engine (rev-decision-model)
- CausalDecisionEngine: Qwen causal backbones with PointerHead and prefix KV-cache (rev-4b, rev-0.5b)
- MockDecisionEngine: Ultra-fast heuristic mock engine for testing without neural weights
"""

from ..engine_base import DecisionEngine
from .mock import MockDecisionEngine

__all__ = [
    "DecisionEngine",
    "MockDecisionEngine",
]
