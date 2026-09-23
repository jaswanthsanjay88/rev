"""
AutoModel dispatcher for rev: Rev.from_pretrained(...)
Provides a single, unified entrypoint across all rev decision backbones.
"""

import os
from typing import Any, Dict, Optional, Union
from .engine_base import DecisionEngine


def _detect_architecture(
    model_id_or_path: str,
    token: Optional[str] = None,
) -> str:
    """
    Detect the architecture of a rev checkpoint:
    Returns: 'vision' | 'encoder' | 'causal' | 'mock'
    """
    m = model_id_or_path.lower().strip()

    # 1. Direct aliases
    if m in ("mock", "test", "dummy"):
        return "mock"
    if "vision" in m:
        return "vision"
    if "4b" in m or "0.5b" in m or "qwen" in m:
        return "causal"
    if "decision-model" in m or "modernbert" in m or "laya" in m:
        return "encoder"

    # 2. Local directory inspection
    if os.path.isdir(model_id_or_path):
        files = os.listdir(model_id_or_path)
        if "vlm_agent_config.json" in files:
            return "vision"
        if "rl_agent_config.json" in files or "encoder" in files:
            return "encoder"
        if "head.pt" in files or "adapter_config.json" in files:
            return "causal"

    # 3. Hugging Face Hub file inspection (fast shallow probe)
    try:
        from huggingface_hub import HfApi

        api = HfApi(token=token or os.environ.get("HF_TOKEN"))
        repo_files = set(api.list_repo_files(model_id_or_path))

        if "vlm_agent_config.json" in repo_files:
            return "vision"
        if "rl_agent_config.json" in repo_files or "encoder/config.json" in repo_files:
            return "encoder"
        if "head.pt" in repo_files or "adapter_config.json" in repo_files:
            return "causal"
    except Exception:
        pass

    # Default fallback to encoder
    return "encoder"


class Rev:
    """
    Primary developer entrypoint for the rev ecosystem.

    Usage:
        from rev import Rev

        # Automatically detects and dispatches to the correct backbone:
        model = Rev.from_pretrained("jaswanthsanjay88/rev-vision")
        model = Rev.from_pretrained("jaswanthsanjay88/rev-decision-model")
        model = Rev.from_pretrained("jaswanthsanjay88/rev-4b")
        model = Rev.from_pretrained("jaswanthsanjay88/rev-0.5b")
        model = Rev.from_pretrained("mock")

        # All return the exact same .predict(state, questions) shape!
        res = model.predict(state, questions)
    """

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str = "jaswanthsanjay88/rev-decision-model",
        device: Optional[str] = None,
        token: Optional[str] = None,
        **kwargs: Any,
    ) -> DecisionEngine:
        """
        Instantiate the appropriate DecisionEngine for any rev checkpoint.

        Args:
            pretrained_model_name_or_path: Hugging Face repo ID, local path, or alias
                                           ('mock', 'rev-vision', 'rev-decision-model', etc.)
            device: 'cuda', 'cpu', or device ID
            token: Optional Hugging Face access token
            **kwargs: Additional engine-specific arguments
        """
        arch = kwargs.pop("architecture", None) or _detect_architecture(
            pretrained_model_name_or_path,
            token=token,
        )

        if arch == "mock":
            from .adapters.mock import MockDecisionEngine
            return MockDecisionEngine(model_name=pretrained_model_name_or_path)

        if arch == "vision":
            from .adapters.vision import VisionDecisionEngine
            return VisionDecisionEngine(
                model_id_or_path=pretrained_model_name_or_path,
                device=device,
                token=token,
                **kwargs,
            )

        if arch == "causal":
            from .adapters.causal import CausalDecisionEngine
            return CausalDecisionEngine(
                model_id_or_path=pretrained_model_name_or_path,
                device=device,
                token=token,
                **kwargs,
            )

        # Default: encoder (ModernBERT)
        from .adapters.encoder import EncoderDecisionEngine
        return EncoderDecisionEngine(
            model_id_or_path=pretrained_model_name_or_path,
            device=device,
            token=token,
            **kwargs,
        )
