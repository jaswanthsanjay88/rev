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
    if "lfm" in m or "liquid" in m:
        return "lfm"
    if "4b" in m or "0.5b" in m or "qwen" in m:
        return "causal"
    if "decision-model" in m or "modernbert" in m or "laya" in m:
        return "encoder"

    # 2. Local directory inspection
    if os.path.isdir(model_id_or_path):
        files = os.listdir(model_id_or_path)
        if "manifest.json" in files:
            return "bundle"
        if "vlm_agent_config.json" in files:
            return "vision"
        if "rl_agent_config.json" in files or "encoder" in files:
            return "encoder"
        if "readout_head.pt" in files:
            return "lfm"
        if "adapter_config.json" in files:
            try:
                import json
                with open(os.path.join(model_id_or_path, "adapter_config.json"), "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    base_ref = str(cfg.get("base_model_name_or_path", "")).lower()
                    if "lfm" in base_ref or "liquid" in base_ref:
                        return "lfm"
            except Exception:
                pass
            return "causal"
        if "head.pt" in files:
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

        if arch == "bundle":
            from .bundle import load_manifest
            from .finetune import RevFinetunedModel
            from .adapters.lfm import LfmReadoutHead
            from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
            import torch

            manifest = load_manifest(pretrained_model_name_or_path)
            dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
            tok_path = pretrained_model_name_or_path if os.path.exists(os.path.join(pretrained_model_name_or_path, "tokenizer.json")) else (manifest.backbone if manifest else "LiquidAI/LFM2.5-Encoder-350M")
            tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
            if not tokenizer.pad_token_id:
                tokenizer.pad_token_id = tokenizer.eos_token_id or 0

            base_ref = manifest.backbone if manifest else "LiquidAI/LFM2.5-Encoder-350M"
            try:
                full = AutoModelForMaskedLM.from_pretrained(base_ref, trust_remote_code=True)
                base_model = getattr(full, "lfm2", full)
            except Exception:
                base_model = AutoModel.from_pretrained(base_ref, trust_remote_code=True)

            if os.path.exists(os.path.join(pretrained_model_name_or_path, "adapter_config.json")):
                from peft import PeftModel
                backbone = PeftModel.from_pretrained(base_model, pretrained_model_name_or_path).to(dev)
            else:
                backbone = base_model.to(dev)

            hidden_dim = getattr(base_model.config, "hidden_size", getattr(base_model.config, "d_model", 1024))
            head = LfmReadoutHead(hidden_size=hidden_dim, head_dim=256).to(dev)
            head_file = os.path.join(pretrained_model_name_or_path, "readout_head.pt")
            if os.path.exists(head_file):
                h_data = torch.load(head_file, map_location=dev)
                head.load_state_dict(h_data.get("head", h_data))

            return RevFinetunedModel(
                backbone=backbone,
                tokenizer=tokenizer,
                readout_head=head,
                labels=manifest.labels if manifest else [],
                optimal_temperature=manifest.optimal_temperature if manifest else 1.0,
                deferral_threshold=manifest.deferral_threshold if manifest else 0.80,
                cost_matrix=manifest.cost_matrix if manifest else None,
                device=str(dev),
                data_hash=manifest.data_hash if manifest else "",
                backbone_name=manifest.backbone if manifest else "",
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

        if arch == "lfm":
            from .adapters.lfm import LFMDecisionEngine
            return LFMDecisionEngine(
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

    @classmethod
    def finetune(
        cls,
        backbone: str = "jaswanthsanjay88/rev-decision-model",
        dataset: Any = None,
        cost_matrix: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ):
        """
        Launches client fine-tuning run for custom decision data.
        
        Example:
            from rev import Rev, Dataset
            ds = Dataset.from_csv("tickets.csv", text="body", label="dept")
            run = Rev.finetune("jaswanthsanjay88/rev-decision-model", ds)
            model = run.fit(epochs=3)
            model.save("./my-calibrated-model")
        """
        from .finetune import FinetuneRun
        return FinetuneRun(
            backbone=backbone,
            dataset=dataset,
            cost_matrix=cost_matrix,
            **kwargs,
        )
