"""
rev.hardware: Automatic hardware planner and execution profile generator.

Inspects available compute (CUDA, MPS, CPU, VRAM) and dataset characteristics
to automatically produce optimal batch size, precision (bf16/fp16/fp32),
gradient accumulation, and adaptation strategy.
"""

import os
from typing import Any, Dict, Optional
import torch


class HardwareProfile:
    """Hardware capabilities and recommended training hyperparameters."""

    def __init__(
        self,
        device: str,
        vram_gb: float,
        device_name: str,
        dtype: torch.dtype,
        batch_size: int,
        grad_accum_steps: int,
        use_gradient_checkpointing: bool,
        lora_r: int,
        strategy: str,
    ):
        self.device = device
        self.vram_gb = vram_gb
        self.device_name = device_name
        self.dtype = dtype
        self.batch_size = batch_size
        self.grad_accum_steps = grad_accum_steps
        self.use_gradient_checkpointing = use_gradient_checkpointing
        self.lora_r = lora_r
        self.strategy = strategy

    def to_dict(self) -> Dict[str, Any]:
        return {
            "device": self.device,
            "vram_gb": round(self.vram_gb, 2),
            "device_name": self.device_name,
            "dtype": str(self.dtype).replace("torch.", ""),
            "batch_size": self.batch_size,
            "grad_accum_steps": self.grad_accum_steps,
            "use_gradient_checkpointing": self.use_gradient_checkpointing,
            "lora_r": self.lora_r,
            "strategy": self.strategy,
        }

    def summary(self) -> str:
        lines = [
            "=== Rev Hardware Plan ===",
            f"Device: {self.device} ({self.device_name})",
            f"VRAM: {self.vram_gb:.1f} GB" if self.vram_gb > 0 else "System Memory Only",
            f"Precision: {str(self.dtype).replace('torch.', '')}",
            f"Batch Size: {self.batch_size} (effective batch {self.batch_size * self.grad_accum_steps})",
            f"Gradient Accumulation: {self.grad_accum_steps} steps",
            f"Gradient Checkpointing: {self.use_gradient_checkpointing}",
            f"Strategy: {self.strategy}",
        ]
        return "\n".join(lines)


class HardwarePlanner:
    """Detects system hardware and generates optimal training profiles."""

    @classmethod
    def plan(
        cls,
        dataset_size: int = 1000,
        preferred_device: Optional[str] = None,
    ) -> HardwareProfile:
        device_str = preferred_device
        vram_gb = 0.0
        device_name = "CPU"

        if device_str is None:
            if torch.cuda.is_available():
                device_str = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_str = "mps"
            else:
                device_str = "cpu"

        if device_str.startswith("cuda") and torch.cuda.is_available():
            dev_idx = 0
            device_name = torch.cuda.get_device_name(dev_idx)
            vram_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
            vram_gb = vram_bytes / (1024 ** 3)

            # Precision selection
            if torch.cuda.is_bf16_supported():
                dtype = torch.bfloat16
            else:
                dtype = torch.float16

            # Memory tiering
            if vram_gb >= 16.0:
                batch_size = 16
                grad_accum_steps = 2
                use_checkpointing = False
                lora_r = 16
            elif vram_gb >= 8.0:
                batch_size = 8
                grad_accum_steps = 4
                use_checkpointing = True
                lora_r = 16
            else:
                batch_size = 4
                grad_accum_steps = 8
                use_checkpointing = True
                lora_r = 8

        elif device_str == "mps":
            device_name = "Apple Silicon GPU (MPS)"
            dtype = torch.float32
            batch_size = 8
            grad_accum_steps = 4
            use_checkpointing = True
            lora_r = 8
        else:
            device_str = "cpu"
            device_name = f"CPU ({os.cpu_count() or 4} cores)"
            dtype = torch.float32
            batch_size = 4
            grad_accum_steps = 4
            use_checkpointing = False
            lora_r = 8

        # Strategy selection based on dataset size
        if dataset_size < 100:
            strategy = "few-shot: LoRA rank 8 + calibrated readout head"
        elif dataset_size > 10000 and vram_gb >= 24.0:
            strategy = "full fine-tune: entire encoder unfreezed"
        else:
            strategy = "parameter-efficient: LoRA encoder projections + calibrated readout head"

        return HardwareProfile(
            device=device_str,
            vram_gb=vram_gb,
            device_name=device_name,
            dtype=dtype,
            batch_size=batch_size,
            grad_accum_steps=grad_accum_steps,
            use_gradient_checkpointing=use_checkpointing,
            lora_r=lora_r,
            strategy=strategy,
        )
