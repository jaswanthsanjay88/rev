"""
rev.bundle: Portable .rev model bundle specification and serialization.

A .rev bundle is a standalone, self-describing directory containing:
  - manifest.json: metadata, label map, task schema, data hash, rev version
  - calibration.json: optimal temperature, deferral threshold, cost matrix
  - metrics.json: evaluation scorecard, baseline gain, ECE, coverage curve
  - adapter_model.safetensors / model weights
  - readout_head.pt
  - tokenizer files
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
import torch


class BundleManifest:
    """Manifest describing a saved .rev decision model bundle."""

    def __init__(
        self,
        rev_version: str,
        backbone: str,
        labels: List[str],
        label_to_id: Dict[str, int],
        optimal_temperature: float = 1.0,
        deferral_threshold: float = 0.80,
        cost_matrix: Optional[Dict[str, float]] = None,
        data_hash: Optional[str] = None,
        created_at: Optional[str] = None,
    ):
        self.rev_version = rev_version
        self.backbone = backbone
        self.labels = labels
        self.label_to_id = label_to_id
        self.optimal_temperature = optimal_temperature
        self.deferral_threshold = deferral_threshold
        self.cost_matrix = cost_matrix or {"false_action": 10.0, "defer": 1.0}
        self.data_hash = data_hash or ""
        self.created_at = created_at or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rev_version": self.rev_version,
            "backbone": self.backbone,
            "labels": self.labels,
            "label_to_id": self.label_to_id,
            "optimal_temperature": self.optimal_temperature,
            "deferral_threshold": self.deferral_threshold,
            "cost_matrix": self.cost_matrix,
            "data_hash": self.data_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BundleManifest":
        return cls(
            rev_version=d.get("rev_version", "0.3.0"),
            backbone=d.get("backbone", "jaswanthsanjay88/rev-decision-model"),
            labels=d.get("labels", []),
            label_to_id=d.get("label_to_id", {}),
            optimal_temperature=float(d.get("optimal_temperature", 1.0)),
            deferral_threshold=float(d.get("deferral_threshold", 0.80)),
            cost_matrix=d.get("cost_matrix"),
            data_hash=d.get("data_hash", ""),
            created_at=d.get("created_at"),
        )


def save_bundle(
    target_dir: str,
    manifest: BundleManifest,
    tokenizer: Any,
    backbone: Any,
    readout_head: Any,
    metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves a complete, portable .rev bundle to target_dir."""
    os.makedirs(target_dir, exist_ok=True)

    # 1. Save Manifest
    manifest_path = os.path.join(target_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest.to_dict(), f, indent=2)

    # 2. Save Calibration
    calib_path = os.path.join(target_dir, "calibration.json")
    with open(calib_path, "w", encoding="utf-8") as f:
        json.dump({
            "optimal_temperature": manifest.optimal_temperature,
            "deferral_threshold": manifest.deferral_threshold,
            "cost_matrix": manifest.cost_matrix,
        }, f, indent=2)

    # 3. Save Metrics
    if metrics:
        metrics_path = os.path.join(target_dir, "metrics.json")
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # 4. Save Tokenizer
    if hasattr(tokenizer, "save_pretrained"):
        tokenizer.save_pretrained(target_dir)

    # 5. Save Model / LoRA Adapters
    if hasattr(backbone, "save_pretrained"):
        backbone.save_pretrained(target_dir)

    # 6. Save Readout Head
    if readout_head is not None:
        head_path = os.path.join(target_dir, "readout_head.pt")
        torch.save(readout_head.state_dict(), head_path)

    return target_dir


def load_manifest(bundle_dir: str) -> Optional[BundleManifest]:
    """Loads BundleManifest from bundle_dir if manifest.json is present."""
    p = os.path.join(bundle_dir, "manifest.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return BundleManifest.from_dict(data)
    return None
