"""
Tests for rev client fine-tuning, Dataset validation, hardware planning, and .rev bundle.
"""

import os
import shutil
import pytest
import torch
import torch.nn as nn
from rev import (
    Dataset,
    ValidationReport,
    HardwarePlanner,
    BundleManifest,
    save_bundle,
    load_manifest,
    Rev,
    RevFinetunedModel,
)


def test_dataset_validation_healthy():
    records = [
        {"text": "How do I update my billing credit card?", "label": "billing"},
        {"text": "My payment was rejected twice.", "label": "billing"},
        {"text": "Invoice has an extra charge.", "label": "billing"},
        {"text": "Please refund my last transaction.", "label": "billing"},
        {"text": "Need receipt for business tax deduction.", "label": "billing"},
        {"text": "The mobile app keeps crashing on launch.", "label": "tech"},
        {"text": "Error 500 when accessing account dashboard.", "label": "tech"},
        {"text": "Connection timeout on production API server.", "label": "tech"},
        {"text": "Password reset email is never arriving.", "label": "tech"},
        {"text": "Cannot login on Chrome browser.", "label": "tech"},
    ]
    ds = Dataset.from_records(records)
    assert len(ds) == 10
    assert ds.labels == ["billing", "tech"]

    report = ds.validate()
    assert report.is_valid is True
    assert report.num_classes == 2
    assert report.class_counts == {"billing": 5, "tech": 5}
    assert report.duplicates_count == 0
    assert report.conflicting_labels_count == 0


def test_dataset_validation_detects_label_noise_and_duplicates():
    noisy_records = [
        {"text": "Refund my card", "label": "billing"},
        {"text": "Refund my card", "label": "tech"},  # Conflict: same text, different label
        {"text": "Refund my card", "label": "billing"},  # Duplicate
        {"text": "Server down", "label": "tech"},
    ]
    ds = Dataset.from_records(noisy_records)
    report = ds.validate()
    assert report.conflicting_labels_count == 1
    assert report.duplicates_count == 2
    # Flagged because classes have < 5 samples
    assert report.is_valid is False
    assert any("minimum 5 required" in iss for iss in report.issues)


def test_dataset_stratified_split():
    records = []
    for i in range(20):
        records.append({"text": f"Billing issue number {i}", "label": "billing"})
        records.append({"text": f"Tech bug number {i}", "label": "tech"})

    ds = Dataset.from_records(records)
    train_ds, val_ds, test_ds = ds.split(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42)

    assert len(train_ds) == 32
    assert len(val_ds) == 4
    assert len(test_ds) == 4
    assert train_ds.labels == ["billing", "tech"]
    assert val_ds.labels == ["billing", "tech"]


def test_hardware_planner():
    profile = HardwarePlanner.plan(dataset_size=500)
    assert profile.device in ("cuda", "mps", "cpu")
    assert profile.batch_size in (4, 8, 16, 32)
    assert profile.grad_accum_steps >= 1
    assert "parameter-efficient" in profile.strategy or "few-shot" in profile.strategy


def test_bundle_serialization_and_loading(tmp_path):
    bundle_dir = str(tmp_path / "test-bundle")
    manifest = BundleManifest(
        rev_version="0.3.0",
        backbone="mock-backbone",
        labels=["billing", "tech"],
        label_to_id={"billing": 0, "tech": 1},
        optimal_temperature=1.25,
        deferral_threshold=0.85,
    )

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 1)

    head = DummyHead()
    save_bundle(
        target_dir=bundle_dir,
        manifest=manifest,
        tokenizer=None,
        backbone=None,
        readout_head=head,
    )

    loaded_manifest = load_manifest(bundle_dir)
    assert loaded_manifest is not None
    assert loaded_manifest.backbone == "mock-backbone"
    assert loaded_manifest.labels == ["billing", "tech"]
    assert loaded_manifest.optimal_temperature == 1.25
    assert loaded_manifest.deferral_threshold == 0.85


def test_finetuned_model_predict_and_deferral():
    class DummyBackbone(nn.Module):
        def forward(self, input_ids=None, attention_mask=None, **kwargs):
            bsz = input_ids.shape[0] if input_ids is not None else 2
            # Return dummy hidden state with shape [batch, seq_len, 16]
            hidden = torch.ones(bsz, 4, 16)
            mask = torch.ones(bsz, 4)
            return type("Output", (), {"last_hidden_state": hidden, "attention_mask": mask})()

    class DummyTokenizer:
        def __call__(self, texts, **kwargs):
            return {
                "input_ids": torch.ones(len(texts), 4, dtype=torch.long),
                "attention_mask": torch.ones(len(texts), 4, dtype=torch.long),
            }

    class DummyHead(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(16, 1)

        def forward(self, x):
            # Output biased logits: first option higher
            return torch.tensor([5.0, 1.0])

    model = RevFinetunedModel(
        backbone=DummyBackbone(),
        tokenizer=DummyTokenizer(),
        readout_head=DummyHead(),
        labels=["billing", "tech"],
        optimal_temperature=1.0,
        deferral_threshold=0.75,
    )

    res = model.predict("I want a refund on my credit card")
    assert res["decision"] == "billing"
    assert res["confidence"] > 0.90
    assert res["action"] == "act"
    assert res["deferral_reason"] is None

    # Test deferral when confidence is low
    model.deferral_threshold = 0.999
    res_def = model.predict("I want a refund on my credit card")
    assert res_def["action"] == "defer"
    assert "below deferral threshold" in res_def["deferral_reason"]
