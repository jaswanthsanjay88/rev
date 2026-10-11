import pytest
from rev.auto import _detect_architecture, Rev
from rev.adapters.lfm import LfmReadoutHead
import torch

def test_lfm_architecture_detection():
    assert _detect_architecture("LiquidAI/LFM2.5-Encoder-350M") == "lfm"
    assert _detect_architecture("LiquidAI/LFM2.5-350M") == "lfm"
    assert _detect_architecture("jaswanthsanjay88/rev-lfm") == "lfm"
    assert _detect_architecture("liquid-decision") == "lfm"

def test_lfm_readout_head():
    head = LfmReadoutHead(hidden_size=1024, head_dim=256)
    # Test tensor of 4 options with hidden_size 1024
    dummy_hidden = torch.randn(4, 1024)
    logits = head(dummy_hidden)
    assert logits.shape == (4,)
    probs = torch.softmax(logits, dim=-1)
    assert probs.shape == (4,)
    assert torch.isclose(probs.sum(), torch.tensor(1.0), atol=1e-5)


def test_lfm_finetuned_checkpoint_detection_and_head():
    import os
    if os.path.exists("rev_lfm_finetuned"):
        assert _detect_architecture("rev_lfm_finetuned") == "lfm"
        # Verify saved readout_head.pt matches LfmReadoutHead architecture if present
        head_path = os.path.join("rev_lfm_finetuned", "readout_head.pt")
        if os.path.exists(head_path):
            weights = torch.load(head_path, map_location="cpu")
            head = LfmReadoutHead(hidden_size=1024, head_dim=256)
            head.load_state_dict(weights)
            assert head.proj[0].weight.shape == (256, 1024)
