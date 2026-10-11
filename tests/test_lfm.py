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
