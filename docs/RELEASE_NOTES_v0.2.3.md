# rev v0.2.3: The Unified AutoModel Release (`Rev.from_pretrained`)

This release fundamentally consolidates the entire **rev** ecosystem around the **`AutoModel` / `timm`** design pattern, bringing all text, vision, and causal decision models behind a single universal interface.

---

### 🚨 What's New & Migration Guide

#### 1. Universal Auto-Dispatcher (`from rev import Rev`)
Instead of separate loading classes per backbone, use `Rev.from_pretrained`:

```python
from rev import Rev

# All return the exact same .predict(state, questions) shape!
vision_model = Rev.from_pretrained("jaswanthsanjay88/rev-vision")         # SmolVLM Multimodal (<40ms)
text_model   = Rev.from_pretrained("jaswanthsanjay88/rev-decision-model") # ModernBERT-large (<1ms)
causal_model = Rev.from_pretrained("jaswanthsanjay88/rev-4b")             # Qwen 4B Causal
edge_model   = Rev.from_pretrained("jaswanthsanjay88/rev-0.5b")           # Qwen 0.5B + KV-Cache
mock_model   = Rev.from_pretrained("mock")                                # Zero-weight instant tester
```

#### 2. Standardized `DecisionEngine` Contract
All backbones inherit from `DecisionEngine` and emit identical calibrated probability distributions for `choice`, `noul`, and `score`.

#### 3. Air-Gapped & Offline Mocking
`Rev.from_pretrained("mock")` executes with zero weight downloads and zero Hugging Face auth calls—ideal for unit tests and CI pipelines.

#### 4. Backward Compatibility
- `Agent`, `Model`, and `UnifiedModel` remain fully functional aliases.
- `RevVision = Rev` is re-exported to prevent import breaks.

---

### 📦 Installation
- **Python / PyPI**: `pip install --upgrade rev-decision==0.2.3`
- **Node.js / npm**: `npm install rev-decision@0.2.3`

**Full Changelog**: https://github.com/jaswanthsanjay88/rev/commits/v0.2.3
