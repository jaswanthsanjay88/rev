---
license: apache-2.0
base_model: HuggingFaceTB/SmolVLM-256M-Instruct
pipeline_tag: image-text-to-text
library_name: transformers
tags:
- vision
- multimodal
- decision-engine
- non-autoregressive
- classification
- structured-output
- agent-tools
- function-calling
---

# rev-vision

**rev-vision** is a lightweight, non-autoregressive multimodal decision engine derived from [SmolVLM-256M](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct). Instead of generating an answer token-by-token, it runs a single forward prefill pass (<40ms) and reads calibrated probabilities directly off the hidden states of your prompt's option terminators.

This makes it a purpose-built decision head for agents and pipelines that need a fast, reliable, machine-readable verdict from an image + question — not a chat response to parse.

Part of the **rev** decision-engine family. See the text-only sibling: [rev-decision](https://huggingface.co/jaswanthsanjay88/rev-decision-model).

---

## Why rev-vision instead of prompting a VLM normally

| | Standard VLM prompting | rev-vision |
|---|---|---|
| **Decoding** | Autoregressive, token-by-token | Single forward pass |
| **Latency** | Hundreds of ms+ | < 40ms |
| **Output** | Free-text, needs JSON/regex parsing | Calibrated probability distribution |
| **Failure mode** | Malformed JSON, refusals, verbosity | None — it's a projection, not generation |
| **Best for** | Open-ended conversation | Boolean checks, multiple-choice, rubric scoring |

rev-vision projects the hidden states at each option's terminator token through a small multi-layer decision head, producing calibrated probabilities over:
- **Boolean propositions** — yes/no, true/false, pass/fail
- **Structured choices** — multiple-choice / classification labels
- **Ordinal rubric scales** — e.g. quality score 1–5, severity low/medium/high

---

## Installation

```bash
pip install rev-decision
```

Or install from source:

```bash
git clone https://github.com/jaswanthsanjay88/rev
cd rev
pip install -e .
```

---

## Quickstart

```python
from rev import Rev  # (or `from rev import RevVision`)

model = Rev.from_pretrained("jaswanthsanjay88/rev-vision")

result = model.predict(
    image="receipt.jpg",
    prompt="Is this receipt from a restaurant?",
    options=["yes", "no"],
)
print(result)
# {
#   "answer": "yes",
#   "probabilities": {"yes": 0.94, "no": 0.06},
#   "latency_ms": 27.4
# }
```

### Ordinal / rubric scoring

```python
result = model.predict(
    image="product_photo.jpg",
    prompt="Rate the packaging condition.",
    options=["damaged", "acceptable", "good", "excellent"],
)
```

### Multiple-choice / classification

```python
result = model.predict(
    image="dashboard_screenshot.png",
    prompt="Which alert state is shown?",
    options=["normal", "warning", "critical", "unknown"],
)
```

---

## Guide for Agents (tool / function-calling integration)

rev-vision is designed to sit behind a tool call in an agent loop — the agent hands it an image and a closed set of options, and gets back a typed, calibrated result it can branch on directly, with no parsing step.

### 1. Recommended tool schema

Expose it to your agent framework (OpenAI-style function calling, MCP tool, etc.) like this:

```json
{
  "name": "rev_vision_decide",
  "description": "Answer a visual yes/no, multiple-choice, or rubric question about an image with a calibrated probability distribution. Use this instead of asking a general-purpose VLM when the answer must be one of a fixed set of options and you need speed + a confidence score, not prose.",
  "parameters": {
    "type": "object",
    "properties": {
      "image": {
        "type": "string",
        "description": "Path, URL, or base64-encoded image to evaluate."
      },
      "prompt": {
        "type": "string",
        "description": "The question to answer about the image."
      },
      "options": {
        "type": "array",
        "items": {
          "type": "string"
        },
        "description": "Closed set of allowed answers (2 or more). Order does not affect calibration."
      }
    },
    "required": ["image", "prompt", "options"]
  }
}
```

### 2. Minimal tool-server wrapper

```python
from rev import Rev

model = Rev.from_pretrained("jaswanthsanjay88/rev-vision")

def rev_vision_decide(image: str, prompt: str, options: list[str]) -> dict:
    result = model.predict(image=image, prompt=prompt, options=options)
    return {
        "answer": result["answer"],
        "confidence": max(result["probabilities"].values()),
        "probabilities": result["probabilities"],
    }
```

### 3. When your agent should call this tool vs. a general VLM

**Call rev-vision when:**
- The valid answers form a fixed, known set (booleans, categories, rubric levels).
- You need a confidence score to decide whether to defer to a human or another model.
- Latency matters (real-time UI checks, high-volume batch triage, gating steps before a more expensive call).

**Fall back to a general-purpose VLM when:**
- The task needs free-form description, reasoning explanation, or open-ended generation.
- The option set can't be enumerated ahead of time.

### 4. Confidence-gated agent pattern

```python
result = rev_vision_decide(image, "Does this invoice total exceed $500?", ["yes", "no"])

if result["confidence"] < 0.65:
    # low-confidence — escalate to a larger VLM or a human reviewer
    escalate(image, result)
else:
    act_on(result["answer"])
```

### 5. Batch / pipeline usage

```python
results = model.predict_batch(
    images=["frame_001.png", "frame_002.png", "frame_003.png"],
    prompt="Is a person visible in this frame?",
    options=["yes", "no"],
)
```

---

## Direct Inference via Transformers & PEFT

If using raw Hugging Face libraries without the high-level `rev-vision` wrapper:

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from peft import PeftModel
from safetensors.torch import load_file
from huggingface_hub import hf_hub_download

REPO_ID = "jaswanthsanjay88/rev-vision"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

# 1. Define Decision Head
class RevVisionDecisionHead(nn.Module):
    def __init__(self, hidden_size: int = 576, head_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, head_hidden),
            nn.GELU(),
            nn.LayerNorm(head_hidden),
            nn.Linear(head_hidden, 1)
        )
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.net(hidden_states).squeeze(-1)

# 2. Processor with single-tile configuration (64 visual tokens)
processor = AutoProcessor.from_pretrained(REPO_ID, subfolder="processor")
if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "do_image_splitting"):
    processor.image_processor.do_image_splitting = False

# 3. Base SmolVLM + Trained LoRA Adapter
base_model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=DTYPE,
    device_map=DEVICE
)
model = PeftModel.from_pretrained(base_model, REPO_ID, subfolder="adapter")
model.eval()

# 4. Load Decision Head
head_weights_path = hf_hub_download(repo_id=REPO_ID, filename="head.safetensors")
hidden_dim = base_model.config.text_config.hidden_size if hasattr(base_model.config, 'text_config') else base_model.config.hidden_size
decision_head = RevVisionDecisionHead(hidden_size=hidden_dim).to(device=DEVICE, dtype=DTYPE)
decision_head.load_state_dict(load_file(head_weights_path))
decision_head.eval()
```

---

## Architecture

- **Backbone**: SmolVLM-256M (SigLIP vision encoder + SmolLM 135M decoder)
- **Decision head**: Multi-layer projection head reading hidden states at each option-terminator token position
- **Inference**: Single forward prefill pass — no autoregressive decoding, no sampling
- **Output**: Calibrated softmax distribution over the supplied option set

This mirrors the design of the text-only [rev-decision](https://huggingface.co/jaswanthsanjay88/rev-decision-model) model, extended to accept image input alongside text.

---

## Limitations

- Requires the option set to be specified up front — it does not generate open-ended answers.
- Calibration quality depends on how close the input distribution is to the training/fine-tuning data; out-of-domain images may need re-calibration.
- Not a substitute for a general VLM on tasks requiring explanation or free-text description.

> *Note: Package name, PyPI link, and GitHub URL above (`rev-vision`, `jaswanthsanjay88/rev-vision`) match the rev-decision naming convention.*

---

## Citation

```bibtex
@misc{revvision2026,
  author = {Nekkanti Jaswanth Sanjay},
  title = {rev-vision: A Non-Autoregressive Multimodal Decision Engine},
  year = {2026},
  publisher = {Hugging Face},
  howpublished = {\url{https://huggingface.co/jaswanthsanjay88/rev-vision}}
}
```

---

## Links

- **Model**: [https://huggingface.co/jaswanthsanjay88/rev-vision](https://huggingface.co/jaswanthsanjay88/rev-vision)
- **Text-only sibling (rev-decision)**: [https://huggingface.co/jaswanthsanjay88/rev-decision-model](https://huggingface.co/jaswanthsanjay88/rev-decision-model)
- **rev-decision on PyPI**: [https://pypi.org/project/rev-decision/](https://pypi.org/project/rev-decision/)
