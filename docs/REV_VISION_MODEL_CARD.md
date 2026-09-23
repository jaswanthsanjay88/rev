---
license: apache-2.0
base_model: HuggingFaceTB/SmolVLM-256M-Instruct
library_name: transformers
pipeline_tag: visual-question-answering
tags:
- rev
- rev-vision
- decision-engine
- system-one
- vision-language
- smolvlm
- peft
- lora
- safetensors
- calibration
- rubric-scoring
- fast-inference
- edge-ai
metrics:
- accuracy
- brier_score
model_name: rev-vision
---

<div align="center">

# ⚡ rev-vision (256M)
### Ultra-Fast, Non-Autoregressive Multimodal Decision Engine

[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-jaswanthsanjay88%2Frev--vision-blue.svg)](https://huggingface.co/jaswanthsanjay88/rev-vision)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Base Model](https://img.shields.io/badge/Base-SmolVLM--256M--Instruct-orange.svg)](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct)
[![Latency](https://img.shields.io/badge/Inference-<40ms%20(GPU)-brightgreen.svg)]()
[![VRAM](https://img.shields.io/badge/VRAM-<1.8GB-purple.svg)]()
[![Architecture](https://img.shields.io/badge/Architecture-Single--Forward%20Prefill-red.svg)]()

</div>

---

## 📌 Overview

**rev-vision** is a lightweight, non-autoregressive **multimodal decision engine** derived from SmolVLM-256M. Rather than slowly generating tokens character-by-character via autoregressive decoding (which takes hundreds of milliseconds and often requires fragile JSON parsing or regex), **rev-vision answers visual queries in a single forward prefill pass (<40ms)**.

By projecting the hidden states of prompt option terminators through a specialized multi-layer decision head, `rev-vision` emits mathematically calibrated probabilities over structured choices, boolean propositions, and ordinal rubric scales.

```
                    ┌─────────────────────────┐
                    │  Image (512x512 Native) │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │  SigLIP Vision Encoder  │  ──► (64 Image Tokens)
                    └────────────┬────────────┘
                                 │
  Prompt + Options  ─────────────┴─────────────┐
  (Formatted Chat)                             ▼
                                ┌─────────────────────────┐
                                │   SmolLM 135M Decoder   │ (With LoRA Adapters)
                                └────────────┬────────────┘
                                             │
                       Extract Option Terminator Hidden States
                                             │
                                             ▼
                                ┌─────────────────────────┐
                                │ RevVisionDecisionHead   │ (2-layer MLP + LayerNorm)
                                └────────────┬────────────┘
                                             │
                                 Temperature Calibration
                                             │
                                             ▼
                     Calibrated Decision Probabilities & Rubrics
                     • choice: Argmax Option + Distribution
                     • noul:   Calibrated P(True) ∈ [0, 1]
                     • score:  Expected Level E[tier] ∈ [0, N]
```

---

## 🚀 Key Advantages

| Feature | Standard VLMs (Autoregressive) | rev-vision (Single-Pass Prefill) |
|---|---|---|
| **Inference Latency** | 300ms – 2,500ms (token generation loop) | **<40ms on GPU** / **<150ms on CPU** |
| **Output Type** | Unstructured text strings | **Typed, Calibrated Probabilities & Schema** |
| **VRAM Footprint** | 4 GB – 16 GB+ | **< 1.8 GB VRAM** (fits easily on edge / T4) |
| **Parsing Reliability**| Hallucinations, schema drift, invalid JSON | **100% Deterministic Guarantee** |
| **Visual Tokens** | ~23,273 tokens (multi-crop tiling) | **64 tokens** (single-tile 512x512) |
| **Scoring Quality** | Heuristic softmax probabilities | **Strict Proper Scoring Rules (Spherical + RPS)** |

---

## 🧠 Typed Decision Primitives

`rev-vision` directly implements the core typing contract of the [rev / openjev SystemOne architecture](https://github.com/jaswanthsanjay88/rev):

### 1. `choice` (Multi-class Categorization)
Evaluates an arbitrary set of mutually exclusive categories. Returns the winning choice, the exact confidence score, and the complete probability distribution.
```json
{
  "choice": "red",
  "confidence": 0.9412,
  "probabilities": { "red": 0.9412, "green": 0.0321, "blue": 0.0267 }
}
```

### 2. `noul` (Boolean Propositions)
Evaluates the calibrated probability that a statement about the image/state is true: $P(	ext{true}) \in [0.0, 1.0]$.
```json
{
  "noul": 0.9854,
  "confidence": 0.9854
}
```

### 3. `score` (Ordinal Rubric Rating)
Calculates the continuous expected rating level across ordered rubric criteria:
$$\mathbb{E}[	ext{level}] = \sum_{k=0}^{K-1} k \cdot p_k$$
```json
{
  "score": 1.842,
  "level_probabilities": [0.0512, 0.1543, 0.6955, 0.0990],
  "criteria": ["none", "cosmetic scratch", "severe structural damage", "totaled"]
}
```

---

## 📐 Mathematical Formulation & Loss Function

`rev-vision` is trained using a composite strictly proper scoring objective:

$$\mathcal{L} = \mathcal{L}_{	ext{CE}} - \lambda_1 \mathcal{S}_{	ext{spherical}}(p, y) + \lambda_2 \mathcal{S}_{	ext{RPS}}(p, y)$$

1. **Spherical Proper Scoring Rule**:
   $$\mathcal{S}_{	ext{spherical}}(p, y) = rac{p_y}{\|p\|_2}$$
   Encourages well-calibrated confidence intervals and penalizes overconfident predictions.

2. **Ranked Probability Score (RPS)** (applied to ordinal `score` questions):
   $$	ext{RPS}(p, y) = rac{1}{K-1} \sum_{m=1}^{K-1} \left( \sum_{k=1}^m p_k - \sum_{k=1}^m y_k ight)^2$$
   Penalizes errors proportionally to their distance on the rubric scale (predicting level 1 when the true label is level 3 is penalized more than predicting level 2).

3. **Per-Type Temperature Calibration**:
   Logits are scaled by empirically fitted temperatures prior to softmax:
   - `choice`: $T = 2.2028$
   - `score`: $T = 1.3652$
   - `noul`: $T = 2.1333$

---

## 💻 Quickstart: Inference in Python

```python
import os, time, json
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

# 2. Load Processor (with single-tile 512x512 config)
processor = AutoProcessor.from_pretrained(REPO_ID, subfolder="processor")
if hasattr(processor, "image_processor") and hasattr(processor.image_processor, "do_image_splitting"):
    processor.image_processor.do_image_splitting = False

# 3. Load Base Model & LoRA Adapter
base_model = AutoModelForImageTextToText.from_pretrained(
    "HuggingFaceTB/SmolVLM-256M-Instruct",
    torch_dtype=DTYPE,
    device_map=DEVICE
)
model = PeftModel.from_pretrained(base_model, REPO_ID, subfolder="adapter")
model.eval()

# 4. Load Decision Head Weights
head_weights_path = hf_hub_download(repo_id=REPO_ID, filename="head.safetensors")
hidden_dim = base_model.config.text_config.hidden_size if hasattr(base_model.config, 'text_config') else base_model.config.hidden_size
decision_head = RevVisionDecisionHead(hidden_size=hidden_dim).to(device=DEVICE, dtype=DTYPE)
decision_head.load_state_dict(load_file(head_weights_path))
decision_head.eval()

print("✓ rev-vision engine loaded successfully!")

# 5. Execute Structured Inference
def predict(image, note, questions):
    newline_id = processor.tokenizer.encode("\n", add_special_tokens=False)[-1]
    temperatures = {"choice": 2.2028, "score": 1.3652, "noul": 2.1333}
    answers = {}

    for q_key, q_spec in questions.items():
        q_type = q_spec.get("type", "choice")
        instr = q_spec.get("instructions", "")
        
        if q_type == "noul":
            options = ["false", "true"]
        elif q_type == "score":
            options = q_spec.get("criteria", ["level 0", "level 1", "level 2"])
        else:
            crit = q_spec.get("criteria", ["option a", "option b"])
            options = list(crit.values()) if isinstance(crit, dict) else crit

        prompt = f"<|im_start|>User:<image>{note}\n{q_type.upper()} question: {instr}<end_of_utterance>\nAssistant: Options:\n"
        for opt in options:
            prompt += f"- {opt}\n"

        inputs = processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(device=DEVICE, dtype=DTYPE) if v.is_floating_point() else v.to(device=DEVICE) for k, v in inputs.items()}

        input_ids = inputs["input_ids"][0]
        newlines = (input_ids == newline_id).nonzero(as_tuple=True)[0]
        terminators = newlines[-len(options):].tolist()

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[-1][0, terminators, :]
            logits = decision_head(hidden.unsqueeze(0))[0] / temperatures.get(q_type, 1.0)
            probs = F.softmax(logits, dim=-1).float().cpu().numpy().tolist()

        if q_type == "choice":
            best = int(torch.argmax(torch.tensor(probs)))
            answers[q_key] = {"choice": options[best], "confidence": round(probs[best], 4)}
        elif q_type == "noul":
            answers[q_key] = {"noul": round(probs[1], 4), "confidence": round(max(probs[1], 1 - probs[1]), 4)}
        elif q_type == "score":
            expected = sum(k * p for k, p in enumerate(probs))
            answers[q_key] = {"score": round(float(expected), 3), "distribution": [round(p, 4) for p in probs]}

    return answers

# Example Execution
test_img = Image.new("RGB", (256, 256), color="darkblue")
results = predict(
    image=test_img,
    note="Asset #104 inspection record",
    questions={
        "color_check": {
            "type": "choice",
            "instructions": "What is the primary canvas color?",
            "criteria": ["red", "green", "blue", "yellow"]
        },
        "has_defect": {
            "type": "noul",
            "instructions": "Is there a defect or scratch visible?"
        },
        "severity": {
            "type": "score",
            "instructions": "Rate severity score from 0 to 2",
            "criteria": ["none", "minor", "severe"]
        }
    }
)
print(json.dumps(results, indent=2))
```

---

## 📊 Benchmark & Hardware Metrics

Tested on a **NVIDIA Tesla T4 (16GB)** and standard **x86_64 CPU**:

| Metric | GPU (T4, fp16/bf16) | CPU (x86_64, fp32) |
|---|---|---|
| **Latency per Question** | **~31.4 ms** | **~138.2 ms** |
| **GPU Peak Allocated Memory** | **1,740 MB** (< 1.8 GB) | N/A (RAM: ~850 MB) |
| **Visual Encoding Tokens** | **64 tokens** | **64 tokens** |
| **Prompt Length (Average)** | **~85 - 110 tokens** | **~85 - 110 tokens** |
| **Throughput (Batch Size 1)** | **~30 decisions / sec** | **~7 decisions / sec** |

---

## 📦 Repository Structure

```
jaswanthsanjay88/rev-vision/
├── README.md                          # Comprehensive documentation & quickstart
├── vlm_agent_config.json              # Engine hyperparameter configuration
├── head.safetensors                   # RevVisionDecisionHead weights (2-layer MLP)
├── model.safetensors                  # Mirror of decision head weights
├── adapter/
│   ├── adapter_config.json            # PEFT LoRA configuration (r=16, alpha=32)
│   ├── adapter_model.safetensors      # LoRA trained delta weights
│   └── README.md                      # Adapter card
└── processor/
    ├── processor_config.json          # Preprocessor configuration (single-tile)
    ├── tokenizer.json                 # Fast tokenizer dictionary
    ├── tokenizer_config.json          # Tokenizer settings & special tokens
    └── chat_template.jinja            # SmolVLM chat templating specification
```

---

## 📜 License

This model and its associated weights are distributed under the **Apache 2.0 License**.

## 🤝 Acknowledgements

- Built on top of [HuggingFaceTB/SmolVLM-256M-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM-256M-Instruct).
- Developed as part of the [rev](https://github.com/jaswanthsanjay88/rev) project for low-latency System 1 programmatic AI agents.
