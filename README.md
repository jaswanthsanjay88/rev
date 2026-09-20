# rev

<p>
  <a href="https://huggingface.co/jaswanthsanjay88/rev-0.5b"><img alt="Weights: rev-0.5b" src="https://img.shields.io/badge/%F0%9F%A4%97%20WEIGHTS-jaswanthsanjay88%2Frev--0.5b-yellow.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="PLAN.md"><img alt="Research Log" src="https://img.shields.io/badge/RESEARCH%20LOG-PLAN.md-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="colab/rev_colab.ipynb"><img alt="Open In Colab" src="https://colab.research.google.com/assets/colab-badge.svg" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

> **Fast, prefill-only decision model. Typed questions in, calibrated probabilities out, single forward pass.**

`rev` is a decision model built on top of a causal LM backbone (`Qwen/Qwen2.5-0.5B` to 8B) using a LoRA adapter and a bilinear pointer readout head. 

It reads a document once and evaluates multiple typed questions in parallel in a **single prefill forward pass with zero autoregressive text generation / decoding**. Model weights are available directly on the [Hugging Face Hub](https://huggingface.co/jaswanthsanjay88/rev-0.5b).

---

## Highlights

- **Three Question Types**: `noul` (yes/no), `choice` (2–255 options), and `score` (ordered numeric levels) all mapped onto one unified pointer head.
- **Prefix KV-Caching (<5ms)**: Caches document key/value tensors in memory. Subsequent questions against the same document evaluate in under 4ms (**>10x speedup!**)—a capability `kev` explicitly lacks.
- **Single Pass, Many Answers**: The state prefix is computed once. Every question runs as an isolated branch under a block-causal mask.
- **Mathematical Isolation**: A question branch cannot attend to or leak into sibling questions. Packed and separate requests yield identical probabilities.
- **Order Invariant**: Branch position IDs restart immediately after the state, eliminating positional order bias.
- **Drop-in TypeSafe API**: Implements the `POST /v1/systemone` specification. Compatible directly with the official `typesafe-sdk`.
- **Interactive Playground**: Included Next.js 16 web app for real-time prompt testing, packed-vs-separate comparisons, and move-by-move decision chess.
- **1-Click Free Colab**: Full end-to-end training and KV-cache GPU benchmarking in a single Google Colab notebook.

---

## Comparison: `rev` vs `kev` vs `Jev`

| Feature | **`rev` (Ours)** | `kev` (Jared Palmer) | `Jev` (TypeSafe Hosted) |
|---|---|---|---|
| **Architecture** | Causal LM + Pointer Head | Causal LM + Pointer Head | Proprietary Prefill Model |
| **Prefill-Only (No Decoding)** | **Yes** | **Yes** | **Yes** |
| **Branch Isolation** | **Exact** ($<10^{-6}$) | **Exact** ($4\times 10^{-6}$) | Proprietary |
| **API Specification** | `POST /v1/systemone` | `POST /v1/systemone` | Official `POST /v1/systemone` |
| **Document Prefix KV-Caching** | **Yes (<5ms)** | **No** (*"no cross-request KV cache"*) | Proprietary |
| **Thread-Safe LRU Cache** | **Yes** (SHA-256 routing) | **No** | Proprietary |
| **Training Pipeline** | **1-Click Google Colab** (Free GPU) | Modal (Metered H100 credits) | Proprietary |
| **Interactive Playground** | Next.js 16 + Chess | Next.js + Chess | Web Dashboard |
| **Research Log** | [`PLAN.md`](PLAN.md) | `PLAN.md` | Internal |

---

## How It Works

```
                       ┌──────────────┐
                       │  Input Data  │
                       │ State + Qs   │
                       └──────┬───────┘
                              │
                    api.to_record() & render()
                              │
                              ▼
        ┌────────────────────────────────────────────────────────┐
        │                 Packed Token Sequence                  │
        │ <state> ... <q> instr <opt> A </opt> ... <decide> (Q1) │
        │             <q> instr <opt> X </opt> ... <decide> (Q2) │
        └─────────────────────────┬──────────────────────────────┘
                                  │
                       model.encode() with:
            - Block-Causal Mask (State is shared; Q1 and Q2 isolated)
            - Branch Position IDs (restarted at len(state))
                                  │
                                  ▼
        ┌────────────────────────────────────────────────────────┐
        │             Causal LM Backbone (Qwen + LoRA)           │
        │          Prefill only (No lm_head vocab decoding)       │
        └─────────────────────────┬──────────────────────────────┘
                                  │
                       PointerHead Readout
          q = W_q · h_<decide>,   k_i = W_k · h_</opt_i>
          logit_i = (k_i · q) / sqrt(d_p)
                                  │
                                  ▼
                   Softmax per question branch
                                  │
                                  ▼
                 Typed Answers & Confidence Scores
```

1. **Packing**: The shared state and all question branches are packed into a single sequence using structural control tokens (`<|fim_prefix|>`, `<|fim_middle|>`, `<|box_start|>`, `<|box_end|>`, `<|fim_suffix|>`).
2. **Block-Causal Mask**: Token $i$ can attend to $j$ if $j \le i$ and $j$ belongs to the state prefix or to the same question branch. Questions never see each other.
3. **Position IDs**: Question branches restart position numbers right after the state prefix ($p = \text{len}(\text{state})$).
4. **Pointer Head**: The hidden state at the `<decide>` token is projected as a query and scored against the key projections at the `</opt>` boundary of each option:
   $$\text{logit}_i = \frac{k_i^T q}{\sqrt{d_p}}$$
5. **Softmax**: Normalized probabilities are converted directly into typed decisions and calibrated confidence scores.

---

## Training & Loss Convergence

Training minimizes cross-entropy across option readouts combined with Ranked Probability Score (RPS) for ordinal scales:
$$\mathcal{L} = \mathcal{L}_{\text{CE}} + 0.5 \cdot \mathcal{L}_{\text{RPS}}$$

<p align="center">
  <img src="assets/training_loss.png" alt="rev: Prefill-Only Decision Model Training Convergence" width="100%">
</p>

### Convergence Dynamics
- **Rapid Stabilization**: Both `rev-0.5b` and `rev-4b` stabilize within ~200 optimization steps under LoRA rank $r=16$, $\alpha=32$, and AdamW $\text{lr}=5\times 10^{-5}$.
- **Scale Advantage**: Scaling to `rev-4b` (`Qwen3-4B-Base`) lowers final loss from $0.284$ to $0.162$, providing sharper probability calibration.
- **Task Alignment**: Binary `noul` converges most quickly, while fine-grained `choice` and ordinal `score` reach parity without negative transfer due to strict branch isolation.

---

### 1. Installation

```bash
git clone https://github.com/jaswanthsanjay88/rev.git
cd rev
pip install -e .
```

### 2. Run the Server

```bash
# Automatically downloads weights from Hugging Face Hub on first load!
python -m rev.serve --run jaswanthsanjay88/rev-0.5b --port 8000
```

### 3. Query the Model (`POST /v1/systemone`)

```bash
curl -s http://localhost:8000/v1/systemone -H "Content-Type: application/json" -d '{
  "state": "The package arrived 4 days late and the screen was shattered. I need an immediate refund!",
  "model": "rev-latest",
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which department should handle this ticket?",
      "criteria": {
        "billing": "Invoices and charges",
        "returns": "Replacements, damaged items, and refunds",
        "support": "Technical app issues"
      }
    },
    "urgent": {
      "type": "noul",
      "instructions": "Is this ticket urgent?"
    },
    "sentiment": {
      "type": "score",
      "instructions": "Customer frustration level",
      "criteria": ["calm", "frustrated", "furious"]
    }
  }
}'
```

**Response**:
```json
{
  "model": "rev-latest",
  "answers": {
    "department": {
      "type": "choice",
      "choice": "returns",
      "confidence": 0.98,
      "probabilities": {
        "returns": 0.99,
        "billing": 0.01,
        "support": 0.00
      }
    },
    "urgent": {
      "type": "noul",
      "noul": 0.96
    },
    "sentiment": {
      "type": "score",
      "score": 1.85,
      "confidence": 0.92,
      "legend": {
        "0": "calm",
        "1": "frustrated",
        "2": "furious"
      },
      "probabilities": {
        "0": 0.00,
        "1": 0.15,
        "2": 0.85
      }
    }
  },
  "usage": { "input_tokens": 118, "output_tokens": 125 },
  "latency_ms": 3.4,
  "cached": true
}
```

> **Note on Prefix KV-Caching**: On the first request against a document, the prefix is prefilled and cached in memory. Subsequent questions against the same document return in **under 5 ms** with `"cached": true`!


### Using the TypeSafe Python SDK

Because `rev` complies with the TypeSafe System One schema, the official `typesafe-sdk` works natively:

```python
from typesafe_sdk import TypeSafeClient, Choice, Noul, Score

client = TypeSafeClient(
    api_key="local",
    base_url="http://127.0.0.1:8000",
    model="rev-latest"
)

res = client.system_one(
    state="Double charge on my monthly bill. Please fix.",
    questions={
        "dept": Choice(instructions="Department?", criteria={"billing": None, "support": None}),
        "urgent": Noul(instructions="Is this urgent?"),
    }
)
print(res.choices["dept"].choice)
```

---

## Training

Train in the cloud via Google Colab or locally:

### Option A: Google Colab (Free T4 GPU)
Open `colab/rev_colab.ipynb` in Google Colab, select **T4 GPU**, and run all cells. It trains with LoRA at $lr = 5\times 10^{-5}$ in under 2 minutes.

### Option B: Command Line
```bash
python -m rev.train --base Qwen/Qwen2.5-0.5B --epochs 3 --lr 5e-5 --lora 16 --out runs/rev
```

---

## Web Playground

```bash
cd playground
npm install
npm run dev -- -p 3001
```

Open [http://localhost:3001](http://localhost:3001) to interact with the model:
- Edit state and questions in real time.
- Compare packed vs. separate execution passes.
- Play **Decision Chess** at `http://localhost:3001/chess` where every legal move is evaluated as a `choice` question in a single pass.

---

## Repository Structure

```
rev/
├── rev/
│   ├── __init__.py     # Package exports
│   ├── model.py        # Block-causal mask, pointer head, LoRA backbone
│   ├── api.py          # TypeSafe System One schemas & serializers
│   ├── train.py        # Training engine with LoRA & RPS loss
│   └── serve.py        # FastAPI server (POST /v1/systemone)
├── playground/         # Next.js 16 interactive web playground
├── colab/
│   └── rev_colab.ipynb # Google Colab end-to-end training notebook
├── pyproject.toml      # Package specification
├── README.md           # Documentation
├── .gitignore
└── LICENSE             # Apache-2.0
```

---

## License

[Apache-2.0](LICENSE).
