# rev: Fast Prefill-Only Decision Engine

<p align="center">
  <a href="https://pypi.org/project/rev-decision/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/rev-decision.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://www.npmjs.com/package/rev-decision"><img alt="npm version" src="https://img.shields.io/npm/v/rev-decision.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://pypi.org/project/rev-decision/"><img alt="Python Versions" src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://github.com/jaswanthsanjay88/rev/actions/workflows/ci.yml"><img alt="CI Status" src="https://github.com/jaswanthsanjay88/rev/actions/workflows/ci.yml/badge.svg?style=for-the-badge" height="28"></a>
  <a href="https://pepy.tech/project/rev-decision"><img alt="Downloads" src="https://img.shields.io/badge/pepy-downloads-orange.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="https://huggingface.co/jaswanthsanjay88/rev-decision-model"><img alt="HF Model" src="https://img.shields.io/badge/%F0%9F%A4%97%20MODEL-rev--decision--model-blue.svg?style=for-the-badge&labelColor=000000" height="28"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0a0a0a.svg?style=for-the-badge&labelColor=000000" height="28"></a>
</p>

> **Typed questions in, mathematically calibrated probabilities out — single prefill pass, zero token generation.**

---

## 1. The "Why": Why `rev` Instead of an LLM Call?

Routing incoming requests, enforcing safety guardrails, classifying support tickets, and evaluating agent traces should not cost 800 milliseconds and 50 cents per call. Generic LLMs (like GPT-4o-mini or local LLaMA-3) require multi-token autoregressive decoding to output JSON, exposing production pipelines to parsing errors, hallucinated schemas, and high API costs. 

`rev` is a **System 1 decision engine**. It reads a document once and evaluates multiple typed questions concurrently in a **single forward pass**. By pooling hidden states at delimiter tokens through a bilinear readout head, `rev` delivers sub-millisecond execution, zero decoding overhead, 100% deterministic schema conformance, and probabilities calibrated via strictly proper scoring rules (ECE < 0.09). It auto-routes across 26 Unicode scripts with zero prompt engineering, slashing decision latency by over 90% while cutting inference costs by 100x.

---

## 2. 30-Second Quickstart

### Installation

**Python (PyPI):**
```bash
pip install rev-decision
```

**TypeScript / JavaScript (npm, pnpm, yarn):**
```bash
# npm
npm install rev-decision

# pnpm
pnpm add rev-decision

# yarn
yarn add rev-decision
```

> [!NOTE]
> When testing Python locally, name your script `app.py` or `main.py` (avoid naming your file `rev.py` so Python imports the library rather than your script).

### Code Examples

#### Python (PyPI)

```python
import rev

# 1. Provide an input state and evaluate multiple typed questions in parallel
state = "The package arrived 4 days late and the screen was completely shattered. I demand an immediate refund!"

# 2. Evaluate using pre-built enterprise schemas (or custom dicts)
res = rev.predict(
    state=state,
    questions=rev.presets.triage_questions(),
)

# 3. Access calibrated outputs and routing metadata
print(res["answers"]["intent"]["choice"])          # -> "refund"
print(res["answers"]["intent"]["confidence"])      # -> 0.984
print(res["answers"]["frustration"]["score"])      # -> 2.85 (out of 3.0)
print(res["answers"]["refund_requested"]["noul"])  # -> 0.992
print(res["routing"]["model"])                     # -> "english" (ModernBERT-large)
```

#### TypeScript / JavaScript (npm / pnpm)

```typescript
import { rev } from "rev-decision";

const res = await rev.predict({
  state: "Customer received broken parcel and demands refund.",
  questions: {
    intent: {
      type: "choice",
      instructions: "Determine customer intent",
      criteria: {
        refund: "Customer demands money back",
        support: "General inquiry",
      },
    },
    urgent: {
      type: "noul",
      instructions: "Does this require immediate human escalation?",
    },
  },
});

console.log(res.answers.intent.choice);      // "refund"
console.log(res.answers.intent.confidence);  // 0.984
console.log(res.answers.urgent.noul);        // 0.96
console.log(`Latency: ${res.latency_ms} ms`);
```

**Output Payload:**
```json
{
  "answers": {
    "intent": {
      "type": "choice",
      "choice": "refund",
      "confidence": 0.984,
      "probabilities": { "refund": 0.984, "technical_help": 0.012, "other": 0.004 }
    },
    "frustration": {
      "type": "score",
      "score": 2.85,
      "confidence": 0.91,
      "legend": { "0": "calm", "1": "concerned", "2": "annoyed", "3": "furious" }
    },
    "refund_requested": {
      "type": "noul",
      "noul": 0.992
    }
  },
  "routing": {
    "model": "english",
    "reason": "English Latin text -> routed to ModernBERT-large"
  }
}
```

---

## 3. Scannable Key Features

- **Unified 'One Model' Architecture**: Combines `ModernBERT-large` (421M, English) and `mmBERT-base` (322M, 100+ languages) behind a clean single entry point (`rev.predict()`).
- **Sub-Microsecond Script Routing (`rev.lang`)**: Zero-dependency Unicode script detector covering 26 non-Latin scripts (Devanagari, Cyrillic, Arabic, Han, Hangul, etc.) in $<1\ \mu\text{s}$, eliminating language collapse without manual configuration.
- **Single-Pass Option-Marker Pooling**: Extracts probabilities directly from delimiter hidden states ($q^T k_i / \sqrt{d}$) with exact block-causal question isolation ($<10^{-6}$ mathematical variance). Zero autoregressive decoding.
- **Document Prefix KV-Caching (<4ms)**: Caches document key/value activations via thread-safe SHA-256 LRU cache. Repeated questions against the same document return in **under 4ms (>10x speedup)**.
- **Strictly Proper Scoring Rules (RLCD)**: Combines Logarithmic Score + Spherical Score + Ranked Probability Score (RPS) directly penalizing overconfidence (ECE 0.081 vs Jev's 0.246).
- **Coarse-to-Fine Candidate Shortlisting (`rev.shortlist`)**: Bi-encoder similarity filtering handles high-cardinality questions (20–255+ options) before cross-attentive scoring.
- **Turnkey Production Presets (`rev.presets`)**: 8 battle-tested enterprise schemas for ticket triage, inbound email security, LLM guardrails, moderation, invoice matching, SOC alerts, and agent traces.
- **Drop-In TypeSafe API**: Native compliance with the `POST /v1/systemone` specification. Fully compatible with `typesafe-sdk`.

---

## 4. Benchmarks Against Baselines

Evaluated on a standard enterprise workload (450-token document evaluated against 3 concurrent typed questions):

| Metric | **`rev` (Ours)** | `TypeSafe Jev` (Hosted) | `kev` (Palmer) | `GPT-4o-mini` | `LLaMA-3-8B-Instruct` |
|---|---|---|---|---|---|
| **Architecture** | **Bidirectional Encoders + Causal KV** | Proprietary Cloud | Causal LM Only (Qwen) | Autoregressive Decoder | Autoregressive Decoder |
| **Model Size** | **322M – 500M** | Proprietary | 500M – 8B | ~8B (est.) | 8.03B |
| **Execution Mode** | **Prefill-only (0 tokens generated)** | Cloud API | Prefill-only | Autoregressive JSON decoding | Autoregressive JSON decoding |
| **Cold Latency (p50 / p95)** | **18.4 ms / 32.8 ms** | 185 ms / 276 ms | 31.2 ms / 48.6 ms | 420 ms / 780 ms | 260 ms / 510 ms |
| **Warm Cached Latency (p50)** | **3.4 ms (<5 ms)** | Not supported | Not supported | Not supported | 120 ms (KV prefill) |
| **Throughput (qps / A100)** | **1,240 decisions/sec** | Hosted rate limits | 410 decisions/sec | Rate-limited | 85 decisions/sec |
| **Expected Calibration Error (ECE)** | **0.081** | 0.246 | 0.182 | 0.294 | 0.231 |
| **Cost per 1M Decisions** | **$0.12** (self-hosted) | $15.00 – $40.00 | $0.25 (self-hosted) | $6.20 | $3.80 (cloud GPU) |
| **Malformed JSON Failure Rate** | **0.000% (Guaranteed)** | 0.000% | 0.000% | 0.84% | 1.92% |
| **Multilingual Script Support** | **26 scripts (Zero collapse)** | English-only | Degraded | High (via token spend) | Moderate |

*Detailed methodology, ablation tests, and replication commands are documented in [docs/benchmarks.md](docs/benchmarks.md).*

---

## 5. CLI Usage & Self-Hosted Server

`rev` includes a production-ready HTTP server implementing the TypeSafe `POST /v1/systemone` specification:

### Starting the Server

```bash
# Using the installed CLI entry point
rev-server --port 8000

# Or using the Python module
python -m rev.serve --port 8000 --host 0.0.0.0
```

### Querying the Server (`POST /v1/systemone`)

```bash
curl -X POST http://localhost:8000/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "Cancel my account immediately. I was charged twice for the enterprise license.",
    "questions": {
      "department": {
        "type": "choice",
        "instructions": "Which department handles this?",
        "criteria": { "billing": "Billing and refund requests", "tech": "Technical support" }
      },
      "urgent": {
        "type": "noul",
        "instructions": "Is this urgent?"
      }
    }
  }'
```

---

## 6. Enterprise Presets

Import turnkey production schemas directly from `rev.presets`:

```python
import rev

# Support ticket triage
triage = rev.presets.triage_questions()

# Inbound email routing & phishing filtering
email = rev.presets.email_questions()

# Real-time LLM prompt guardrails (jailbreaks, injections, PII)
guard = rev.presets.guard_questions()

# Content safety and policy moderation
moderation = rev.presets.moderation_questions()

# Accounts payable invoice discrepancy matching
invoice = rev.presets.invoice_questions()

# Security Operations Center (SOC) incident response
security = rev.presets.security_questions()

# Autonomous agent trace evaluation and guardrail gating
observability = rev.presets.observability_questions()
```

*Full parameter descriptions and schema details are available in [docs/presets.md](docs/presets.md).*

---

## 7. Documentation Directory

- **[docs/api_reference.md](docs/api_reference.md)**: Complete signature reference for `rev.predict`, `rev.Model`, `rev.shortlist_choice`, `rev.proper_reward`, and data schemas.
- **[docs/presets.md](docs/presets.md)**: In-depth schemas and usage guides for all 8 enterprise presets.
- **[docs/benchmarks.md](docs/benchmarks.md)**: Latency profiling, ECE calibration comparisons, and replication scripts.
- **[docs/limitations.md](docs/limitations.md)**: Transparent failure modes, high-cardinality trade-offs, and boundary conditions.
- **[CHANGELOG.md](CHANGELOG.md)**: Version history following Keep a Changelog.
- **[SECURITY.md](SECURITY.md)**: Security vulnerability disclosure policy.

---

## 8. Citation & Links

- **PyPI Package (Python)**: [https://pypi.org/project/rev-decision/](https://pypi.org/project/rev-decision/)
- **npm Package (TypeScript / JavaScript)**: [https://www.npmjs.com/package/rev-decision](https://www.npmjs.com/package/rev-decision)
- **Hugging Face Decision Model**: [https://huggingface.co/jaswanthsanjay88/rev-decision-model](https://huggingface.co/jaswanthsanjay88/rev-decision-model)
- **Hugging Face Weights**: [https://huggingface.co/jaswanthsanjay88/rev-0.5b](https://huggingface.co/jaswanthsanjay88/rev-0.5b)
- **Interactive Playground**: [playground/](playground/) (Next.js 16 Web App with Decision Chess)
- **Training Notebook**: [notebooks/train_laya_system_one_decision_model.ipynb](notebooks/train_laya_system_one_decision_model.ipynb)

If you use `rev` in your research or production systems, please cite:

```bibtex
@software{rev_decision_2026,
  author = {Jaswanth Sanjay},
  title = {rev: Fast Prefill-Only Decision Engine with Sub-Microsecond Multi-Backbone Routing},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jaswanthsanjay88/rev}},
  version = {0.2.0}
}
```

---

## 9. License

This project is licensed under the [Apache 2.0 License](LICENSE).
