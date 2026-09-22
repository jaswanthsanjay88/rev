# Limitations & Failure Modes

An honest, transparent appraisal of the technical boundaries, failure modes, and architectural trade-offs inherent to `rev` (`rev-decision`).

---

## Overview

`rev` is engineered as a high-speed, prefill-only **System 1 decision engine**. By prioritizing sub-millisecond latency, zero generative hallucination, and strict probability calibration, certain capabilities are deliberately omitted or bounded.

Understanding these boundaries ensures you deploy `rev` where it excels and avoid architectural mismatches.

---

## 1. High-Cardinality Choice Questions (>20 Options)

### The Limitation
When a `choice` question contains more than 20–25 options (e.g., BANKING77 with 77 intent classes, or full ICD-10 medical coding with thousands of codes), passing every option definition concurrently inside a single forward pass causes:
1. **Token Budget Bloat**: Option descriptions consume sequence length, squeezing out document context.
2. **Attention Weight Diffusion**: Softmax distribution across dozens of option delimiter tokens dilutes the readout signal, degrading accuracy.

### Recommended Mitigation
Use **Coarse-to-Fine Shortlisting (`rev.shortlist`)**:
```python
import rev

# Automatically prunes criteria to top-k semantic candidates before cross-attentive scoring
res = rev.predict_shortlist(
    model,
    state=document,
    questions=questions_with_80_options,
    embed_fn=rev.embed_fn_from_agent(model.load("english")),
    k=15  # Only the top 15 most relevant options enter the cross-attention pass
)
```

---

## 2. Generative & Free-Form Text Generation

### The Limitation
`rev` is strictly **prefill-only**. 
- It does not contain an autoregressive language modeling head (`lm_head`).
- It cannot write natural language summaries, compose conversational responses, or generate code.
- Trying to prompt `rev` to "explain why" or "write a draft" will fail by design.

### Architectural Role: System 1 / System 2 Pairing
`rev` is designed to be the fast, low-cost front door that orchestrates when and how a heavier System 2 model (e.g., Claude 3.5 Sonnet, GPT-4o, or DeepSeek R1) is invoked:

```
User Input ──> [rev: 18ms System 1]
                    │
                    ├──> Low urgency / Simple ──> Deterministic API Action (0 ms generation)
                    └──> High complexity / Defer ──> [System 2 LLM: Full Text Generation]
```

---

## 3. Specialized Niche Domains Without Fine-Tuning

### The Limitation
`rev`'s pretrained backbones (ModernBERT-large and mmBERT-base) possess strong world knowledge across standard enterprise workflows (support tickets, invoice triage, security alerts, content moderation, agent trace evaluation).

However, in ultra-specialized domains:
- **Biochemical / Clinical Pharmacology** (interpreting specific molecular binding affinities or rare medical trial criteria).
- **Statutory Legal Analysis** (interpreting subtle cross-references in regional statutes).

Out-of-the-box accuracy may drop if option definitions rely on jargon absent from general corpora.

### Recommended Mitigation
Fine-tune a LoRA adapter or custom checkpoint on your domain dataset using `rev.train` or the provided notebook (`notebooks/train_laya_system_one_decision_model.ipynb`).

---

## 4. Extreme Context Lengths (>8,192 Tokens)

### The Limitation
While ModernBERT supports extended sequences, documents exceeding 8,192 tokens:
1. Incur quadratic memory pressure unless FlashAttention-2 / SDPA is enabled.
2. Lead to subtle needle-in-a-haystack attention dilution when instructions require synthesizing disparate sections of a 50-page document.

### Recommended Mitigation
- For multi-page PDFs or legal contracts, chunk documents into logical sections or pass pre-extracted document summaries into the `state`.
- Leverage `rev.cache` prefix caching on chunked segments.

---

## 5. Subtle Code-Switching & Mixed Scripts

### The Limitation
`rev.lang` uses a zero-dependency Unicode character frequency heuristic operating in $<1\ \mu\text{s}$. If an input consists of 95% English text with an isolated foreign slang phrase or loanword (e.g. *"I am feeling very gemütlich today"*), the text may be classified as English.

Conversely, if an English sentence contains excessive non-Latin decorative emojis or symbols, it may trigger the multilingual threshold.

### Recommended Mitigation
If your application domain has known language distributions, pass an explicit hint:
```python
# Force multilingual routing regardless of text heuristics:
res = rev.predict(state, questions, lang="multilingual")

# Or force English:
res = rev.predict(state, questions, lang="en")
```

---

## 6. Calibration Limits on Heavy Class Imbalance

### The Limitation
While RLCD proper scoring rules achieve an outstanding Expected Calibration Error ($ECE = 0.081$), if your deployment environment experiences severe real-world class imbalance (e.g., fraud occurring in 0.001% of transactions), raw model probabilities will exhibit prior drift.

### Recommended Mitigation
Apply temperature scaling and Bayesian prior correction:
```python
# Bounded temperature adjustment in rev.common
scaled_temp = rev.clamp_temperature(raw_temp)
```
