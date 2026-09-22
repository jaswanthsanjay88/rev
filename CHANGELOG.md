# Changelog

All notable changes to `rev` (`rev-decision`) are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.2.0] - 2026-09-22

### Added
- **Unified "One Model" Decision Engine**: Single entry point (`rev.Model()` and `rev.predict()`) that unifies ModernBERT-large (421M, English) and mmBERT-base (322M, 100+ languages) with zero manual model switching.
- **Sub-Microsecond Script & Language Routing (`rev.lang`)**: Zero-dependency Unicode script detector covering 26 non-Latin scripts + European Latin stopwords/diacritics in $<1\ \mu\text{s}$.
- **RLCD Strictly Proper Scoring Rules (`rev.common`)**: Implemented LogScore + Spherical Score + Ranked Probability Score (RPS) directly penalizing overconfident predictions on ambiguous boundary samples.
- **Action / Deferral Head (`act_head`)**: Automatic classification of whether to act autonomously or defer/escalate to a System 2 LLM or human based on uncertainty metrics.
- **Continuous Expected Score**: Support for floating-point continuous expected levels ($\mathbb{E}[\text{score}] = \sum i \cdot p_i$) on ordinal `score` questions.
- **High-Cardinality Candidate Shortlisting (`rev.shortlist`)**: Bi-encoder cosine similarity shortlisting for questions with 20 to 255+ options before cross-attentive scoring.
- **Enterprise Presets (`rev.presets`)**: Turnkey decision schemas for ticket triage, email threat filtering, LLM guardrails, moderation, invoice verification, incident response, and trace observability.
- **Email Sanitization Utilities (`rev.email`)**: Automatic stripping of quoted reply chains, signature blocks, and legal disclaimers.
- **PyPI Packaging (`rev-decision`)**: PEP 517/518/621 build configuration in `pyproject.toml` and `setup.py` with CLI entry point `rev-server`.
- **PEP 561 Type Annotations**: Added `rev/py.typed` marker for Mypy and Pyright static type checking.
- **Jupyter Notebook**: Added [`notebooks/train_laya_system_one_decision_model.ipynb`](notebooks/train_laya_system_one_decision_model.ipynb) for end-to-end training and calibration on Google Colab or Kaggle.
- **Unit Test Suite**: Added [`tests/test_unified_model.py`](tests/test_unified_model.py) verifying routing, scoring rules, shortlisting, and presets.

### Changed
- Re-architected model runtime to support bidirectional transformer encoders alongside causal backbones.
- Bounded post-hoc calibration temperatures to $[0.5, 5.0]$ to prevent pathological logit over-sharpening.
- Upgraded `POST /v1/systemone` server endpoints to support routing previews and unified encoder dispatch.

---

## [0.1.0] - 2026-09-20

### Added
- **Prefill-Only Decision Model**: Built on causal LM backbones (`Qwen/Qwen2.5-0.5B` to 8B) with a bilinear pointer readout head.
- **Document Prefix KV-Caching**: In-memory SHA-256 LRU cache for key/value activations, enabling repeated queries in under 4ms (>10× speedup).
- **Block-Causal Attention Masking**: Guaranteed mathematical branch isolation between concurrent questions.
- **TypeSafe System One Specification**: Native support for `POST /v1/systemone` matching the official `typesafe-sdk`.
- **FastAPI Inference Server (`rev.serve`)**: Multi-endpoint server with packed, separate, and permutation testing routes.
- **Interactive Web Playground**: Next.js 16 application with packed-vs-separate comparisons and move-by-move Decision Chess.
- **Google Colab Training Pipeline**: 1-click training notebook for LoRA adapters and pointer head readout.
