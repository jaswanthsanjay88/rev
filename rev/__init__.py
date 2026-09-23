"""
rev: Unified System 1 Decision Engine with TypeSafe API and Sub-Microsecond Multi-Backbone Routing.

Unifies Pretrained Encoders:
  - ModernBERT-large (421M): English backbone with long-context, deep reasoning.
  - mmBERT-base (322M): Multilingual backbone natively supporting 100+ languages.
  - Typed Decisions (421M): Fine-tuned on specialized enterprise decision schemas.
  - Causal LM Backbones: Document Prefix KV-Caching (<5ms repeated queries).

Features:
  - Sub-microsecond Unicode script routing (rev.lang).
  - RLCD strictly proper scoring rules (log score + spherical + RPS) for calibrated probabilities.
  - High-cardinality candidate shortlisting (rev.shortlist).
  - Production presets for triage, email, guardrails, moderation, invoice, security, traces.
"""

__version__ = "0.2.3"

# Language and Script Detection
from . import lang

# Common Utilities & Scoring Rules
from .common import (
    QTYPES,
    QTYPE_NAMES,
    serialize_state,
    render_criterion,
    render_options,
    build_sequence,
    collate_items,
    proper_reward,
    confidence_from_probs,
    clamp_temperature,
    ece_score,
    amp_dtype,
)

# Encoder Model Architecture
from .encoder_model import (
    EncoderDecisionModel,
    build_encoder_model,
)

# Inference Runtime
from .agent import Agent

# Unified 'One Model' Router
from .router import (
    UnifiedModel,
    Router,
    RouteDecision,
    predict,
    get_default_model,
)

# Core AutoModel & Base Contract
from .engine_base import DecisionEngine
from .auto import Rev

# Primary 'Model', 'Rev', and 'RevVision' aliases
Model = UnifiedModel
RevVision = Rev

# Candidate Shortlisting
from .shortlist import (
    shortlist_choice,
    predict_shortlist,
    embed_fn_from_agent,
)

# Workflow Presets
from . import presets

# Email Utilities
from . import email
from .email import email_state, clean_email_body

# TypeSafe API Models (for backward compatibility and server endpoints)
from .api import (
    Noul,
    Choice,
    Score,
    SystemOneRequest,
    to_record,
    to_answers,
    render,
    choice_confidence,
    score_confidence,
)

# Optional Causal LM Backbone and Pointer Head
try:
    from .model import (
        DecisionModel as CausalDecisionModel,
        PointerHead,
        encode,
        user_tokens,
        branch_mask_batch,
        encode_state,
        encode_question_branches,
    )
except ImportError:
    CausalDecisionModel = None
    PointerHead = None

__all__ = [
    # AutoModel & Core Contract
    "Rev",
    "RevVision",
    "DecisionEngine",
    # Unified Model
    "Model",
    "UnifiedModel",
    "Router",
    "Agent",
    "EncoderDecisionModel",
    "predict",
    "get_default_model",
    "RouteDecision",
    # Presets & Modules
    "presets",
    "lang",
    "email",
    "email_state",
    "clean_email_body",
    # Shortlisting
    "shortlist_choice",
    "predict_shortlist",
    "embed_fn_from_agent",
    # Scoring & Calibration
    "proper_reward",
    "confidence_from_probs",
    "clamp_temperature",
    "ece_score",
    "render_options",
    "build_sequence",
    # TypeSafe API
    "Noul",
    "Choice",
    "Score",
    "SystemOneRequest",
    "to_record",
    "to_answers",
    "render",
    "choice_confidence",
    "score_confidence",
    # Causal
    "CausalDecisionModel",
    "PointerHead",
]
