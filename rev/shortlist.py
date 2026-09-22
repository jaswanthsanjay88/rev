"""
Opt-in candidate shortlisting for high-cardinality choice questions in rev.

When a decision question has 20 to 255+ options (e.g. BANKING77, intent categorization,
catalog hierarchies), option descriptions can exceed token budgets.
`predict_shortlist` performs coarse-to-fine filtering:
  1. Embeds state and candidate options using a fast bi-encoder.
  2. Shortlists the top-k highest similarity candidates.
  3. Executes the full cross-attentive decision model over the shortlisted candidates.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence
import numpy as np

from .common import render_options, serialize_state

DEFAULT_SHORTLIST_K = 20


def _check_k(k: Any) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError(f"k must be a positive integer, got {k!r}")
    return k


def _subset_criteria(criteria: Any, kept_labels: Sequence[Any]) -> Any:
    kept_set = set(kept_labels)
    if isinstance(criteria, dict):
        return {lbl: criteria[lbl] for lbl in kept_labels if lbl in criteria}
    if isinstance(criteria, (list, tuple)):
        return [lbl for lbl in kept_labels if lbl in kept_set]
    return criteria


def _cosine_scores(query_vec: np.ndarray, doc_vecs: np.ndarray) -> np.ndarray:
    q_norm = np.linalg.norm(query_vec)
    d_norm = np.linalg.norm(doc_vecs, axis=1)
    denom = np.maximum(q_norm * d_norm, 1e-12)
    dots = doc_vecs @ query_vec
    scores = dots / denom
    scores[np.isnan(scores)] = 0.0
    return scores


def _rank(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int,
    instructions: Optional[str] = None,
):
    dummy_q = {
        "t": "choice",
        "ins": instructions or "",
        "crit": criteria,
    }
    opt_texts = render_options(dummy_q)
    labels = list(criteria.keys()) if isinstance(criteria, dict) else list(criteria)
    n = len(labels)

    if n <= k:
        return labels, None, True, n

    query_str = serialize_state(state)
    if instructions:
        query_str = f"{instructions} {query_str}"

    texts_to_embed = [query_str] + opt_texts
    vectors = np.asarray(embed_fn(texts_to_embed), dtype=np.float32)

    query_vec = vectors[0]
    opt_vecs = vectors[1:]

    scores = _cosine_scores(query_vec, opt_vecs)

    # Sort descending by score, stable sort preserving original order on ties
    ranked_indices = np.argsort(-scores, kind="mergesort")[:k]
    kept_labels = [labels[idx] for idx in ranked_indices]
    kept_scores = [float(scores[idx]) for idx in ranked_indices]

    return kept_labels, kept_scores, False, n


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Optional[str] = None,
) -> List[Any]:
    """Returns top-k choice labels for state based on semantic similarity."""
    labels, _scores, _passthrough, _n = _rank(state, criteria, embed_fn, _check_k(k), instructions)
    return labels


def predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """
    Shortlists high-cardinality choice questions, then executes the decision model.
    Non-choice questions and choice questions with <= k options pass through untouched.
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question_id -> definition")
    checked_k = _check_k(k)
    reduced: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}

    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise ValueError(f"Question {qid!r} is a choice question but has no criteria.")
        labels, scores, passthrough, n = _rank(
            state, qdef["criteria"], embed_fn, checked_k, qdef.get("instructions")
        )
        meta[qid] = {
            "labels": list(labels),
            "scores": scores,
            "k": checked_k,
            "n": n,
            "passthrough": passthrough,
        }
        if passthrough:
            reduced[qid] = qdef
        else:
            updated = dict(qdef)
            updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
            reduced[qid] = updated

    predict_method = getattr(agent, "predict", None) or getattr(agent, "system_one", None)
    if predict_method is None:
        raise AttributeError("agent must implement predict() or system_one()")

    result = predict_method(state, reduced, **predict_kwargs)
    out = dict(result)
    out["shortlist"] = meta
    return out


def embed_fn_from_agent(
    agent: Any,
    max_length: int = 512,
    batch_size: int = 32,
) -> Callable[[Sequence[str]], np.ndarray]:
    """
    Creates an embedding function using the loaded agent's backbone encoder with mean-pooling.
    Zero additional weight download required.
    """
    import torch

    tok = agent.tok
    encoder = agent.model.encoder
    device = agent.device

    def embed_fn(texts: Sequence[str]) -> np.ndarray:
        rows = ["" if t is None else str(t) for t in texts]
        hidden_size = encoder.config.hidden_size
        if not rows:
            return np.zeros((0, hidden_size), dtype=np.float32)

        parts: List[np.ndarray] = []
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            encoded = tok(
                chunk,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)
            with torch.inference_mode():
                hidden_states = encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                mask = attention_mask.unsqueeze(-1).to(dtype=hidden_states.dtype)
                pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
            parts.append(pooled.float().cpu().numpy())
        return np.vstack(parts)

    return embed_fn
