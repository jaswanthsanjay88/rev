"""
Core model utilities, sequence construction, and proper scoring rules for rev.
Supports both bidirectional encoders (ModernBERT, mmBERT) and causal LM decoders.
"""
import json
import math
import os
from typing import Dict, List, Optional, Union, Tuple, Any

import numpy as np
import torch
import torch.nn as nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}


def serialize_state(state: Union[str, dict, list]) -> str:
    """Serialize arbitrary state (str, dict, list) to text."""
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_criterion(value: Any) -> str:
    """Render a single rubric criterion as a string or compact JSON."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def render_options(q: Dict) -> List[str]:
    """Render option texts in label-index order. Noul is always [false, true]."""
    t = q.get("t") or q.get("type", "choice")
    crit = q.get("crit") or q.get("criteria", {})
    if t == "choice":
        if isinstance(crit, dict):
            return [k if v is None or v == "" else f"{k}: {render_criterion(v)}" for k, v in crit.items()]
        elif isinstance(crit, list):
            return [render_criterion(v) for v in crit]
        return [str(crit)]
    if t == "score":
        if isinstance(crit, list):
            return [f"level {i}: {render_criterion(c)}" for i, c in enumerate(crit)]
        return [f"level {k}: {render_criterion(v)}" for k, v in crit.items()] if isinstance(crit, dict) else [str(crit)]
    
    # Noul (binary boolean)
    crit = crit or {}
    if isinstance(crit, dict):
        false_crit = crit.get("false")
        true_crit = crit.get("true")
    else:
        false_crit, true_crit = None, None
    return [
        "false: " + (render_criterion(false_crit) if false_crit not in (None, "") else "no, the statement does not hold"),
        "true: " + (render_criterion(true_crit) if true_crit not in (None, "") else "yes, the statement holds"),
    ]


def build_sequence(
    tok,
    state: Union[str, dict, list],
    q: Dict,
    max_len: int = 512,
    head_max_len: int = 192,
    option_order: Optional[List[int]] = None,
    truncate_left: bool = False,
) -> Tuple[List[int], List[int]]:
    """
    Constructs an encoder sequence formatted with option markers:
    Format: [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    Returns (token_ids, marker_positions).
    """
    mask_tok = getattr(tok, "mask_token", "[MASK]") or "[MASK]"
    mask_tok_id = getattr(tok, "mask_token_id", None)
    if mask_tok_id is None:
        mask_tok_id = tok.convert_tokens_to_ids(mask_tok)
        
    cls_id = getattr(tok, "cls_token_id", None) or getattr(tok, "bos_token_id", None) or 1
    sep_id = getattr(tok, "sep_token_id", None) or getattr(tok, "eos_token_id", None) or 2

    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    q_type = q.get("t") or q.get("type", "choice")
    ins = str(q.get("ins") or q.get("instructions", "")).replace(mask_tok, " ")

    head_ids = tok(f"{q_type} question: {ins}", add_special_tokens=False)["input_ids"]
    opt_ids = []
    for i in order:
        opt_ids.append(
            [mask_tok_id]
            + tok(" " + opts[i].replace(mask_tok, " "), add_special_tokens=False)["input_ids"][:48]
        )
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    
    ids = [cls_id] + head_ids + [sep_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(sep_id)
    
    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [sep_id]
    
    return ids[:max_len], [m for m in markers if m < max_len]


def collate_items(batch, pad_id: int):
    """Collate token sequences and marker locations for batch inference."""
    items = [it for group in batch for it in group] if (batch and isinstance(batch[0], list)) else batch
    if not items:
        return None
    n = len(items)
    L = max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    
    ids = torch.full((n, L), pad_id, dtype=torch.long)
    att = torch.zeros((n, L), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    has_target = any("target" in it for it in items)
    target = torch.zeros((n, kmax), dtype=torch.float32) if has_target else None

    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        if has_target and "target" in it:
            target[i, : len(it["target"])] = torch.tensor(it["target"], dtype=torch.float32)

    res = {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": mpos,
        "marker_mask": mmask,
        "qtype": torch.tensor([it["qtype"] for it in items]),
        "label": torch.tensor([it.get("label", -1) for it in items]),
        "meta": [{k: it[k] for k in it if k not in ("ids", "markers", "target")} for it in items],
    }
    if target is not None:
        res["target"] = target
    return res


def proper_reward(
    q: torch.Tensor,
    target: torch.Tensor,
    qtype: torch.Tensor,
    mask: torch.Tensor,
    w_sph: float = 0.5,
    w_rps: float = 1.0,
    log_floor: float = -9.21,
) -> torch.Tensor:
    """
    Strictly proper scoring rule reward for RLCD:
    Combines Logarithmic Score + Spherical Score + Ranked Probability Score (RPS).
    """
    q = q * mask
    logq = torch.log(q.clamp_min(1e-12)).clamp_min(log_floor)
    log_score = (target * logq).sum(-1)
    sph = (target * q).sum(-1) / q.norm(dim=-1).clamp_min(1e-9)
    r = log_score + w_sph * sph
    
    is_score = (qtype == QTYPES["score"]).float()
    if is_score.any():
        k = mask.sum(-1).clamp(min=2).float()
        cdf_q = torch.cumsum(q, -1)
        cdf_t = torch.cumsum(target, -1)
        rps = (((cdf_q - cdf_t) ** 2) * mask).sum(-1) / (k - 1)
        r = r - w_rps * rps * is_score
    return r


def confidence_from_probs(p: np.ndarray, k: int) -> float:
    """
    Normalized Shannon entropy confidence metric: 1 - H(p) / log(k).
    Returns 1.0 for completely certain predictions and 0.0 for uniform guessing.
    """
    if k < 2:
        return 1.0
    p = p[:k]
    ent = -(p * np.log(np.clip(p, 1e-12, 1.0))).sum()
    return float(np.clip(1.0 - ent / math.log(k), 0.0, 1.0))


TEMP_MIN = 0.5
TEMP_MAX = 5.0


def clamp_temperature(t: Any, lo: float = TEMP_MIN, hi: float = TEMP_MAX) -> float:
    """Clamp temperature scaling factor to prevent pathological logit sharpening."""
    try:
        t = float(t)
    except (TypeError, ValueError):
        return 1.0
    if t != t or t in (float("inf"), float("-inf")):
        return 1.0
    return min(hi, max(lo, t))


def ece_score(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected Calibration Error across confidence bins."""
    if len(conf) == 0:
        return float("nan")
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (conf > lo) & (conf <= hi)
        if sel.any():
            e += sel.mean() * abs(conf[sel].mean() - correct[sel].mean())
    return float(e)


def temp_bucket(qtype: int, k: int) -> str:
    """Returns temperature bucket key based on question type and number of options."""
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return "%s:%s" % (QTYPE_NAMES[int(qtype)], size)


def amp_dtype(name: Optional[str]) -> torch.dtype:
    """Resolve automatic mixed precision dtype."""
    if name == "bf16":
        return torch.bfloat16
    return torch.float16
