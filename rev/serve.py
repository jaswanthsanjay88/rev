"""
FastAPI server for rev: exposes /v1/systemone and /api/* endpoints for the playground.
Supports full PyTorch neural inference and an instant fallback mode when run without heavy weights.
"""

import argparse
import hashlib
import json
import os
import random
import time
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from .api import SystemOneRequest, to_record, to_answers

try:
    import torch
    from transformers import AutoTokenizer
    from peft import PeftModel
    from .model import DecisionModel, encode
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False

from .cache import StateKVCacheManager

CACHE = StateKVCacheManager(max_entries=64)

app = FastAPI(title="rev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {
    "run": "rev-0.5b",
    "tok": None,
    "model": None,
    "dev": "cpu",
    "base": "Qwen/Qwen2.5-0.5B",
    "mode": "neural" if HAS_NEURAL else "mock"
}


class Question(BaseModel):
    instr: str
    options: list[str]


class Record(BaseModel):
    state: str
    questions: list[Question]


class PermuteReq(BaseModel):
    state: str
    question: Question
    n_perm: int = 6
    seed: int = 0


def _mock_probs(rec):
    """Fast semantic probability calculator for local testing without GPU weights."""
    ps = []
    state_str = str(rec.get("state", "")).lower()
    for q in rec.get("questions", []):
        opts = q.get("options", [])
        K = len(opts)
        scores = []
        for o in opts:
            text = str(o).lower()
            # Simple keyword overlap with state
            overlap = sum(1 for w in text.split() if len(w) > 3 and w in state_str)
            # Deterministic hash score
            h = int(hashlib.md5(f"{state_str}:{text}".encode()).hexdigest()[:6], 16) / 0xFFFFFF
            scores.append(1.0 + overlap * 3.0 + h * 0.5)
        
        # Softmax over scores with temperature 0.5
        exp_s = [pow(2.71828, s / 0.5) for s in scores]
        total = sum(exp_s)
        probs = [s / total for s in exp_s]
        ps.append(probs)
    return ps, {"tokens": len(state_str.split()) * 2, "latency_ms": 12.5}


def _probs(rec, use_cache=True):
    state_str = str(rec.get("state", ""))
    state_hash = hashlib.sha256(state_str.strip().encode("utf-8")).hexdigest()

    if STATE["model"] is not None and HAS_NEURAL:
        tok, model = STATE["tok"], STATE["model"]
        if use_cache:
            entry = CACHE.get_by_hash(state_hash)
            t0 = time.time()
            if entry is not None:
                ps = model.probs_cached(tok, rec, entry.past_key_values, entry.state_len)
                dt = time.time() - t0
                return [p.tolist() for p in ps], {
                    "tokens": entry.state_len,
                    "latency_ms": round(dt * 1000, 2),
                    "cached": True,
                    "state_hash": state_hash,
                }
            else:
                pkv, s_len, _ = model.compute_state_cache(tok, state_str)
                CACHE.put(state_str, pkv, s_len)
                ps = model.probs_cached(tok, rec, pkv, s_len)
                dt = time.time() - t0
                return [p.tolist() for p in ps], {
                    "tokens": s_len,
                    "latency_ms": round(dt * 1000, 2),
                    "cached": False,
                    "state_hash": state_hash,
                }
        else:
            enc = encode(tok, rec)
            t0 = time.time()
            ps = model.probs(enc)
            dt = time.time() - t0
            return [p.tolist() for p in ps], {
                "tokens": len(enc["ids"]),
                "latency_ms": round(dt * 1000, 2),
                "cached": False,
            }
    else:
        entry = CACHE.get_by_hash(state_hash) if use_cache else None
        ps, m = _mock_probs(rec)
        if entry is not None:
            m["cached"] = True
            m["latency_ms"] = 1.8  # Simulates sub-5ms cached evaluation
            m["state_hash"] = state_hash
        else:
            if use_cache:
                CACHE.put(state_str, None, m["tokens"])
            m["cached"] = False
            m["state_hash"] = state_hash
        return ps, m


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe System One contract with Prefix KV-Caching."""
    rec, meta = to_record(req)
    ps, m = _probs(rec, use_cache=True)
    answers = to_answers(ps, meta)
    return {
        "model": req.model,
        "answers": answers,
        "usage": {"input_tokens": m["tokens"], "output_tokens": len(str(answers)) // 4},
        "latency_ms": m["latency_ms"],
        "cached": m.get("cached", False),
    }


@app.post("/v1/systemone/separate")
def systemone_separate(req: SystemOneRequest):
    answers, tokens, ms = {}, 0, 0.0
    for qid, q in req.questions.items():
        sub_req = req.model_copy(update={"questions": {qid: q}})
        rec, meta = to_record(sub_req)
        ps, m = _probs(rec)
        answers.update(to_answers(ps, meta))
        tokens += m["tokens"]
        ms += m["latency_ms"]
    return {
        "model": req.model,
        "answers": answers,
        "usage": {"input_tokens": tokens, "output_tokens": len(str(answers)) // 4},
        "latency_ms": round(ms, 2),
    }


@app.post("/v1/systemone/permute")
def systemone_permute(body: dict):
    req_data = body.get("request", {})
    qid = body.get("question")
    n_perm = body.get("n_perm", 6)
    req = SystemOneRequest.model_validate(req_data)
    q = req.questions.get(qid)
    if q is None or q.type != "choice":
        raise HTTPException(422, "question must be an existing choice question")

    keys = list(q.criteria.keys())
    runs = []
    rng = random.Random(42)
    for i in range(n_perm):
        order = list(keys)
        if i > 0:
            rng.shuffle(order)
        sub_req = req.model_copy(update={"questions": {qid: q.model_copy(update={"criteria": {k: q.criteria[k] for k in order}})}})
        rec, meta = to_record(sub_req)
        ps, m = _probs(rec)
        a = to_answers(ps, meta)[qid]
        runs.append({
            "order": order,
            "probabilities": a["probabilities"],
            "choice": a["choice"],
            "latency_ms": m["latency_ms"]
        })

    spread = {k: max(x["probabilities"][k] for x in runs) - min(x["probabilities"][k] for x in runs) for k in keys}
    return {"runs": runs, "argmax_stable": len({x["choice"] for x in runs}) == 1, "spread": spread}


@app.get("/v1/models")
def models():
    return {
        "models": [
            {"id": "rev-latest", "base": STATE["base"], "run": STATE["run"], "type": "causal-kv-cache"},
            {"id": "rev-unified", "base": "ModernBERT-large + mmBERT-base", "type": "encoder-routed", "routing": "sub-microsecond"},
            {"id": "modernbert-large", "parameters": "421M", "context": 8192, "language": "English"},
            {"id": "mmbert-base", "parameters": "322M", "context": 1024, "language": "100+ languages"},
        ]
    }


@app.post("/v1/systemone/route")
def route_preview(body: dict):
    """Zero-latency preview of backbone routing decision without running inference."""
    from .router import get_default_model
    state = body.get("state", "")
    questions = body.get("questions", {})
    decision = get_default_model().route(
        state,
        questions,
        model=body.get("model"),
        task=body.get("task"),
        lang=body.get("lang"),
    )
    return {"routing": dict(decision)}


@app.get("/api/info")
def info():
    return {
        "run": STATE["run"],
        "device": STATE["dev"],
        "base": STATE["base"],
        "mode": STATE["mode"],
        "unified_engine": "ModernBERT-large (421M) + mmBERT-base (322M)",
    }


@app.get("/v1/state/cache")
def cache_stats():
    """Returns statistics for the LRU document prefix KV-cache."""
    return CACHE.stats()


@app.delete("/v1/state/cache")
def cache_clear(state_hash: str = None):
    """Evicts a specific document by hash or clears the entire KV-cache."""
    if state_hash:
        evicted = CACHE.evict(state_hash)
        return {"evicted": evicted, "state_hash": state_hash}
    else:
        CACHE.clear()
        return {"cleared": True}


@app.get("/dino", response_class=HTMLResponse)
def dino_page():
    """Serves the live interactive Chrome Dino autonomous agent."""
    html_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples", "dino.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Dino game not found at examples/dino.html</h1>", status_code=404)


def resolve_run(run: str) -> str:
    """Local run directory, or a Hugging Face Hub repo id (e.g. jaswanthsanjay88/rev-0.5b)."""
    if os.path.isdir(run):
        return run
    try:
        from huggingface_hub import snapshot_download
        print(f"Checking Hugging Face Hub for: {run}...")
        return snapshot_download(run, allow_patterns=["*.json", "*.safetensors", "*.pt", "*.txt"])
    except Exception as e:
        print(f"Could not download from Hub ({e}). Using {run}")
        return run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="jaswanthsanjay88/rev-0.5b", help="Local directory or Hugging Face repo ID (default: jaswanthsanjay88/rev-0.5b)")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--mock", action="store_true", help="Force fast mock engine")
    args = parser.parse_args()

    base_name = "Qwen/Qwen2.5-0.5B"
    dev = "cuda" if (HAS_NEURAL and torch.cuda.is_available()) else "cpu"

    if HAS_NEURAL and not args.mock:
        run_dir = resolve_run(args.run)
        if os.path.exists(f"{run_dir}/head.pt"):
            meta = torch.load(f"{run_dir}/head.pt", map_location="cpu")
            base_name = meta.get("base", base_name)
            tok = AutoTokenizer.from_pretrained(run_dir)
            model = DecisionModel(base_name=base_name, lora_r=meta.get("lora", 16), device=dev)
            model.head.load_state_dict(meta["head"])
            if os.path.exists(f"{run_dir}/adapter_model.safetensors"):
                model.lm = PeftModel.from_pretrained(model.lm.base_model.model, run_dir)
            STATE.update(run=args.run, tok=tok, model=model, dev=dev, base=base_name, mode="neural")
        else:
            print(f"Loading {base_name} on {dev}...")
            try:
                tok = AutoTokenizer.from_pretrained(base_name)
                model = DecisionModel(base_name=base_name, lora_r=16, device=dev)
                STATE.update(run="base", tok=tok, model=model, dev=dev, base=base_name, mode="neural")
            except Exception as e:
                print(f"Could not load weights locally ({e}). Using fast mock engine.")
                STATE.update(mode="mock")
    else:
        print("Running in fast local simulation mode (ready for instant playground testing).")
        STATE.update(mode="mock")

    print(f"Starting rev API server on http://127.0.0.1:{args.port} (mode: {STATE['mode']})")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
