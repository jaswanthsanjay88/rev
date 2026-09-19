"""
FastAPI server for rev: exposes /v1/systemone and /api/* endpoints for the playground.
"""

import argparse
import json
import os
import random
import time
import threading
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer
from peft import PeftModel
from .api import SystemOneRequest, to_record, to_answers
from .model import DecisionModel, encode

app = FastAPI(title="rev")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

STATE = {"run": None, "tok": None, "model": None, "dev": None, "base": None}


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


def _rec(r: Record):
    return {"state": r.state, "questions": [{"instr": q.instr, "options": q.options, "label": 0} for q in r.questions]}


def _probs(rec):
    tok, model = STATE["tok"], STATE["model"]
    enc = encode(tok, rec)
    t0 = time.time()
    ps = model.probs(enc)
    dt = time.time() - t0
    return [p.tolist() for p in ps], {"tokens": len(enc["ids"]), "latency_ms": round(dt * 1000, 2)}


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    """TypeSafe System One contract."""
    rec, meta = to_record(req)
    ps, m = _probs(rec)
    answers = to_answers(ps, meta)
    return {
        "model": req.model,
        "answers": answers,
        "usage": {"tokens": m["tokens"]},
        "latency_ms": m["latency_ms"],
    }


@app.get("/v1/models")
def models():
    return {"models": [{"id": "rev-latest", "base": STATE.get("base", "Qwen/Qwen2.5-0.5B")}]}


@app.get("/api/info")
def info():
    return {"run": STATE["run"], "device": STATE["dev"], "base": STATE["base"]}


@app.post("/api/predict")
def predict(r: Record):
    ps, meta = _probs(_rec(r))
    return {"probs": ps, **meta}


@app.post("/api/predict_separate")
def predict_separate(r: Record):
    rec = _rec(r)
    out, tokens, ms = [], 0, 0.0
    for q in rec["questions"]:
        ps, meta = _probs({"state": rec["state"], "questions": [q]})
        out.append(ps[0])
        tokens += meta["tokens"]
        ms += meta["latency_ms"]
    return {"probs": out, "tokens": tokens, "latency_ms": round(ms, 2)}


@app.post("/api/permute")
def permute(r: PermuteReq):
    rng = random.Random(r.seed)
    K = len(r.question.options)
    runs = []
    for i in range(r.n_perm):
        perm = list(range(K))
        if i > 0:
            rng.shuffle(perm)
        q = {"instr": r.question.instr, "options": [r.question.options[j] for j in perm], "label": 0}
        ps, meta = _probs({"state": r.state, "questions": [q]})
        orig = [0.0] * K
        for pos, j in enumerate(perm):
            orig[j] = ps[0][pos]
        runs.append({
            "perm": perm,
            "probs": orig,
            "argmax": perm[max(range(K), key=lambda i: ps[0][i])],
            "latency_ms": meta["latency_ms"],
        })
    argmaxes = {x["argmax"] for x in runs}
    spread = [max(x["probs"][j] for x in runs) - min(x["probs"][j] for x in runs) for j in range(K)]
    return {"runs": runs, "argmax_stable": len(argmaxes) == 1, "spread": spread}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/rev")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    base_name = "Qwen/Qwen2.5-0.5B"

    if os.path.exists(f"{args.run}/head.pt"):
        meta = torch.load(f"{args.run}/head.pt", map_location="cpu")
        base_name = meta.get("base", base_name)
        tok = AutoTokenizer.from_pretrained(args.run)
        model = DecisionModel(base_name=base_name, lora_r=meta.get("lora", 16), device=dev)
        model.head.load_state_dict(meta["head"])
        if os.path.exists(f"{args.run}/adapter_model.safetensors") or os.path.exists(f"{args.run}/adapter_model.bin"):
            model.lm = PeftModel.from_pretrained(model.lm.base_model.model, args.run)
    else:
        print(f"No checkpoint found at {args.run}. Loading base model {base_name} with untrained head.")
        tok = AutoTokenizer.from_pretrained(base_name)
        model = DecisionModel(base_name=base_name, lora_r=16, device=dev)

    STATE.update(run=args.run, tok=tok, model=model, dev=dev, base=base_name)
    print(f"Starting rev API server on http://127.0.0.1:{args.port} ({dev})")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
