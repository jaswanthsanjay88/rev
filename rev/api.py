"""
TypeSafe-compatible request/response shapes (POST /v1/systemone) mapped onto the single pointer primitive.

Noul   -> 2 options [false, true];              answer = p(true)
Choice -> options 'name' or 'name: desc';       answer = argmax, probabilities by name, confidence
Score  -> options = ordered level descriptions; answer = expected level, legend, probabilities by index
"""

from typing import Any, Literal, Union, Dict, List
from pydantic import BaseModel, Field, model_validator

JSONContent = Union[str, dict, list, int, float, bool, None]
MAX_OPTIONS = 255


class Noul(BaseModel):
    type: Literal["noul"] = "noul"
    instructions: JSONContent
    criteria: dict[str, JSONContent] | None = None


class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: JSONContent
    criteria: dict[str, JSONContent]

    @model_validator(mode="after")
    def _check(self):
        if not 1 <= len(self.criteria) <= MAX_OPTIONS:
            raise ValueError(f"criteria must have 1..{MAX_OPTIONS} options")
        return self


class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: JSONContent
    criteria: list[JSONContent] = Field(min_length=2, max_length=MAX_OPTIONS)


Question = Union[Noul, Choice, Score]


class SystemOneRequest(BaseModel):
    state: JSONContent = ""
    image: str | None = None
    image_b64: str | None = None
    model: str = "rev-latest"
    questions: dict[str, Question] = Field(min_length=1)


def render(v: JSONContent, indent: int = 0) -> str:
    """Flatten str | object | array into text the model sees."""
    pad = "  " * indent
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    if isinstance(v, list):
        return "\n".join(f"{pad}- {render(x, indent + 1).lstrip()}" for x in v)
    return "\n".join(
        f"{pad}{k}:\n{render(x, indent + 1)}" if isinstance(x, (dict, list)) else f"{pad}{k}: {render(x)}"
        for k, x in v.items()
    )


def option_text(name: str, desc: JSONContent) -> str:
    return name if desc is None or desc == "" else f"{name}: {render(desc)}"


def to_record(req: SystemOneRequest):
    """-> internal record for encode(), plus per-question metadata to map probabilities back."""
    qs, meta = [], []
    for qid, q in req.questions.items():
        instr = render(q.instructions)
        if q.type == "noul":
            c = q.criteria or {}
            opts = [option_text("no", c.get("false")), option_text("yes", c.get("true"))]
            meta.append({"id": qid, "type": "noul"})
        elif q.type == "choice":
            opts = [option_text(k, v) for k, v in q.criteria.items()]
            meta.append({"id": qid, "type": "choice", "keys": list(q.criteria.keys())})
        else:
            opts = [render(x) for x in q.criteria]
            meta.append({"id": qid, "type": "score", "legend": {str(i): render(x) for i, x in enumerate(q.criteria)}})
        qs.append({"instr": instr, "options": opts, "label": 0})
    img = req.image or req.image_b64
    if img is None and isinstance(req.state, dict):
        img = req.state.get("image") or req.state.get("image_b64")
    rec: dict[str, Any] = {"state": render(req.state), "questions": qs}
    if img is not None:
        rec["image"] = img
    return rec, meta


def choice_confidence(p: list[float]) -> float:
    K = len(p)
    return 1.0 if K == 1 else (max(p) - 1.0 / K) / (1.0 - 1.0 / K)


def score_confidence(p: list[float]) -> float:
    L = len(p)
    mode = max(range(L), key=lambda i: p[i])
    return 1.0 - sum(pi * abs(i - mode) for i, pi in enumerate(p)) / (L - 1)


def to_answers(probs: list[list[float]], meta: list[dict]) -> dict[str, Any]:
    out = {}
    for p, m in zip(probs, meta):
        if m["type"] == "noul":
            out[m["id"]] = {"type": "noul", "noul": round(p[1], 3)}
        elif m["type"] == "choice":
            best_idx = max(range(len(p)), key=lambda i: p[i])
            out[m["id"]] = {
                "type": "choice",
                "choice": m["keys"][best_idx],
                "confidence": round(choice_confidence(p), 3),
                "probabilities": {k: round(v, 4) for k, v in zip(m["keys"], p)},
            }
        else:
            score = sum(i * pi for i, pi in enumerate(p))
            out[m["id"]] = {
                "type": "score",
                "score": round(score, 2),
                "confidence": round(score_confidence(p), 3),
                "legend": m["legend"],
                "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)},
            }
    return out
