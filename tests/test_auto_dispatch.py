"""
Tests for Rev.from_pretrained and the unified AutoModel dispatcher.
"""

import pytest
from rev import Rev, DecisionEngine, Model, Agent
from rev.auto import _detect_architecture


def test_detect_architecture():
    # Aliases
    assert _detect_architecture("mock") == "mock"
    assert _detect_architecture("test") == "mock"
    assert _detect_architecture("jaswanthsanjay88/rev-vision") == "vision"
    assert _detect_architecture("jaswanthsanjay88/rev-4b") == "causal"
    assert _detect_architecture("jaswanthsanjay88/rev-0.5b") == "causal"
    assert _detect_architecture("jaswanthsanjay88/rev-decision-model") == "encoder"


def test_rev_from_pretrained_mock():
    engine = Rev.from_pretrained("mock")
    assert isinstance(engine, DecisionEngine)

    state = "The server at 192.168.1.10 triggered high memory alert and crashed."
    questions = {
        "is_critical": {
            "type": "noul",
            "instructions": "Is this a critical incident?",
        },
        "action": {
            "type": "choice",
            "criteria": ["restart_service", "ignore", "escalate"],
        },
        "severity": {
            "type": "score",
            "criteria": ["low", "medium", "high", "catastrophic"],
        },
    }

    res = engine.predict(state, questions)

    # Base contract assertions
    assert "answers" in res
    assert "latency_ms" in res
    assert "model" in res
    assert res["model"] == "mock"

    answers = res["answers"]
    assert "is_critical" in answers
    assert "action" in answers
    assert "severity" in answers

    # noul structure
    noul_ans = answers["is_critical"]
    assert "noul" in noul_ans
    assert "confidence" in noul_ans
    assert 0.0 <= noul_ans["noul"] <= 1.0

    # choice structure
    choice_ans = answers["action"]
    assert "choice" in choice_ans
    assert "confidence" in choice_ans
    assert "probabilities" in choice_ans
    assert choice_ans["choice"] in ["restart_service", "ignore", "escalate"]

    # score structure
    score_ans = answers["severity"]
    assert "score" in score_ans
    assert "distribution" in score_ans
    assert len(score_ans["distribution"]) == 4


def test_backward_compatibility_imports():
    assert Rev is not None
    assert Model is not None
    assert Agent is not None
    assert DecisionEngine is not None
