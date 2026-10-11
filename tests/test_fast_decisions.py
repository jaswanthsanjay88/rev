"""
Tests for rev.fast_decisions loader and schema conversion.
"""

import pytest
from rev.fast_decisions import (
    DOMAINS,
    is_binary_labels,
    is_ordinal_labels,
    convert_fast_decision_row,
    to_system_one_request,
    FastDecisionsDataset,
)


def test_is_binary_and_ordinal_detection():
    assert is_binary_labels(["yes", "no"]) is True
    assert is_binary_labels(["True", "False"]) is True
    assert is_binary_labels(["billing", "technical", "sales"]) is False

    assert is_ordinal_labels("urgency", ["low", "normal", "high", "critical"]) is True
    assert is_ordinal_labels("sentiment", ["negative", "neutral", "positive"]) is True
    assert is_ordinal_labels("department", ["billing", "sales"]) is False


def test_convert_fast_decision_row():
    raw_sample = {
        "input": "My flight was delayed by 6 hours and I need compensation.",
        "output": {
            "classifications": [
                {
                    "task": "intent",
                    "labels": ["compensation", "baggage", "flight_status"],
                    "true_label": ["compensation"],
                },
                {
                    "task": "is_urgent",
                    "labels": ["no", "yes"],
                    "true_label": ["yes"],
                },
                {
                    "task": "urgency",
                    "labels": ["low", "normal", "high", "critical"],
                    "true_label": ["high"],
                },
            ]
        },
    }

    converted = convert_fast_decision_row(raw_sample)
    assert converted["state"] == raw_sample["input"]
    assert len(converted["questions"]) == 3

    # Check Choice
    q0 = converted["questions"][0]
    assert q0["id"] == "intent"
    assert q0["qtype"] == "choice"
    assert q0["label"] == 0
    assert q0["gold"] == "compensation"

    # Check Noul (Binary)
    q1 = converted["questions"][1]
    assert q1["id"] == "is_urgent"
    assert q1["qtype"] == "noul"
    assert q1["label"] == 1

    # Check Score (Ordinal)
    q2 = converted["questions"][2]
    assert q2["id"] == "urgency"
    assert q2["qtype"] == "score"
    assert q2["label"] == 2
    assert q2["gold"] == "high"


def test_to_system_one_request():
    converted = {
        "state": "Refund request",
        "questions": [
            {
                "id": "department",
                "instr": "Classify the Department.",
                "options": ["billing", "support"],
                "label": 0,
                "qtype": "choice",
            },
            {
                "id": "escalate",
                "instr": "Escalate to manager?",
                "options": ["no", "yes"],
                "label": 1,
                "qtype": "noul",
            },
        ],
    }
    req = to_system_one_request(converted)
    assert req["state"] == "Refund request"
    assert "department" in req["questions"]
    assert req["questions"]["department"]["type"] == "choice"
    assert "escalate" in req["questions"]
    assert req["questions"]["escalate"]["type"] == "noul"


def test_fast_decisions_split_and_summary():
    dataset = FastDecisionsDataset(domains=["agent_handoff"])
    # Seed manual domain records
    sample_rec = {
        "state": "sample",
        "questions": [
            {
                "id": "intent",
                "instr": "intent",
                "options": ["a", "b"],
                "label": 0,
                "qtype": "choice",
                "gold": "a",
            }
        ],
    }
    dataset._records_by_domain["agent_handoff"] = [sample_rec] * 10

    train, val, test = dataset.split(train_ratio=0.8, val_ratio=0.1, seed=42)
    assert len(train) == 8
    assert len(val) == 1
    assert len(test) == 1

    summary = dataset.summary()
    assert summary["total_domains"] == 1
    assert summary["total_samples"] == 10
