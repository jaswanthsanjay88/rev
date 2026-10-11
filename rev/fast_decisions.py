"""
rev.fast_decisions: Dataset loader, parser, and converter for fastino/fast-decisions.

Bridges GLiNER2.5-Decide's 17-domain enterprise benchmark into rev's TypeSafe System 1 format:
- State prefix <- input document
- Questions dict <- output classifications (mapped to Noul, Score, or Choice)
"""

import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
from huggingface_hub import hf_hub_download

DOMAINS: List[str] = [
    "agent_handoff",
    "banking_intent",
    "benefits_request",
    "clinic_request",
    "document_type",
    "email_triage",
    "news_topic",
    "paper_field",
    "product_feedback",
    "restaurant_review",
    "review_sentiment",
    "screen_tags",
    "sports_recap",
    "support_intent",
    "support_topic",
    "ticket_route",
    "travel_request",
]

# Known ordinal scales that should be mapped to Score questions
ORDINAL_SCALES: Dict[str, List[str]] = {
    "urgency": ["low", "normal", "high", "critical"],
    "sentiment": ["negative", "neutral", "positive"],
    "rating": ["1", "2", "3", "4", "5"],
}


def is_binary_labels(labels: List[str]) -> bool:
    """Checks if labels represent a binary boolean choice (mapped to Noul)."""
    norm = [l.lower() for l in labels]
    return set(norm) in [{"yes", "no"}, {"true", "false"}, {"on", "off"}]


def is_ordinal_labels(task_name: str, labels: List[str]) -> bool:
    """Checks if labels represent an ordered ordinal rubric (mapped to Score)."""
    norm_labels = [str(l).lower() for l in labels]
    for scale_key, scale_labels in ORDINAL_SCALES.items():
        if scale_key in task_name.lower() or set(norm_labels) == set(scale_labels):
            return True
    return False


def convert_fast_decision_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Converts a single fastino/fast-decisions raw row into rev's internal record format:
    {
      "state": str,
      "questions": [
        {
          "id": task_name,
          "instr": str,
          "options": list[str],
          "label": int,
          "qtype": "choice" | "noul" | "score",
          "gold": str
        }
      ]
    }
    """
    state_text = row.get("input", "")
    classifications = row.get("output", {}).get("classifications", [])
    questions = []

    for head in classifications:
        task_name = head["task"]
        labels = head["labels"]
        true_labels = head["true_label"]
        gold_str = true_labels[0] if true_labels else labels[0]

        if gold_str not in labels:
            continue

        label_idx = labels.index(gold_str)

        # Infer rev question type
        if is_binary_labels(labels):
            qtype = "noul"
            # Noul maps index 1 to True ('yes') and 0 to False ('no')
            options = ["no", "yes"] if set(labels) == {"yes", "no"} else labels
            gold_idx = 1 if gold_str.lower() in ("yes", "true", "on") else 0
        elif is_ordinal_labels(task_name, labels):
            qtype = "score"
            options = labels
            gold_idx = label_idx
        else:
            qtype = "choice"
            options = labels
            gold_idx = label_idx

        # Natural question instruction formatting
        instr_name = task_name.replace("_", " ").title()
        instr = f"Classify the {instr_name} for this request."

        questions.append(
            {
                "id": task_name,
                "instr": instr,
                "options": options,
                "label": gold_idx,
                "qtype": qtype,
                "gold": gold_str,
                "multi_label": head.get("multi_label", False),
            }
        )

    return {
        "state": state_text,
        "questions": questions,
    }


def to_system_one_request(record: Dict[str, Any]) -> Dict[str, Any]:
    """Converts a converted record into a TypeSafe SystemOneRequest payload."""
    questions_dict = {}
    for q in record["questions"]:
        if q["qtype"] == "noul":
            questions_dict[q["id"]] = {
                "type": "noul",
                "instructions": q["instr"],
                "criteria": {"true": "Affirmative", "false": "Negative"},
            }
        elif q["qtype"] == "score":
            questions_dict[q["id"]] = {
                "type": "score",
                "instructions": q["instr"],
                "criteria": q["options"],
            }
        else:
            questions_dict[q["id"]] = {
                "type": "choice",
                "instructions": q["instr"],
                "criteria": {opt: opt for opt in q["options"]},
            }

    return {
        "state": record["state"],
        "questions": questions_dict,
    }


class FastDecisionsDataset:
    """
    Manager and loader for the fastino/fast-decisions benchmark across all 17 enterprise domains.
    """

    def __init__(
        self,
        repo_id: str = "fastino/fast-decisions",
        domains: Optional[Union[str, List[str]]] = None,
        cache_dir: Optional[str] = None,
    ):
        self.repo_id = repo_id
        self.cache_dir = cache_dir

        if domains is None or domains == "all":
            self.domains = DOMAINS
        elif isinstance(domains, str):
            self.domains = [d.strip() for d in domains.split(",") if d.strip()]
        else:
            self.domains = list(domains)

        self._records_by_domain: Dict[str, List[Dict[str, Any]]] = {}

    def load(self, force_download: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """Loads and parses specified domains from Hugging Face Hub."""
        for d in self.domains:
            filename = f"{d}.jsonl"
            try:
                path = hf_hub_download(
                    repo_id=self.repo_id,
                    filename=filename,
                    repo_type="dataset",
                    cache_dir=self.cache_dir,
                    force_download=force_download,
                )
                domain_records = []
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            raw = json.loads(line)
                            domain_records.append(convert_fast_decision_row(raw))
                self._records_by_domain[d] = domain_records
            except Exception as e:
                print(f"Warning: Could not load domain '{d}' ({e})")

        return self._records_by_domain

    def all_records(self) -> List[Dict[str, Any]]:
        """Returns all records aggregated across all loaded domains."""
        if not self._records_by_domain:
            self.load()
        records = []
        for d, recs in self._records_by_domain.items():
            records.extend(recs)
        return records

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Creates stratified train, validation, and test splits per domain."""
        if not self._records_by_domain:
            self.load()

        rng = random.Random(seed)
        train_set, val_set, test_set = [], [], []

        for d, recs in self._records_by_domain.items():
            shuffled = list(recs)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)

            train_set.extend(shuffled[:n_train])
            val_set.extend(shuffled[n_train : n_train + n_val])
            test_set.extend(shuffled[n_train + n_val :])

        return train_set, val_set, test_set

    def summary(self) -> Dict[str, Any]:
        """Returns a high-level summary of domains, head count, and sample sizes."""
        if not self._records_by_domain:
            self.load()
        stats = {}
        for d, recs in self._records_by_domain.items():
            tasks = set()
            for r in recs:
                for q in r["questions"]:
                    tasks.add(q["id"])
            stats[d] = {
                "num_samples": len(recs),
                "tasks": sorted(list(tasks)),
                "heads_per_sample": len(tasks),
            }
        return {
            "total_domains": len(self._records_by_domain),
            "total_samples": sum(s["num_samples"] for s in stats.values()),
            "domain_stats": stats,
        }
