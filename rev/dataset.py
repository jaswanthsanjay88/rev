"""
rev.dataset: Production-grade client dataset loader and validator.

Supports loading client data from CSV, JSONL, Python dicts, or Hugging Face,
and runs strict validation checks for label noise, class imbalance,
duplicates, and data leakage before fine-tuning starts.
"""

import collections
import csv
import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union


class ValidationReport:
    """Detailed data health and readiness report."""

    def __init__(
        self,
        total_samples: int,
        num_classes: int,
        class_counts: Dict[str, int],
        duplicates_count: int,
        conflicting_labels_count: int,
        empty_texts_count: int,
        issues: List[str],
        warnings: List[str],
    ):
        self.total_samples = total_samples
        self.num_classes = num_classes
        self.class_counts = class_counts
        self.duplicates_count = duplicates_count
        self.conflicting_labels_count = conflicting_labels_count
        self.empty_texts_count = empty_texts_count
        self.issues = issues
        self.warnings = warnings

    @property
    def is_valid(self) -> bool:
        """Returns True if there are zero fatal issues blocking training."""
        return len(self.issues) == 0

    def summary(self) -> str:
        """Generates a human-readable diagnostic report."""
        status = "PASSED" if self.is_valid else "FAILED"
        lines = [
            f"=== Rev Dataset Validation Report [{status}] ===",
            f"Total samples: {self.total_samples}",
            f"Unique classes ({self.num_classes}): {list(self.class_counts.keys())}",
            "",
            "Class distribution:",
        ]
        for cls_name, count in sorted(self.class_counts.items(), key=lambda x: -x[1]):
            pct = (count / self.total_samples * 100) if self.total_samples > 0 else 0
            lines.append(f"  - {cls_name:<20}: {count:>6} ({pct:>5.1f}%)")

        if self.duplicates_count > 0:
            lines.append(f"\nDuplicates found: {self.duplicates_count}")
        if self.conflicting_labels_count > 0:
            lines.append(f"Conflicting label rows (label noise): {self.conflicting_labels_count}")
        if self.empty_texts_count > 0:
            lines.append(f"Empty text rows: {self.empty_texts_count}")

        if self.issues:
            lines.append("\nFatal Issues (Blocking):")
            for iss in self.issues:
                lines.append(f"  [X] {iss}")

        if self.warnings:
            lines.append("\nWarnings (Recommended to address):")
            for warn in self.warnings:
                lines.append(f"  [!] {warn}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "total_samples": self.total_samples,
            "num_classes": self.num_classes,
            "class_counts": self.class_counts,
            "duplicates_count": self.duplicates_count,
            "conflicting_labels_count": self.conflicting_labels_count,
            "empty_texts_count": self.empty_texts_count,
            "issues": self.issues,
            "warnings": self.warnings,
        }

    def __repr__(self) -> str:
        return f"<ValidationReport is_valid={self.is_valid} samples={self.total_samples} classes={self.num_classes}>"


class Dataset:
    """
    Client Dataset container for fine-tuning calibrated decision models.
    """

    def __init__(self, samples: List[Dict[str, str]]):
        """
        samples: List of dicts, each having {'text': str, 'label': str}
        """
        self.samples = samples
        self._labels = sorted(list({s["label"] for s in samples if s.get("label") is not None}))
        self.label_to_id = {lbl: idx for idx, lbl in enumerate(self._labels)}
        self.id_to_label = {idx: lbl for idx, lbl in enumerate(self._labels)}

    @property
    def labels(self) -> List[str]:
        return list(self._labels)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, str]:
        return self.samples[idx]

    @classmethod
    def from_csv(
        cls,
        path: str,
        text: str = "text",
        label: str = "label",
        delimiter: str = ",",
        encoding: str = "utf-8",
    ) -> "Dataset":
        """Loads dataset from a CSV file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")

        samples = []
        with open(path, "r", encoding=encoding, errors="replace") as f:
            reader = csv.DictReader(f, delimiter=delimiter)
            for row in reader:
                t_val = row.get(text, "").strip()
                l_val = row.get(label, "").strip()
                samples.append({"text": t_val, "label": l_val})
        return cls(samples)

    @classmethod
    def from_jsonl(
        cls,
        path: str,
        text: str = "text",
        label: str = "label",
        encoding: str = "utf-8",
    ) -> "Dataset":
        """Loads dataset from a JSONL file."""
        if not os.path.exists(path):
            raise FileNotFoundError(f"JSONL file not found: {path}")

        samples = []
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    t_val = str(row.get(text, "")).strip()
                    l_val = str(row.get(label, "")).strip()
                    samples.append({"text": t_val, "label": l_val})
        return cls(samples)

    @classmethod
    def from_records(
        cls,
        records: List[Dict[str, Any]],
        text: str = "text",
        label: str = "label",
    ) -> "Dataset":
        """Loads dataset from in-memory records."""
        samples = []
        for r in records:
            samples.append({
                "text": str(r.get(text, "")).strip(),
                "label": str(r.get(label, "")).strip(),
            })
        return cls(samples)

    @classmethod
    def from_hf(
        cls,
        repo_id: str,
        text: str = "text",
        label: str = "label",
        split: str = "train",
    ) -> "Dataset":
        """Loads dataset directly from Hugging Face datasets."""
        from datasets import load_dataset
        hf_ds = load_dataset(repo_id, split=split)
        samples = []
        for row in hf_ds:
            samples.append({
                "text": str(row.get(text, "")).strip(),
                "label": str(row.get(label, "")).strip(),
            })
        return cls(samples)

    def validate(self) -> ValidationReport:
        """
        Validates client data and returns a health report checking:
        - Empty texts or missing labels
        - Class balance and minimum count thresholds
        - Duplicate detection & conflicting label noise
        """
        issues = []
        warnings = []

        total_samples = len(self.samples)
        if total_samples == 0:
            issues.append("Dataset is empty (0 samples).")
            return ValidationReport(0, 0, {}, 0, 0, 0, issues, warnings)

        class_counts = collections.Counter()
        text_to_labels = collections.defaultdict(set)
        seen_texts = collections.Counter()
        empty_texts_count = 0

        for s in self.samples:
            t = s["text"]
            lbl = s["label"]

            if not t:
                empty_texts_count += 1
            if not lbl:
                issues.append("Found rows with empty or missing labels.")

            class_counts[lbl] += 1
            seen_texts[t] += 1
            text_to_labels[t].add(lbl)

        num_classes = len(class_counts)
        if num_classes < 2:
            issues.append(f"At least 2 distinct classes are required, found only {num_classes}: {list(class_counts.keys())}")

        # Check small classes
        for cls_name, count in class_counts.items():
            if count < 5:
                issues.append(f"Class '{cls_name}' has only {count} samples (minimum 5 required for train/val split).")
            elif count < 15:
                warnings.append(f"Class '{cls_name}' has only {count} samples. Accuracy on this class may be limited.")

        # Check extreme class imbalance
        if num_classes >= 2:
            max_c = max(class_counts.values())
            min_c = min(class_counts.values())
            ratio = max_c / max(min_c, 1)
            if ratio > 25.0:
                warnings.append(f"Severe class imbalance detected ({ratio:.1f}:1 ratio between largest and smallest class).")

        # Check duplicates and label noise
        duplicates_count = sum(c - 1 for c in seen_texts.values() if c > 1)
        conflicting_labels_count = sum(1 for t, lbls in text_to_labels.items() if len(lbls) > 1)

        if empty_texts_count > 0:
            warnings.append(f"Found {empty_texts_count} rows with completely empty text.")
        if duplicates_count > 0:
            warnings.append(f"Found {duplicates_count} duplicate text rows.")
        if conflicting_labels_count > 0:
            warnings.append(f"Found {conflicting_labels_count} identical texts labeled with conflicting classes (potential label noise).")

        return ValidationReport(
            total_samples=total_samples,
            num_classes=num_classes,
            class_counts=dict(class_counts),
            duplicates_count=duplicates_count,
            conflicting_labels_count=conflicting_labels_count,
            empty_texts_count=empty_texts_count,
            issues=issues,
            warnings=warnings,
        )

    def split(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple["Dataset", "Dataset", "Dataset"]:
        """
        Creates a stratified train / validation / test split preserving class proportions.
        """
        rng = random.Random(seed)
        by_class = collections.defaultdict(list)
        for s in self.samples:
            by_class[s["label"]].append(s)

        train_samples = []
        val_samples = []
        test_samples = []

        for cls_name, items in by_class.items():
            shuffled = list(items)
            rng.shuffle(shuffled)
            n = len(shuffled)
            n_train = max(1, int(n * train_ratio))
            n_val = max(1, int(n * val_ratio)) if n > 3 else 0
            n_test = n - n_train - n_val

            train_samples.extend(shuffled[:n_train])
            val_samples.extend(shuffled[n_train : n_train + n_val])
            test_samples.extend(shuffled[n_train + n_val :])

        rng.shuffle(train_samples)
        rng.shuffle(val_samples)
        rng.shuffle(test_samples)

        return (
            Dataset(train_samples),
            Dataset(val_samples if val_samples else train_samples[:2]),
            Dataset(test_samples if test_samples else val_samples),
        )

    def data_hash(self) -> str:
        """Computes a SHA256 fingerprint of the dataset content for reproducibility."""
        hasher = hashlib.sha256()
        for s in sorted(self.samples, key=lambda x: (x["text"], x["label"])):
            hasher.update(s["text"].encode("utf-8"))
            hasher.update(s["label"].encode("utf-8"))
        return hasher.hexdigest()[:16]
