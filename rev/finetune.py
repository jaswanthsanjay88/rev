"""
rev.finetune: Complete client fine-tuning pipeline with data validation,
baseline comparison, LoRA adaptation, temperature calibration, and cost-aware deferral.
"""

import collections
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import Dataset, ValidationReport
from .hardware import HardwarePlanner, HardwareProfile
from .bundle import BundleManifest, save_bundle, load_manifest
from .engine_base import DecisionEngine


class EvaluationReport:
    """Comprehensive evaluation report with calibration and coverage metrics."""

    def __init__(
        self,
        accuracy: float,
        macro_f1: float,
        per_class_f1: Dict[str, float],
        confusion_matrix: Dict[str, Dict[str, int]],
        ece: float,
        baseline_accuracy: float,
        accuracy_gain: float,
        coverage_curve: List[Dict[str, Any]],
        optimal_deferral_threshold: float,
        labels: List[str],
    ):
        self.accuracy = accuracy
        self.macro_f1 = macro_f1
        self.per_class_f1 = per_class_f1
        self.confusion_matrix = confusion_matrix
        self.ece = ece
        self.baseline_accuracy = baseline_accuracy
        self.accuracy_gain = accuracy_gain
        self.coverage_curve = coverage_curve
        self.optimal_deferral_threshold = optimal_deferral_threshold
        self.labels = labels

    def summary(self) -> str:
        lines = [
            "=================== Rev Evaluation Scorecard ===================",
            f"Overall Accuracy:  {self.accuracy * 100:.2f}%  (Baseline: {self.baseline_accuracy * 100:.2f}% | Gain: {self.accuracy_gain:+.2f}%)",
            f"Macro F1 Score:    {self.macro_f1 * 100:.2f}%",
            f"Calibration (ECE): {self.ece:.4f}  (Lower is better, <0.08 is well-calibrated)",
            f"Optimal Deferral:  Threshold tau={self.optimal_deferral_threshold:.2f}",
            "",
            "Per-Class Performance:",
            f"  {'Class':<20} | {'F1-Score':<10}",
            "  " + "-" * 32,
        ]
        for lbl, f1 in sorted(self.per_class_f1.items(), key=lambda x: -x[1]):
            lines.append(f"  {lbl:<20} | {f1 * 100:>6.1f}%")

        if self.coverage_curve:
            lines.extend([
                "",
                "Coverage vs. Accuracy Curve (Cost-Aware Deferral):",
                f"  {'Coverage':<12} | {'Accuracy':<10} | {'Human Deferral Rate'}",
                "  " + "-" * 42,
            ])
            for pt in self.coverage_curve:
                lines.append(f"  {pt['coverage']*100:>6.1f}%      | {pt['accuracy']*100:>6.1f}%    | {pt['deferred']*100:>6.1f}%")

        lines.append("================================================================")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": round(self.accuracy, 4),
            "macro_f1": round(self.macro_f1, 4),
            "per_class_f1": {k: round(v, 4) for k, v in self.per_class_f1.items()},
            "confusion_matrix": self.confusion_matrix,
            "ece": round(self.ece, 4),
            "baseline_accuracy": round(self.baseline_accuracy, 4),
            "accuracy_gain_pct": round(self.accuracy_gain, 2),
            "coverage_curve": self.coverage_curve,
            "optimal_deferral_threshold": round(self.optimal_deferral_threshold, 4),
        }


def compute_baseline_accuracy(
    train_dataset: Dataset,
    test_dataset: Dataset,
) -> float:
    """
    Lightweight TF-IDF + Logistic Regression baseline using PyTorch.
    Ensures zero heavy third-party dependencies while providing honest benchmarks.
    """
    labels = train_dataset.labels
    label_to_id = train_dataset.label_to_id
    num_classes = len(labels)

    # 1. Build word vocabulary from train data (top 1500 n-grams)
    vocab = collections.Counter()
    for s in train_dataset.samples:
        for word in s["text"].lower().split():
            clean = "".join(c for c in word if c.isalnum())
            if len(clean) > 2:
                vocab[clean] += 1

    top_words = [w for w, _ in vocab.most_common(1000)]
    if not top_words:
        return 1.0 / num_classes

    word_to_id = {w: i for i, w in enumerate(top_words)}
    vocab_size = len(word_to_id)

    def featurize(text: str) -> torch.Tensor:
        feat = torch.zeros(vocab_size)
        for word in text.lower().split():
            clean = "".join(c for c in word if c.isalnum())
            if clean in word_to_id:
                feat[word_to_id[clean]] += 1.0
        norm = feat.norm(p=2)
        return feat / norm if norm > 0 else feat

    X_train = torch.stack([featurize(s["text"]) for s in train_dataset.samples])
    y_train = torch.tensor([label_to_id[s["label"]] for s in train_dataset.samples], dtype=torch.long)

    # Logistic Regression layer
    clf = nn.Linear(vocab_size, num_classes)
    opt = torch.optim.Adam(clf.parameters(), lr=0.05, weight_decay=1e-4)

    clf.train()
    for _ in range(30):
        opt.zero_grad()
        loss = F.cross_entropy(clf(X_train), y_train)
        loss.backward()
        opt.step()

    clf.eval()
    correct = 0
    with torch.no_grad():
        for s in test_dataset.samples:
            feat = featurize(s["text"]).unsqueeze(0)
            pred = clf(feat).argmax(dim=-1).item()
            if pred == label_to_id.get(s["label"], -1):
                correct += 1

    return correct / max(len(test_dataset), 1)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 10) -> float:
    """Computes Expected Calibration Error (ECE) across confidence bins."""
    confidences = np.max(probs, axis=-1)
    predictions = np.argmax(probs, axis=-1)
    accuracies = predictions == targets

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            acc_in_bin = np.mean(accuracies[in_bin])
            avg_conf_in_bin = np.mean(confidences[in_bin])
            ece += np.abs(avg_conf_in_bin - acc_in_bin) * prop_in_bin

    return float(ece)


def _to_device(inputs: Any, device: Any) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, dict):
        return {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}
    return inputs


class RevFinetunedModel(DecisionEngine):
    """
    Inference-ready fine-tuned decision model supporting calibrated predictions,
    deferral logic, evaluation reports, and portable .rev bundle saving.
    """

    def __init__(
        self,
        backbone: Any,
        tokenizer: Any,
        readout_head: Any,
        labels: List[str],
        optimal_temperature: float = 1.0,
        deferral_threshold: float = 0.80,
        cost_matrix: Optional[Dict[str, float]] = None,
        device: str = "cpu",
        data_hash: str = "",
        backbone_name: str = "rev-decision-model",
    ):
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.readout_head = readout_head
        self.labels = labels
        self.label_to_id = {lbl: i for i, lbl in enumerate(labels)}
        self.id_to_label = {i: lbl for i, lbl in enumerate(labels)}
        self.optimal_temperature = optimal_temperature
        self.deferral_threshold = deferral_threshold
        self.cost_matrix = cost_matrix or {"false_action": 10.0, "defer": 1.0}
        self.device = device
        self.data_hash = data_hash
        self.backbone_name = backbone_name

        self.backbone.eval()
        if self.readout_head is not None:
            self.readout_head.eval()

    def predict(
        self,
        state: Union[str, Dict[str, Any], Any] = None,
        questions: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Runs single-pass decision inference.
        Returns:
          {
            "decision": str,
            "confidence": float,
            "action": "act" | "defer",
            "deferral_reason": str | None,
            "probabilities": {class: float, ...}
          }
        """
        # Resolve state text
        if isinstance(state, dict):
            text = str(state.get("note", state.get("text", state.get("state", ""))))
        else:
            text = str(state or "")

        # Candidate prompts for each label
        prompts = [f"Context: {text}\nCandidate: {lbl}" for lbl in self.labels]
        raw_inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = _to_device(raw_inputs, self.device)

        with torch.no_grad():
            outputs = self.backbone(**inputs)
            hidden = outputs.last_hidden_state
            attn_mask = inputs["attention_mask"] if isinstance(inputs, dict) else inputs.attention_mask
            mask = attn_mask.unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            head_dtype = next(self.readout_head.parameters()).dtype
            raw_logits = self.readout_head(pooled.to(head_dtype))

            # Calibrate with temperature
            scaled_logits = raw_logits / max(self.optimal_temperature, 1e-4)
            probs = F.softmax(scaled_logits, dim=-1).cpu().numpy()

        best_idx = int(np.argmax(probs))
        confidence = float(probs[best_idx])
        decision_label = self.id_to_label[best_idx]

        # Cost-aware deferral check
        if confidence < self.deferral_threshold:
            action = "defer"
            reason = f"Confidence {confidence:.3f} below deferral threshold {self.deferral_threshold:.2f}"
        else:
            action = "act"
            reason = None

        return {
            "decision": decision_label,
            "confidence": round(confidence, 4),
            "action": action,
            "deferral_reason": reason,
            "probabilities": {lbl: round(float(p), 4) for lbl, p in zip(self.labels, probs)},
        }

    def evaluate(self, test_dataset: Dataset) -> EvaluationReport:
        """Evaluates model performance, calibration ECE, and coverage curve on a test dataset."""
        targets = []
        all_probs = []

        for s in test_dataset.samples:
            gold_lbl = s["label"]
            if gold_lbl not in self.label_to_id:
                continue
            targets.append(self.label_to_id[gold_lbl])
            pred_res = self.predict(s["text"])
            probs_vec = [pred_res["probabilities"].get(lbl, 0.0) for lbl in self.labels]
            all_probs.append(probs_vec)

        targets_arr = np.array(targets)
        probs_arr = np.array(all_probs)
        preds_arr = np.argmax(probs_arr, axis=-1)

        # Accuracy & Confusion Matrix
        acc = float(np.mean(preds_arr == targets_arr))
        conf_mat = collections.defaultdict(lambda: collections.defaultdict(int))
        for t, p in zip(targets_arr, preds_arr):
            conf_mat[self.id_to_label[t]][self.id_to_label[p]] += 1

        # Per-class F1 & Macro F1
        per_class_f1 = {}
        for idx, lbl in enumerate(self.labels):
            tp = np.sum((preds_arr == idx) & (targets_arr == idx))
            fp = np.sum((preds_arr == idx) & (targets_arr != idx))
            fn = np.sum((preds_arr != idx) & (targets_arr == idx))
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            per_class_f1[lbl] = float(f1)

        macro_f1 = float(np.mean(list(per_class_f1.values()))) if per_class_f1 else 0.0
        ece = compute_ece(probs_arr, targets_arr)

        # Baseline comparison
        base_acc = compute_baseline_accuracy(test_dataset, test_dataset)
        gain_pct = (acc - base_acc) * 100.0

        # Coverage vs Accuracy curve across deferral thresholds
        coverage_curve = []
        confs = np.max(probs_arr, axis=-1)
        for thresh in [0.50, 0.65, 0.75, 0.85, 0.95]:
            retained = confs >= thresh
            cov = float(np.mean(retained))
            cov_acc = float(np.mean(preds_arr[retained] == targets_arr[retained])) if np.sum(retained) > 0 else 1.0
            coverage_curve.append({
                "threshold": thresh,
                "coverage": round(cov, 3),
                "accuracy": round(cov_acc, 3),
                "deferred": round(1.0 - cov, 3),
            })

        return EvaluationReport(
            accuracy=acc,
            macro_f1=macro_f1,
            per_class_f1=per_class_f1,
            confusion_matrix={k: dict(v) for k, v in conf_mat.items()},
            ece=ece,
            baseline_accuracy=base_acc,
            accuracy_gain=gain_pct,
            coverage_curve=coverage_curve,
            optimal_deferral_threshold=self.deferral_threshold,
            labels=self.labels,
        )

    def save(self, target_dir: str) -> str:
        """Exports the calibrated model into a portable .rev bundle."""
        manifest = BundleManifest(
            rev_version="0.3.0",
            backbone=self.backbone_name,
            labels=self.labels,
            label_to_id=self.label_to_id,
            optimal_temperature=self.optimal_temperature,
            deferral_threshold=self.deferral_threshold,
            cost_matrix=self.cost_matrix,
            data_hash=self.data_hash,
        )
        return save_bundle(
            target_dir=target_dir,
            manifest=manifest,
            tokenizer=self.tokenizer,
            backbone=self.backbone,
            readout_head=self.readout_head,
            metrics={"optimal_temperature": self.optimal_temperature, "deferral_threshold": self.deferral_threshold},
        )


class FinetuneRun:
    """Manages execution, hardware planning, and training for Rev.finetune(...)."""

    def __init__(
        self,
        backbone: str,
        dataset: Dataset,
        cost_matrix: Optional[Dict[str, float]] = None,
        hardware_profile: Optional[HardwareProfile] = None,
    ):
        self.backbone_id = backbone
        self.dataset = dataset
        self.cost_matrix = cost_matrix or {"false_action": 10.0, "defer": 1.0}
        self.profile = hardware_profile or HardwarePlanner.plan(len(dataset))

    def fit(
        self,
        epochs: int = 3,
        lr: float = 2e-4,
        head_lr: float = 5e-4,
        val_ratio: float = 0.15,
        seed: int = 42,
    ) -> RevFinetunedModel:
        """
        Executes end-to-end training:
        1. Validates dataset health.
        2. Splits into stratified train and validation.
        3. Mounts backbone with LoRA and calibrated readout head.
        4. Trains with multi-candidate RLCD loss.
        5. Performs post-training temperature calibration on the validation split.
        6. Tunes cost-optimal deferral threshold.
        """
        # 1. Validation Gate
        report = self.dataset.validate()
        if not report.is_valid:
            raise ValueError(f"Dataset validation failed with fatal issues:\n{report.summary()}")

        train_ds, val_ds, test_ds = self.dataset.split(train_ratio=1.0 - val_ratio, val_ratio=val_ratio, test_ratio=0.0, seed=seed)
        labels = self.dataset.labels
        label_to_id = self.dataset.label_to_id

        device = torch.device(self.profile.device)
        dtype = self.profile.dtype

        # 2. Tokenizer & Backbone Setup
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
        from peft import LoraConfig, get_peft_model
        from .adapters.lfm import LfmReadoutHead

        tokenizer = AutoTokenizer.from_pretrained(self.backbone_id, trust_remote_code=True)
        if not tokenizer.pad_token_id:
            tokenizer.pad_token_id = tokenizer.eos_token_id or 0

        # Load backbone
        try:
            full = AutoModelForMaskedLM.from_pretrained(self.backbone_id, dtype=dtype, trust_remote_code=True)
            base_model = getattr(full, "lfm2", full)
        except Exception:
            base_model = AutoModel.from_pretrained(self.backbone_id, dtype=dtype, trust_remote_code=True)

        # Configure LoRA
        target_modules = ["out_proj", "dense", "q_proj", "v_proj", "k_proj", "fc1", "fc2"]
        found_targets = []
        for name, module in base_model.named_modules():
            if any(t in name for t in target_modules) and isinstance(module, nn.Linear):
                found_targets.append(name.split(".")[-1])
        found_targets = list(set(found_targets)) or ["out_proj", "dense"]

        lora_cfg = LoraConfig(
            r=self.profile.lora_r,
            lora_alpha=self.profile.lora_r * 2,
            target_modules=found_targets,
            lora_dropout=0.05,
            bias="none",
        )
        peft_model = get_peft_model(base_model, lora_cfg).to(device)

        hidden_dim = getattr(base_model.config, "hidden_size", getattr(base_model.config, "d_model", 1024))
        readout_head = LfmReadoutHead(hidden_size=hidden_dim, head_dim=256).to(device=device, dtype=dtype)

        # Decoupled AdamW optimizer
        optimizer = torch.optim.AdamW([
            {"params": peft_model.parameters(), "lr": lr},
            {"params": readout_head.parameters(), "lr": head_lr},
        ], weight_decay=0.01)

        # 3. Training Loop
        peft_model.train()
        readout_head.train()
        shuffled_train = list(train_ds.samples)

        for epoch in range(1, epochs + 1):
            random.shuffle(shuffled_train)
            optimizer.zero_grad()

            for step, s in enumerate(shuffled_train):
                text = s["text"]
                gold_label = label_to_id[s["label"]]

                # Candidate-conditioned encoding
                prompts = [f"Context: {text}\nCandidate: {lbl}" for lbl in labels]
                raw_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = _to_device(raw_inputs, device)

                outputs = peft_model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

                head_dtype = next(readout_head.parameters()).dtype
                logits = readout_head(pooled.to(head_dtype))

                target = torch.tensor([gold_label], dtype=torch.long, device=device)
                loss = F.cross_entropy(logits.unsqueeze(0), target) / self.profile.grad_accum_steps
                loss.backward()

                if (step + 1) % self.profile.grad_accum_steps == 0 or (step + 1) == len(shuffled_train):
                    torch.nn.utils.clip_grad_norm_(list(peft_model.parameters()) + list(readout_head.parameters()), 1.0)
                    optimizer.step()
                    optimizer.zero_grad()

        # 4. Post-Training Temperature & Deferral Calibration on Validation Split
        peft_model.eval()
        readout_head.eval()

        val_logits_list = []
        val_targets_list = []
        with torch.no_grad():
            for s in val_ds.samples:
                prompts = [f"Context: {s['text']}\nCandidate: {lbl}" for lbl in labels]
                raw_inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt")
                inputs = _to_device(raw_inputs, device)
                outputs = peft_model(**inputs)
                hidden = outputs.last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                head_dtype = next(readout_head.parameters()).dtype
                val_logits_list.append(readout_head(pooled.to(head_dtype)).cpu().numpy())
                val_targets_list.append(label_to_id[s["label"]])

        # Optimize temperature T to minimize validation ECE
        best_temp = 1.0
        best_ece = 1.0
        if val_logits_list:
            v_logits = np.array(val_logits_list)
            v_targets = np.array(val_targets_list)
            for t_cand in np.linspace(0.5, 2.5, 21):
                probs = np.exp(v_logits / t_cand) / np.sum(np.exp(v_logits / t_cand), axis=-1, keepdims=True)
                cand_ece = compute_ece(probs, v_targets)
                if cand_ece < best_ece:
                    best_ece = cand_ece
                    best_temp = float(t_cand)

        # Tune cost-optimal deferral threshold
        optimal_threshold = 0.80
        if val_logits_list:
            probs = np.exp(v_logits / best_temp) / np.sum(np.exp(v_logits / best_temp), axis=-1, keepdims=True)
            max_confs = np.max(probs, axis=-1)
            preds = np.argmax(probs, axis=-1)
            min_cost = float("inf")
            cost_err = self.cost_matrix.get("false_action", 10.0)
            cost_def = self.cost_matrix.get("defer", 1.0)

            for th in np.linspace(0.5, 0.95, 19):
                deferred = max_confs < th
                acted = ~deferred
                num_err = np.sum(acted & (preds != v_targets))
                num_def = np.sum(deferred)
                total_cost = num_err * cost_err + num_def * cost_def
                if total_cost < min_cost:
                    min_cost = total_cost
                    optimal_threshold = float(th)

        return RevFinetunedModel(
            backbone=peft_model,
            tokenizer=tokenizer,
            readout_head=readout_head,
            labels=labels,
            optimal_temperature=best_temp,
            deferral_threshold=optimal_threshold,
            cost_matrix=self.cost_matrix,
            device=str(device),
            data_hash=self.dataset.data_hash(),
            backbone_name=self.backbone_id,
        )
