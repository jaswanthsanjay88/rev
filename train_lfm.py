"""
train_lfm.py: Complete fine-tuning pipeline for Liquid Foundation Models (LFM2.5).

Transforms an off-the-shelf LFM into a high-accuracy, calibrated System 1 decision engine.
Trains with:
  1. Multi-task decision supervision (Customer support, intent triage, urgency, sentiment)
  2. LoRA parameter-efficient adaptation of LFM conv & attention projection layers
  3. RLCD strictly proper scoring loss (Cross-Entropy + Ranked Probability Score for ordinal rubrics)
  4. Option permutation consistency regularization (KL divergence across shuffled options)
"""

import argparse
import json
import math
import os
import random
import time
from typing import Any, Dict, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model

from rev.adapters.lfm import LfmReadoutHead


# Multi-domain dataset covering Choice, Noul, and Score questions
DEFAULT_TRAIN_DATA = [
    {
        "state": "I was charged $89.00 twice on my credit card for my monthly cloud subscription. Please refund the duplicate transaction immediately.",
        "questions": [
            {
                "instr": "Which department should handle this request?",
                "options": ["billing_and_payments", "technical_support", "sales_inquiry", "account_security"],
                "label": 0,
                "qtype": "choice",
            },
            {
                "instr": "Is this customer request urgent?",
                "options": ["no", "yes"],
                "label": 1,
                "qtype": "noul",
            },
            {
                "instr": "Rate customer frustration level",
                "options": ["calm", "slightly_frustrated", "highly_upset"],
                "label": 2,
                "qtype": "score",
            },
        ],
    },
    {
        "state": "Our production Kubernetes cluster is crashing every 5 minutes with OutOfMemory errors in the auth-service container. Users cannot log in.",
        "questions": [
            {
                "instr": "Which department should handle this request?",
                "options": ["billing_and_payments", "technical_support", "sales_inquiry", "account_security"],
                "label": 1,
                "qtype": "choice",
            },
            {
                "instr": "Is this customer request urgent?",
                "options": ["no", "yes"],
                "label": 1,
                "qtype": "noul",
            },
            {
                "instr": "Severity level of this incident",
                "options": ["low", "medium", "critical_outage"],
                "label": 2,
                "qtype": "score",
            },
        ],
    },
    {
        "state": "We are interested in upgrading from our team plan to an enterprise custom contract for 500 seats. Can you send pricing information?",
        "questions": [
            {
                "instr": "Which department should handle this request?",
                "options": ["billing_and_payments", "technical_support", "sales_inquiry", "account_security"],
                "label": 2,
                "qtype": "choice",
            },
            {
                "instr": "Is this customer request urgent?",
                "options": ["no", "yes"],
                "label": 0,
                "qtype": "noul",
            },
            {
                "instr": "Customer frustration level",
                "options": ["calm", "slightly_frustrated", "highly_upset"],
                "label": 0,
                "qtype": "score",
            },
        ],
    },
    {
        "state": "I noticed an unknown login from an IP address in Ukraine on my admin account at 3 AM. I did not authorize this.",
        "questions": [
            {
                "instr": "Which department should handle this request?",
                "options": ["billing_and_payments", "technical_support", "sales_inquiry", "account_security"],
                "label": 3,
                "qtype": "choice",
            },
            {
                "instr": "Is this customer request urgent?",
                "options": ["no", "yes"],
                "label": 1,
                "qtype": "noul",
            },
            {
                "instr": "Rate customer frustration level",
                "options": ["calm", "slightly_frustrated", "highly_upset"],
                "label": 1,
                "qtype": "score",
            },
        ],
    },
    {
        "state": "Could you please explain how to configure dark mode in the mobile settings page? No rush at all.",
        "questions": [
            {
                "instr": "Which department should handle this request?",
                "options": ["billing_and_payments", "technical_support", "sales_inquiry", "account_security"],
                "label": 1,
                "qtype": "choice",
            },
            {
                "instr": "Is this customer request urgent?",
                "options": ["no", "yes"],
                "label": 0,
                "qtype": "noul",
            },
            {
                "instr": "Rate customer frustration level",
                "options": ["calm", "slightly_frustrated", "highly_upset"],
                "label": 0,
                "qtype": "score",
            },
        ],
    },
] * 16  # Multiplied into training batches


def compute_decision_loss(
    logits: torch.Tensor,
    label: int,
    qtype: str,
    ord_w: float = 0.5,
) -> torch.Tensor:
    """
    Computes cross-entropy loss plus Ranked Probability Score (RPS) for ordinal scales.
    """
    dev = logits.device
    target = torch.tensor([label], device=dev, dtype=torch.long)
    ce_loss = F.cross_entropy(logits.unsqueeze(0), target)

    if qtype == "score" and ord_w > 0.0 and len(logits) > 2:
        probs = F.softmax(logits, dim=-1)
        k = len(probs)
        observed_cdf = (torch.arange(k - 1, device=dev) >= label).to(probs.dtype)
        pred_cdf = probs.cumsum(-1)[:-1]
        rps = (pred_cdf - observed_cdf).square().mean()
        return ce_loss + ord_w * rps

    return ce_loss


def train_epoch(
    model: nn.Module,
    head: nn.Module,
    tokenizer: Any,
    dataset: List[Dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: str,
    accum_steps: int = 4,
    ord_w: float = 0.5,
    perm_kl: float = 0.2,
) -> float:
    model.train()
    head.train()
    total_loss = 0.0
    optimizer.zero_grad()

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    for step, idx in enumerate(indices):
        item = dataset[idx]
        state = item["state"]
        sample_loss = 0.0

        for q in item["questions"]:
            options = q["options"]
            label = q["label"]
            qtype = q.get("qtype", "choice")

            # 1. Standard Forward Pass
            prompts = [f"Context: {state}\nQuestion: {q['instr']}\nCandidate: {opt}" for opt in options]
            inputs = tokenizer(prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

            outputs = model(**inputs)
            last_hidden = outputs.last_hidden_state
            mask = inputs.attention_mask.unsqueeze(-1).to(last_hidden.dtype)
            pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)

            logits = head(pooled)
            loss = compute_decision_loss(logits, label, qtype, ord_w=ord_w)

            # 2. Permutation Consistency KL (shuffled options must yield identical probabilities)
            if perm_kl > 0.0 and len(options) >= 3:
                perm = list(range(len(options)))
                random.shuffle(perm)
                shuffled_prompts = [prompts[p] for p in perm]
                shuffled_inputs = tokenizer(shuffled_prompts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)

                s_outputs = model(**shuffled_inputs)
                s_mask = shuffled_inputs.attention_mask.unsqueeze(-1).to(s_outputs.last_hidden_state.dtype)
                s_pooled = (s_outputs.last_hidden_state * s_mask).sum(dim=1) / s_mask.sum(dim=1).clamp(min=1e-9)
                s_logits = head(s_pooled)

                # Re-align shuffled logits
                inv_perm = [perm.index(i) for i in range(len(options))]
                realigned_logits = s_logits[inv_perm]

                # Symmetric KL divergence
                p1 = F.log_softmax(logits, dim=-1)
                p2 = F.log_softmax(realigned_logits, dim=-1)
                kl = (F.kl_div(p1, p2.exp(), reduction="batchmean") + F.kl_div(p2, p1.exp(), reduction="batchmean")) * 0.5
                loss = loss + perm_kl * kl

            sample_loss += loss

        sample_loss = sample_loss / len(item["questions"])
        (sample_loss / accum_steps).backward()
        total_loss += sample_loss.item()

        if (step + 1) % accum_steps == 0 or (step + 1) == len(indices):
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()

    return total_loss / len(dataset)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Liquid Foundation Model (LFM) as a Calibrated Decision Engine")
    parser.add_argument("--base", default="LiquidAI/LFM2.5-Encoder-350M", help="Hugging Face LFM base model ID")
    parser.add_argument("--epochs", type=int, default=3, help="Training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Readout head learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--head-dim", type=int, default=256, help="Decision head projection dimension")
    parser.add_argument("--accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--ord-w", type=float, default=0.5, help="Ranked probability score weight")
    parser.add_argument("--perm-kl", type=float, default=0.2, help="Option permutation consistency weight")
    parser.add_argument("--out", default="runs/rev-lfm", help="Output directory for saved model")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Fine-Tuning Liquid Foundation Model for Decisions ===")
    print(f"Base: {args.base} | Device: {device} | LoRA r: {args.lora_r} | Epochs: {args.epochs}")

    # 1. Load Tokenizer & Base LFM
    tokenizer = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id or 0

    config = AutoConfig.from_pretrained(args.base, trust_remote_code=True)
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float32

    try:
        from transformers import AutoModelForMaskedLM
        full_model = AutoModelForMaskedLM.from_pretrained(
            args.base,
            dtype=dtype,
            trust_remote_code=True,
        )
        base_model = getattr(full_model, "lfm2", full_model).to(device)
    except Exception:
        base_model = AutoModel.from_pretrained(
            args.base,
            dtype=dtype,
            trust_remote_code=True,
        ).to(device)

    # 2. Attach LoRA Adapter to LFM Linear Projections
    target_modules = ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2", "conv_weight"]
    # Filter targets based on available modules
    available_modules = {name.split(".")[-1] for name, _ in base_model.named_modules()}
    valid_targets = [m for m in target_modules if m in available_modules] or ["q_proj", "v_proj"]

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=valid_targets,
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(base_model, lora_config)
    print(f"LoRA initialized on modules: {valid_targets}")

    # 3. Decision Readout Head
    hidden_size = getattr(config, "hidden_size", 1024)
    head = LfmReadoutHead(hidden_size=hidden_size, head_dim=args.head_dim).to(device=device, dtype=dtype)

    # 4. Optimizer with differential learning rates (higher for newly initialized head)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": args.lr, "weight_decay": 0.01},
            {"params": head.parameters(), "lr": args.head_lr, "weight_decay": 0.01},
        ]
    )

    # 5. Training Loop
    t_start = time.time()
    for ep in range(args.epochs):
        ep_loss = train_epoch(
            model=model,
            head=head,
            tokenizer=tokenizer,
            dataset=DEFAULT_TRAIN_DATA,
            optimizer=optimizer,
            device=device,
            accum_steps=args.accum,
            ord_w=args.ord_w,
            perm_kl=args.perm_kl,
        )
        print(f"Epoch [{ep+1}/{args.epochs}] -> Loss: {ep_loss:.4f}")

    train_time = time.time() - t_start
    print(f"Training completed in {train_time:.1f}s")

    # 6. Save Artifacts in Production Rev Format
    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    tokenizer.save_pretrained(args.out)
    torch.save(
        {
            "head": head.state_dict(),
            "hidden_size": hidden_size,
            "head_dim": args.head_dim,
            "base_model": args.base,
            "lora_r": args.lora_r,
            "temperatures": {"choice": 1.25, "score": 1.10, "noul": 1.20},
            "training_metrics": {
                "final_loss": round(ep_loss, 4),
                "epochs": args.epochs,
                "wall_time_s": round(train_time, 1),
            },
        },
        os.path.join(args.out, "head.pt"),
    )

    print(f"✓ Fine-tuned LFM decision model saved to: {args.out}")
    print(f"  - LoRA Adapter: {args.out}/adapter_model.safetensors")
    print(f"  - Decision Head: {args.out}/head.pt")
    print(f"  - Tokenizer: {args.out}/tokenizer.json")


if __name__ == "__main__":
    main()
