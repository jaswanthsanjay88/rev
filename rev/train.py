"""
Training script for rev: LoRA fine-tuning with Cross-Entropy + Ranked Probability Score.
"""

import argparse
import os
import random
import time
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from .model import DecisionModel, encode


def question_loss(z: torch.Tensor, label: int, qtype: str, ord_w: float = 0.0):
    dev = z.device
    y = torch.tensor([label], device=dev)
    loss = F.cross_entropy(z[None], y)
    if qtype == "score" and ord_w > 0.0:
        p = F.softmax(z, dim=-1)
        observed_cdf = (torch.arange(len(p) - 1, device=dev) >= label).to(p.dtype)
        rps = (p.cumsum(-1)[:-1] - observed_cdf).square().mean()
        loss = loss + ord_w * rps
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--lora", type=int, default=16)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--out", default="runs/rev")
    args = parser.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on: {dev} | Base: {args.base} | LoRA rank: {args.lora} | LR: {args.lr}")

    tok = AutoTokenizer.from_pretrained(args.base)
    model = DecisionModel(base_name=args.base, lora_r=args.lora, device=dev)

    # Sample customer support / policy data for training
    train_data = [
        {
            "state": "I was charged $49 twice for my subscription this month. Please refund the duplicate transaction!",
            "questions": [
                {"instr": "Which department should handle this ticket?", "options": ["billing", "technical_support", "sales", "returns"], "label": 0, "qtype": "choice"},
                {"instr": "Is this ticket urgent?", "options": ["no", "yes"], "label": 1, "qtype": "noul"},
                {"instr": "Customer frustration level", "options": ["calm", "frustrated", "furious"], "label": 1, "qtype": "score"}
            ]
        },
        {
            "state": "My app keeps crashing immediately upon launch on iOS 18. I tried reinstalling.",
            "questions": [
                {"instr": "Which department should handle this ticket?", "options": ["billing", "technical_support", "sales", "returns"], "label": 1, "qtype": "choice"},
                {"instr": "Is this ticket urgent?", "options": ["no", "yes"], "label": 0, "qtype": "noul"},
                {"instr": "Customer frustration level", "options": ["calm", "frustrated", "furious"], "label": 1, "qtype": "score"}
            ]
        },
        {
            "state": "I would like to request a bulk enterprise discount quote for 250 team licenses.",
            "questions": [
                {"instr": "Which department should handle this ticket?", "options": ["billing", "technical_support", "sales", "returns"], "label": 2, "qtype": "choice"},
                {"instr": "Is this ticket urgent?", "options": ["no", "yes"], "label": 0, "qtype": "noul"},
                {"instr": "Customer frustration level", "options": ["calm", "frustrated", "furious"], "label": 0, "qtype": "score"}
            ]
        }
    ] * 20

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    model.train()

    for ep in range(args.epochs):
        random.shuffle(train_data)
        total_loss = 0.0
        optimizer.zero_grad()

        for i, sample in enumerate(train_data):
            enc = encode(tok, sample)
            logits_list = model.forward_batch([enc])[0]

            loss = 0.0
            for z, q in zip(logits_list, sample["questions"]):
                loss = loss + question_loss(z, q["label"], q.get("qtype", "choice"), ord_w=0.5)

            (loss / args.accum).backward()
            total_loss += loss.item()

            if (i + 1) % args.accum == 0 or (i + 1) == len(train_data):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {ep + 1}/{args.epochs} - Loss: {total_loss / len(train_data):.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.lm.save_pretrained(args.out)
    torch.save({
        "head": model.head.state_dict(),
        "base": args.base,
        "lora": args.lora,
        "head_dim": 256,
    }, f"{args.out}/head.pt")
    tok.save_pretrained(args.out)
    print(f"✓ Saved trained model to: {args.out}")


if __name__ == "__main__":
    main()
