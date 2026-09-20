"""
Neural equivalence and branch isolation verification for rev:
1. Verifies that probs(enc) and probs_cached(..., pkv, s_len) produce mathematically identical outputs.
2. Verifies branch isolation (question 1 output is invariant whether evaluated alone or alongside question 2).
3. Measures the speedup of cached prefix evaluation vs full prefill.
"""

import sys
import time
sys.path.insert(0, ".")
import torch
import torch.nn.functional as F
from rev.model import DecisionModel, encode, load_tokenizer


def run_equivalence_test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model on {device}...")
    
    base_name = "Qwen/Qwen2.5-0.5B"
    try:
        tok = load_tokenizer(base_name)
        model = DecisionModel(base_name=base_name, lora_r=16, device=device)
        model.eval()
    except Exception as e:
        print(f"Skipping neural test (could not load {base_name}: {e})")
        return

    rec = {
        "state": (
            "Acme Corp announced Q3 2026 earnings with revenue of $4.2B, exceeding analyst expectations by 8%. "
            "However, operating margins contracted by 140 basis points due to elevated supply chain logistics costs. "
            "The board authorized a $500M share repurchase program, and forward guidance for Q4 remains cautious."
        ),
        "questions": [
            {
                "instr": "Evaluate quarterly financial performance",
                "options": [
                    "Strong beat across revenue and margin expansion",
                    "Revenue beat accompanied by margin compression",
                    "Severe miss across top and bottom line"
                ],
                "label": 1
            },
            {
                "instr": "Assess board capital allocation action",
                "options": [
                    "Approved share buyback program",
                    "Announced special cash dividend",
                    "Suspended all capital distributions"
                ],
                "label": 0
            }
        ]
    }

    print("\n--- 1. Baseline Full Prefill (Non-cached) ---")
    enc = encode(tok, rec)
    t0 = time.time()
    probs_baseline = model.probs(enc)
    t_baseline = (time.time() - t0) * 1000
    print(f"Full prefill latency: {t_baseline:.2f} ms")
    for i, p in enumerate(probs_baseline):
        print(f"  Q{i+1} probs: {p.tolist()}")

    print("\n--- 2. Prefix KV-Cached Evaluation ---")
    t0 = time.time()
    pkv, state_len, state_hash = model.compute_state_cache(tok, rec["state"])
    t_prefill = (time.time() - t0) * 1000
    print(f"Initial state prefill + cache compute: {t_prefill:.2f} ms (hash: {state_hash[:12]}...)")

    t0 = time.time()
    probs_cached = model.probs_cached(tok, rec, pkv, state_len)
    t_cached = (time.time() - t0) * 1000
    print(f"Cached prefix evaluation latency: {t_cached:.2f} ms")
    for i, p in enumerate(probs_cached):
        print(f"  Q{i+1} cached probs: {p.tolist()}")

    print(f"\nSpeedup: {t_baseline / max(t_cached, 0.001):.1f}x faster on repeated query!")

    # Check numerical equivalence
    for i in range(len(probs_baseline)):
        diff = torch.max(torch.abs(probs_baseline[i] - probs_cached[i])).item()
        print(f"Max absolute delta for Question {i+1}: {diff:.6e}")
        assert diff < 1e-4, f"Delta {diff} exceeds threshold 1e-4!"

    print("[PASS] Mathematical equivalence verified: probs(enc) == probs_cached(...)")

    print("\n--- 3. Branch Isolation Verification ---")
    # Evaluate Question 1 alone against the cache
    rec_q1_only = {
        "state": rec["state"],
        "questions": [rec["questions"][0]]
    }
    probs_q1_alone = model.probs_cached(tok, rec_q1_only, pkv, state_len)
    delta_iso = torch.max(torch.abs(probs_cached[0] - probs_q1_alone[0])).item()
    print(f"Branch isolation delta (Q1 alone vs Q1 in batch): {delta_iso:.6e}")
    assert delta_iso < 1e-4, f"Branch isolation failed: delta {delta_iso}"

    print("[PASS] Branch isolation verified: question evaluations do not leak across batch rows!")
    print("\nAll neural equivalence tests passed successfully!")


if __name__ == "__main__":
    run_equivalence_test()
