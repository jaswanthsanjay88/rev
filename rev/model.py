"""
Decision model: causal LM backbone + block-causal branch mask + pointer readout.
"""

import hashlib
import math
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

# Reuse existing Qwen special tokens as delimiters (state, q, opt, /opt, decide)
SPECIAL = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]
MAX_STATE, MAX_BRANCH = 384, 1024

_SPECIAL_RE = re.compile(r"<\|([A-Za-z0-9_]+)\|>")


def load_tokenizer(name="Qwen/Qwen2.5-0.5B"):
    return AutoTokenizer.from_pretrained(name)


def user_tokens(tok, text: str):
    """
    Tokenizes user-supplied text so it can never produce control tokens.
    Rewrites `<|name|>` to `<¦name¦>` before tokenization.
    """
    return tok(_SPECIAL_RE.sub(r"<¦\1¦>", str(text)), add_special_tokens=False).input_ids


OPT_NONE, OPT_DECIDE = -1, -2


def encode(tok, rec, max_state=MAX_STATE, max_branch=MAX_BRANCH):
    """
    Packs state + multiple questions into one token sequence.
    Branch position IDs restart right after state so question order has zero bias.
    """
    state_tokens = user_tokens(tok, rec["state"])
    S = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    ids = list(S)
    seg = [0] * len(S)
    pos = list(range(len(S)))
    opt = [OPT_NONE] * len(S)

    q_id, o_id, c_id, d_id = [tok.convert_tokens_to_ids(t) for t in SPECIAL[1:]]
    decide_idx, opt_idx = [], []
    p0 = len(S)

    for k, q in enumerate(rec["questions"], start=1):
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]

        base = len(ids)
        br_pos = list(range(p0, p0 + len(br)))  # Position IDs restart at len(state)
        br_opt = [OPT_NONE] * len(instr) + [j for j, sp in enumerate(spans) for _ in sp] + [OPT_DECIDE]

        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)

        ids += br
        seg += [k] * len(br)
        pos += br_pos
        opt += br_opt
        decide_idx.append(base + len(br) - 1)
        opt_idx.append([base + e for e in ends])

    return {
        "ids": ids,
        "seg": seg,
        "pos": pos,
        "opt": opt,
        "decide_idx": decide_idx,
        "opt_idx": opt_idx,
        "labels": [q.get("label", 0) for q in rec["questions"]],
    }


def encode_state(tok, state_text: str, max_state: int = MAX_STATE):
    """
    Encodes the state prefix once.
    Returns input_ids, position_ids, and prefix token length.
    """
    state_tokens = user_tokens(tok, state_text)
    prefix_ids = [tok.convert_tokens_to_ids(SPECIAL[0])] + state_tokens[: max_state - 1]
    return {
        "ids": prefix_ids,
        "pos": list(range(len(prefix_ids))),
        "length": len(prefix_ids),
    }


def encode_question_branches(tok, questions: list[dict], state_len: int):
    """
    Encodes each question as an independent branch whose position IDs begin at state_len.
    Branches are evaluated as separate batch rows against the shared prefix cache.
    """
    q_id, o_id, c_id, d_id = [tok.convert_tokens_to_ids(t) for t in SPECIAL[1:]]
    branches = []

    for q in questions:
        instr = [q_id] + user_tokens(tok, q["instr"])
        spans = [[o_id] + user_tokens(tok, o) + [c_id] for o in q["options"]]
        br = instr + [t for sp in spans for t in sp] + [d_id]

        br_pos = list(range(state_len, state_len + len(br)))
        ends, cursor = [], len(instr)
        for sp in spans:
            cursor += len(sp)
            ends.append(cursor - 1)

        branches.append({
            "ids": br,
            "pos": br_pos,
            "decide_idx": len(br) - 1,
            "opt_idx": ends,
            "num_opts": len(spans),
        })

    max_len = max(len(b["ids"]) for b in branches) if branches else 0
    return {"branches": branches, "max_len": max_len}


def clone_or_expand_past_key_values(past_key_values, batch_size: int = 1):
    if past_key_values is None:
        return None
    from transformers.cache_utils import DynamicCache

    # Modern transformers (v4.45+): past_key_values.layers contains DynamicLayer instances
    if hasattr(past_key_values, "layers") and len(past_key_values.layers) > 0:
        new_data = []
        for l in past_key_values.layers:
            k, v = l.keys, l.values
            if batch_size == 1:
                k_out = k.clone()
                v_out = v.clone()
            else:
                k_out = k.expand(batch_size, -1, -1, -1).contiguous()
                v_out = v.expand(batch_size, -1, -1, -1).contiguous()
            sw = getattr(l, "_sliding_window_tensor", None)
            new_data.append((k_out, v_out, sw) if sw is not None else (k_out, v_out))
        return DynamicCache(ddp_cache_data=new_data)

    # Legacy transformers: past_key_values.key_cache and value_cache
    if hasattr(past_key_values, "key_cache") and hasattr(past_key_values, "value_cache"):
        new_cache = DynamicCache()
        if batch_size == 1:
            new_cache.key_cache = [k.clone() for k in past_key_values.key_cache]
            new_cache.value_cache = [v.clone() for v in past_key_values.value_cache]
        else:
            new_cache.key_cache = [k.expand(batch_size, -1, -1, -1).contiguous() for k in past_key_values.key_cache]
            new_cache.value_cache = [v.expand(batch_size, -1, -1, -1).contiguous() for v in past_key_values.value_cache]
        if hasattr(past_key_values, "_seen_tokens"):
            new_cache._seen_tokens = past_key_values._seen_tokens
        return new_cache

    # Tuple / list of (key, value) pairs
    if isinstance(past_key_values, (tuple, list)):
        if batch_size == 1:
            return tuple((k.clone(), v.clone()) for k, v in past_key_values)
        return tuple(
            (k.expand(batch_size, -1, -1, -1).contiguous(), v.expand(batch_size, -1, -1, -1).contiguous())
            for k, v in past_key_values
        )

    return past_key_values


def branch_mask_batch(segs, device, dtype=torch.float32):
    """
    Additive block-causal attention mask:
    Token i can attend to token j iff:
      1. j <= i (causal)
      2. seg[j] == 0 (state is visible to all) OR seg[j] == seg[i] (same question branch)
    Question branches can NEVER attend to each other.
    """
    L = max(len(s) for s in segs)
    s = torch.full((len(segs), L), -1, device=device)
    for b, seg in enumerate(segs):
        s[b, : len(seg)] = torch.tensor(seg, device=device)

    causal = torch.tril(torch.ones(L, L, dtype=torch.bool, device=device))
    same = (s[:, None, :] == s[:, :, None]) | (s[:, None, :] == 0)
    valid_key = (s != -1)[:, None, :]
    allow = (causal[None] & same & valid_key) | torch.eye(L, dtype=torch.bool, device=device)[None]

    mask = torch.zeros(len(segs), L, L, dtype=dtype, device=device)
    return mask.masked_fill(~allow, torch.finfo(dtype).min)[:, None]


class PointerHead(nn.Module):
    """Bilinear pointer readout head scoring option tokens against <decide>."""
    def __init__(self, d: int, dp: int = 256):
        super().__init__()
        self.q = nn.Linear(d, dp)
        self.k = nn.Linear(d, dp)
        self.scale = 1.0 / math.sqrt(dp)

    def forward(self, h_decide, h_opts):
        # h_decide: [d], h_opts: [K, d] -> logits [K]
        return (self.k(h_opts) @ self.q(h_decide)) * self.scale


class DecisionModel(nn.Module):
    def __init__(
        self,
        base_name="Qwen/Qwen2.5-0.5B",
        lora_r=16,
        head_dim=256,
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=None,
        gradient_checkpointing=False,
    ):
        super().__init__()
        self.device = device
        self.base_name = base_name
        self.lora_r = lora_r
        self.head_dim = head_dim

        if dtype is None:
            if "cuda" in str(device):
                dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
            else:
                dtype = torch.float32

        # Backbone only (no lm_head): prefill only
        self.lm = AutoModelForCausalLM.from_pretrained(
            base_name,
            attn_implementation="sdpa" if "cuda" in str(device) else "eager",
            torch_dtype=dtype,
            device_map=device if "cuda" in str(device) else None,
            low_cpu_mem_usage=True,
        ).model

        if gradient_checkpointing:
            self.lm.gradient_checkpointing_enable()

        if lora_r > 0:
            cfg = LoraConfig(
                task_type="FEATURE_EXTRACTION",
                r=lora_r,
                lora_alpha=2 * lora_r,
                lora_dropout=0.05,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            )
            self.lm = get_peft_model(self.lm, cfg)

        self.head = PointerHead(self.lm.config.hidden_size, dp=head_dim).to(device)
        self.to(device)

    def hidden_batch(self, encs):
        L = max(len(e["ids"]) for e in encs)
        ids = torch.full((len(encs), L), 0, device=self.device)
        pos = torch.zeros((len(encs), L), dtype=torch.long, device=self.device)
        for b, e in enumerate(encs):
            ids[b, : len(e["ids"])] = torch.tensor(e["ids"], device=self.device)
            pos[b, : len(e["pos"])] = torch.tensor(e["pos"], device=self.device)

        lm_dtype = next(self.lm.parameters()).dtype
        mask = branch_mask_batch([e["seg"] for e in encs], self.device, dtype=lm_dtype)
        return self.lm(input_ids=ids, position_ids=pos, attention_mask=mask).last_hidden_state.float()

    def forward_batch(self, encs):
        hs = self.hidden_batch(encs)
        return [
            [self.head(hs[b][d], hs[b][torch.tensor(oi, device=self.device)])
             for d, oi in zip(e["decide_idx"], e["opt_idx"])]
            for b, e in enumerate(encs)
        ]

    @torch.no_grad()
    def probs(self, enc):
        logits = self.forward_batch([enc])[0]
        return [F.softmax(z, dim=-1).cpu() for z in logits]

    @torch.no_grad()
    def compute_state_cache(self, tok, state_text: str, max_state: int = MAX_STATE):
        """
        Prefills the state prefix and caches key-value activations.
        Returns:
          - past_key_values: cached KV tensors
          - state_len: prefix token length
          - state_hash: SHA-256 hash of the normalized state
        """
        enc_state = encode_state(tok, state_text, max_state=max_state)
        ids = torch.tensor([enc_state["ids"]], device=self.device, dtype=torch.long)
        pos = torch.tensor([enc_state["pos"]], device=self.device, dtype=torch.long)
        out = self.lm(input_ids=ids, position_ids=pos, use_cache=True)
        h = hashlib.sha256(state_text.strip().encode("utf-8")).hexdigest()
        return out.past_key_values, enc_state["length"], h

    @torch.no_grad()
    def forward_with_cache(self, branch_data: dict, past_key_values, state_len: int):
        """
        Evaluates question branches against the precomputed state KV-cache.
        Each branch is an independent row in the batch, guaranteeing branch isolation.
        """
        branches = branch_data["branches"]
        if not branches:
            return []

        B = len(branches)
        L_max = branch_data["max_len"]

        ids = torch.zeros((B, L_max), dtype=torch.long, device=self.device)
        pos = torch.zeros((B, L_max), dtype=torch.long, device=self.device)

        lm_dtype = next(self.lm.parameters()).dtype
        K_total = state_len + L_max

        # Standard 2D attention mask across [state_prefix + question_branch]
        # Hugging Face causal models natively expand 2D masks with past_key_values in SDPA
        mask_2d = torch.zeros((B, K_total), dtype=torch.long, device=self.device)
        mask_2d[:, :state_len] = 1
        for b, br in enumerate(branches):
            l = len(br["ids"])
            ids[b, :l] = torch.tensor(br["ids"], device=self.device)
            pos[b, :l] = torch.tensor(br["pos"], device=self.device)
            mask_2d[b, state_len : state_len + l] = 1

        expanded_pkv = clone_or_expand_past_key_values(past_key_values, batch_size=B)
        out = self.lm(
            input_ids=ids,
            position_ids=pos,
            attention_mask=mask_2d,
            past_key_values=expanded_pkv,
        )
        hs = out.last_hidden_state.float()

        logits_list = []
        for b, br in enumerate(branches):
            d_idx = br["decide_idx"]
            o_indices = torch.tensor(br["opt_idx"], device=self.device)
            h_decide = hs[b, d_idx]
            h_opts = hs[b, o_indices]
            z = self.head(h_decide, h_opts)
            logits_list.append(z)

        return logits_list

    @torch.no_grad()
    def probs_cached(self, tok, rec: dict, past_key_values, state_len: int):
        """
        Computes answer probabilities for rec using a precomputed prefix KV-cache.
        """
        branch_data = encode_question_branches(tok, rec["questions"], state_len)
        logits = self.forward_with_cache(branch_data, past_key_values, state_len)
        return [F.softmax(z, dim=-1).cpu() for z in logits]

