"""
Bidirectional encoder decision model architecture for rev.
Powers both ModernBERT-large (421M, English) and mmBERT-base (322M, 100+ languages).
"""

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .common import QTYPES


class EncoderDecisionModel(nn.Module):
    """
    Bidirectional transformer encoder backbone + typed decision head + action/deferral head.
    
    Sequence format:
      [CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
    
    The [MASK] markers pool option representations directly in a single forward pass.
    """

    def __init__(
        self,
        encoder: nn.Module,
        head_layers: int = 2,
        n_act: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(
            d, nhead, 4 * d, dropout, batch_first=True, norm_first=True
        )
        self.head = (
            nn.TransformerEncoder(layer, head_layers, enable_nested_tensor=False)
            if head_layers > 0
            else None
        )
        self.type_emb = nn.Embedding(3, d)
        self.scorer = nn.Sequential(
            nn.LayerNorm(d),
            nn.Linear(d, d),
            nn.GELU(),
            nn.Linear(d, 1),
        )
        self.act_head = nn.Sequential(
            nn.Linear(d + 4, 256),
            nn.GELU(),
            nn.Linear(256, n_act),
        )
        self.register_buffer("temperature", torch.ones(3))
        self.head_checkpointing = False

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        marker_pos: torch.Tensor,
        marker_mask: torch.Tensor,
        qtype: torch.Tensor,
        detach_encoder: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass scoring all options in parallel.
        
        Args:
            input_ids: [B, L] token IDs
            attention_mask: [B, L] attention mask
            marker_pos: [B, K] position indices of option markers
            marker_mask: [B, K] boolean mask for valid options
            qtype: [B] question type indices (0: choice, 1: score, 2: noul)
            detach_encoder: if True, detaches backbone encoder gradients
            
        Returns:
            logits: [B, K] option decision logits
            act_logits: [B, n_act] action/deferral logits
        """
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        if detach_encoder:
            h = h.detach()

        # Add question-type embedding
        h = h + self.type_emb(qtype)[:, None, :]

        # Optional decision transformer head
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)

        # Gather marker embeddings at option positions
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        # Compute uncertainty and margin features for action/deferral head
        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        if p.size(-1) >= 2:
            top2 = p.topk(2, -1).values
        else:
            top1 = p.topk(1, -1).values
            top2 = torch.cat([top1, torch.zeros_like(top1)], dim=-1)

        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        pooled = h[:, 0].float()
        act_logits = self.act_head(torch.cat([pooled, feats], -1))

        return logits, act_logits


def build_encoder_model(cfg: Dict, encoder_dir: Optional[str] = None) -> EncoderDecisionModel:
    """Build an EncoderDecisionModel from config and optional local encoder directory."""
    from transformers import AutoConfig, AutoModel

    if encoder_dir and os.path.exists(encoder_dir):
        ecfg = AutoConfig.from_pretrained(encoder_dir)
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
    else:
        enc = AutoModel.from_pretrained(cfg.get("encoder", "answerdotai/ModernBERT-large"), attn_implementation="sdpa")
    return EncoderDecisionModel(
        enc,
        head_layers=cfg.get("head_layers", 2),
        n_act=len(cfg.get("act_costs", {})) + 1,
    )
