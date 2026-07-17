"""Module 1: dual-branch interaction representation learning.

Branch A (interaction context: question + response + question difficulty) is a
1-layer causal Transformer computed in parallel before the time loop — it does
not depend on the mastery state. Branch B (knowledge context: localized
mastery read + concept embedding + concept difficulty) depends on M_t, so it
is a GRU cell stepped inside the time loop. Fusion produces z_t.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config


class BranchA(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.inp = nn.Linear(3 * cfg.d_emb, cfg.d_model)
        self.encoder = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=2 * cfg.d_model,
            dropout=cfg.dropout,
            batch_first=True,
            norm_first=True,
        )

    def forward(
        self,
        e_q: torch.Tensor,      # (B,L,d)
        e_r: torch.Tensor,      # (B,L,d)
        e_dq: torch.Tensor,     # (B,L,d)
        pad_mask: torch.Tensor, # (B,L) True = real step
    ) -> torch.Tensor:
        x = self.inp(torch.cat([e_q, e_r, e_dq], dim=-1))
        L = x.size(1)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=x.device)
        h = self.encoder(x, src_mask=causal, src_key_padding_mask=~pad_mask)
        return h.nan_to_num(0.0)  # fully-padded rows produce NaNs; zero them


class BranchBCell(nn.Module):
    """One step of the knowledge-context branch."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.read_proj = nn.Linear(cfg.d_v, cfg.d_key)
        self.inp = nn.Linear(cfg.d_key + 2 * cfg.d_emb, cfg.d_model)
        self.cell = nn.GRUCell(cfg.d_model, cfg.d_model)
        # layer-normalized recurrence: bounds the carried hidden state so the
        # tBPTT Jacobian stays ~unit-gain (raw GRU here has gain ~4/step,
        # which explodes gradients over 25 steps)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(
        self,
        m_read: torch.Tensor,   # (B,d_v)  localized mastery read
        e_c: torch.Tensor,      # (B,d)    mean KC embedding
        e_dc: torch.Tensor,     # (B,d)    mean concept difficulty embedding
        h_prev: torch.Tensor,   # (B,d_model)
        step_valid: torch.Tensor,  # (B,) True = real step (carry h through pads)
    ) -> torch.Tensor:
        x = self.inp(torch.cat([self.read_proj(m_read), e_c, e_dc], dim=-1))
        h = self.norm(self.cell(x, h_prev))
        return torch.where(step_valid.unsqueeze(-1), h, h_prev)


class Fusion(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.proj = nn.Linear(2 * cfg.d_model, cfg.d_model)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        return self.norm(self.proj(torch.cat([h_a, h_b], dim=-1)))
