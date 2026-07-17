"""Embedding tables shared across modules.

Question id 0 is padding (padding_idx). KC slots use -1 for padding; callers
clamp to 0 and mask, so no padding_idx is set on the KC table.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config


def masked_softmax(
    logits: torch.Tensor, mask: torch.Tensor, dim: int = -1
) -> tuple[torch.Tensor, torch.Tensor]:
    """Softmax over `mask`==True entries; fully-masked rows return zeros.

    Returns (weights, has_valid) where has_valid broadcasts over `dim`.
    """
    filled = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    w = torch.softmax(filled, dim=dim)
    has_valid = mask.any(dim=dim, keepdim=True)
    return w * has_valid, has_valid


class InteractionEmbeddings(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        d = cfg.d_emb
        self.q = nn.Embedding(cfg.n_questions, d, padding_idx=0)
        self.kc = nn.Embedding(cfg.n_kcs, d)
        self.r = nn.Embedding(2, d)
        self.q_diff = nn.Embedding(cfg.n_difficulty_bins, d)
        self.kc_diff = nn.Embedding(cfg.n_difficulty_bins, d)

    def kc_mean(self, kc: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean KC embedding over valid slots. kc: (..., K_max), -1 padded."""
        valid = kc >= 0
        e = self.kc(kc.clamp(min=0)) * valid.unsqueeze(-1)
        denom = valid.sum(-1, keepdim=True).clamp(min=1)
        return e.sum(-2) / denom, valid

    def kc_diff_mean(self, kc: torch.Tensor, kc_diff_bin: torch.Tensor) -> torch.Tensor:
        """Mean concept-difficulty embedding over valid slots."""
        valid = kc >= 0
        bins = kc_diff_bin[kc.clamp(min=0)]
        e = self.kc_diff(bins) * valid.unsqueeze(-1)
        denom = valid.sum(-1, keepdim=True).clamp(min=1)
        return e.sum(-2) / denom
