"""Module 2.1: distribution projection z_t -> N(mu, diag sigma^2).

The "statistical-descriptor decoder" of the paper, realized as two linear
heads. With use_distributional=False (ablation) the variance head is bypassed
and logvar is a constant zero: patterns degrade to point-embedding pooling.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config

LOGVAR_MIN, LOGVAR_MAX = -6.0, 2.0


class GaussianProjection(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.mu = nn.Linear(cfg.d_model, cfg.d_z)
        self.logvar = nn.Linear(cfg.d_model, cfg.d_z)
        self.distributional = cfg.use_distributional

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu = self.mu(z)
        if self.distributional:
            logvar = self.logvar(z).clamp(LOGVAR_MIN, LOGVAR_MAX)
        else:
            logvar = torch.zeros_like(mu)
        return mu, logvar


def kl_to_standard_normal(
    mu: torch.Tensor, logvar: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Mean KL(N(mu,sigma^2) || N(0,1)) over mask==True positions."""
    kl = 0.5 * (logvar.exp() + mu.pow(2) - 1.0 - logvar).sum(-1)
    return (kl * mask).sum() / mask.sum().clamp(min=1)
