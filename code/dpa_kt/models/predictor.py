"""Module 4: prediction aggregation with KC contributions and guess/slip.

beta = softmax over related KCs of (K_c . W_p e_q) is the KC-to-prediction
contribution W of the attribution trace. The mastery read u is combined with
the target-question embedding and difficulty through an MLP into p_master,
then corrected by learnable bounded guess/slip scalars:

    y_hat = (1 - s) * p_master + g * (1 - p_master)
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import Config
from .embeddings import masked_softmax


def _inv_sigmoid(x: float) -> float:
    return math.log(x / (1.0 - x))


class PredictionHead(nn.Module):
    G_MAX, S_MAX = 0.35, 0.30

    def __init__(self, cfg: Config):
        super().__init__()
        self.W_pred = nn.Linear(cfg.d_emb, cfg.d_key)
        self.read_norm = nn.LayerNorm(cfg.d_v)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_v + 2 * cfg.d_emb, cfg.d_emb),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_emb, 1),
        )
        # init so g ~= 0.2, s ~= 0.1
        self.theta_g = nn.Parameter(torch.tensor(_inv_sigmoid(0.2 / self.G_MAX)))
        self.theta_s = nn.Parameter(torch.tensor(_inv_sigmoid(0.1 / self.S_MAX)))

    @property
    def guess(self) -> torch.Tensor:
        return self.G_MAX * torch.sigmoid(self.theta_g)

    @property
    def slip(self) -> torch.Tensor:
        return self.S_MAX * torch.sigmoid(self.theta_s)

    def forward(
        self,
        M: torch.Tensor,        # (B,C,d_v) mastery BEFORE seeing this step
        rel: torch.Tensor,      # (B,K_rel) related KCs of the target question
        kc_key: nn.Embedding,   # shared KC key table from MasteryState
        e_q: torch.Tensor,      # (B,d_emb)
        e_dq: torch.Tensor,     # (B,d_emb) question difficulty embedding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (y_hat (B,), beta (B,K_rel))."""
        valid = rel >= 0
        relc = rel.clamp(min=0)
        keys = kc_key(relc)
        logits = torch.einsum("bkd,bd->bk", keys, self.W_pred(e_q))
        beta, _ = masked_softmax(logits, valid)
        rows = M.gather(1, relc.unsqueeze(-1).expand(-1, -1, M.size(-1)))
        u = self.read_norm(torch.einsum("bk,bkv->bv", beta, rows))  # normed read
        p_master = torch.sigmoid(self.mlp(torch.cat([u, e_q, e_dq], -1))).squeeze(-1)
        y_hat = (1.0 - self.slip) * p_master + self.guess * (1.0 - p_master)
        return y_hat, beta
