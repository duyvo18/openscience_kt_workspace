"""Module 2.2-2.4: fixed pattern operators, distributional pooling, readout.

Each operator i produces weights w over the interaction prefix j <= t, then
pools the per-step Gaussians by moment matching (a Gaussian mixture's first
two moments), preserving both location and dispersion:

    mu_P  = sum_j w_j mu_j
    var_P = sum_j w_j (var_j + mu_j^2) - mu_P^2

Operators (pattern structure is identical for every learner, as required):
    temporal  - exponential recency decay over the last k steps (learnable rate)
    same-KC   - attention restricted to steps sharing a KC with step t
    prereq    - attention restricted to steps on prerequisite KCs of step t
    neighbor  - attention restricted to steps on neighboring KCs of step t

A step whose operator has empty support falls back to a learned per-operator
null distribution. Readout maps each pooled pattern to a 64-d block v_i;
z' = [v_1;v_2;v_3;v_4] with disabled operators contributing zeros.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from ..config import Config
from .distribution import LOGVAR_MIN, LOGVAR_MAX
from .embeddings import masked_softmax

PATTERN_NAMES = ["temporal", "samekc", "prereq", "neighbor"]


def build_pattern_masks(
    kc: torch.Tensor,        # (B,L,K_max) int64, -1 padded
    pad_mask: torch.Tensor,  # (B,L) True = real step
    P_rel: torch.Tensor,     # (C,C) bool, prereq[i,j]: i is prerequisite of j
    N_rel: torch.Tensor,     # (C,C) bool, symmetric neighbors
) -> dict[str, torch.Tensor]:
    """Precompute (B,L,L) boolean masks: mask[b,t,j] = step j is in the
    operator's support for target step t. Causality (j <= t) is applied
    later when slicing the prefix, so only KC relations are encoded here."""
    B, L, K = kc.shape
    valid = kc >= 0                                    # (B,L,K)
    kcc = kc.clamp(min=0)
    # dims: (B, L_t, L_j, K_t, K_j)
    pair_valid = valid.view(B, L, 1, K, 1) & valid.view(B, 1, L, 1, K)
    # a: KCs of target step t; b: KCs of source step j
    a = kcc.view(B, L, 1, K, 1).expand(B, L, L, K, K)
    b = kcc.view(B, 1, L, 1, K).expand(B, L, L, K, K)

    samekc = ((a == b) & pair_valid).any(-1).any(-1)   # (B,L,L)
    prereq = (P_rel[b, a] & pair_valid).any(-1).any(-1)   # source j prereq of target t
    neighbor = (N_rel[a, b] & pair_valid).any(-1).any(-1)

    step_pair = pad_mask.unsqueeze(1) & pad_mask.unsqueeze(2)  # (B,L_t,L_j)
    return {
        "samekc": samekc & step_pair,
        "prereq": prereq & step_pair,
        "neighbor": neighbor & step_pair,
    }


class PatternOperators(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.enabled = {
            "temporal": cfg.use_pattern_temporal,
            "samekc": cfg.use_pattern_samekc,
            "prereq": cfg.use_pattern_prereq,
            "neighbor": cfg.use_pattern_neighbor,
        }
        self.decay = nn.Parameter(torch.tensor(0.0))  # softplus -> temporal rate
        self.null_mu = nn.ParameterDict(
            {n: nn.Parameter(torch.zeros(cfg.d_z)) for n in PATTERN_NAMES}
        )
        self.null_logvar = nn.ParameterDict(
            {n: nn.Parameter(torch.zeros(cfg.d_z)) for n in PATTERN_NAMES}
        )
        self.readout = nn.ModuleDict(
            {n: nn.Linear(2 * cfg.d_z, cfg.d_z) for n in PATTERN_NAMES}
        )
        self.scale = 1.0 / math.sqrt(cfg.d_z)

    def _pool(
        self,
        w: torch.Tensor,          # (B,T) weights over prefix, rows may be all-zero
        has: torch.Tensor,        # (B,1) row has support
        mu: torch.Tensor,         # (B,T,d_z) prefix means
        var: torch.Tensor,        # (B,T,d_z) prefix variances
        name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mu_p = torch.einsum("bt,btd->bd", w, mu)
        ex2 = torch.einsum("bt,btd->bd", w, var + mu.pow(2))
        var_p = (ex2 - mu_p.pow(2)).clamp(min=1e-3)  # floor caps d log/d var
        mu_p = torch.where(has, mu_p, self.null_mu[name].unsqueeze(0))
        var_p = torch.where(has, var_p, self.null_logvar[name].exp().unsqueeze(0))
        return mu_p, var_p

    def forward(
        self,
        t: int,
        mu_prefix: torch.Tensor,   # (B,t+1,d_z)
        var_prefix: torch.Tensor,  # (B,t+1,d_z)
        masks: dict[str, torch.Tensor],  # (B,L,L) relation masks
        pad_mask: torch.Tensor,    # (B,L)
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Pool the prefix (steps 0..t) for each enabled operator.

        Returns {name: {mu, var, v, w}} with v the readout block (B,d_z)
        (zeros when the operator is disabled) and w the pooling weights.
        """
        B, T, _ = mu_prefix.shape
        dev = mu_prefix.device
        out: dict[str, dict[str, torch.Tensor]] = {}

        mu_t = mu_prefix[:, -1]  # (B,d_z) current step's location = query
        logits = torch.einsum("bd,btd->bt", mu_t, mu_prefix) * self.scale

        for name in PATTERN_NAMES:
            if not self.enabled[name]:
                zeros = mu_t.new_zeros(B, self.cfg.d_z)
                out[name] = {"mu": zeros, "var": zeros, "v": zeros,
                             "w": mu_t.new_zeros(B, T)}
                continue
            if name == "temporal":
                ages = torch.arange(T - 1, -1, -1, device=dev, dtype=mu_t.dtype)
                mask = pad_mask[:, :T] & (ages < self.cfg.temporal_k).unsqueeze(0)
                lg = (-torch.nn.functional.softplus(self.decay) * ages).unsqueeze(0)
                lg = lg.expand(B, T)
            else:
                mask = masks[name][:, t, :T]
                lg = logits
            w, has = masked_softmax(lg, mask)
            mu_p, var_p = self._pool(w, has, mu_prefix, var_prefix, name)
            v = self.readout[name](torch.cat([mu_p, var_p.log().clamp(LOGVAR_MIN, LOGVAR_MAX)], dim=-1))
            out[name] = {"mu": mu_p, "var": var_p, "v": v, "w": w}
        return out
