"""Module 3: explicit mastery state with pattern-to-KC gating.

M_t: (B,C,d_v). Each step updates only the <= K_rel related KCs of the
current question via a DKVMN-style erase-add residual increment. The gating
A_i = softmax over related KCs of (K_c . W_g v_i) quantifies how much each
learning pattern contributes to each related KC — this is the interpretable
pattern->KC block of the attribution trace.

A shared scalar readout u maps a mastery row to a scalar mastery in (0,1),
used by the monotonicity loss and all mastery visualizations.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from ..config import Config
from .embeddings import masked_softmax
from .patterns import PATTERN_NAMES


class MasteryState(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.M0 = nn.Parameter(torch.zeros(cfg.n_kcs, cfg.d_v))
        self.kc_key = nn.Embedding(cfg.n_kcs, cfg.d_key)
        self.W_read = nn.Linear(cfg.d_emb, cfg.d_key)   # query for localized read
        self.W_gate = nn.ModuleDict(
            {n: nn.Linear(cfg.d_z, cfg.d_key) for n in PATTERN_NAMES}
        )
        self.W_erase = nn.ModuleDict(
            {n: nn.Linear(cfg.d_z, cfg.d_v) for n in PATTERN_NAMES}
        )
        self.W_add = nn.ModuleDict(
            {n: nn.Linear(cfg.d_z, cfg.d_v) for n in PATTERN_NAMES}
        )
        self.scalar_u = nn.Linear(cfg.d_v, 1)           # scalar mastery readout
        self.read_norm = nn.LayerNorm(cfg.d_v)          # stabilizes recurrence

    def init_state(self, batch_size: int) -> torch.Tensor:
        return self.M0.unsqueeze(0).expand(batch_size, -1, -1).contiguous()

    def scalar_mastery(self, rows: torch.Tensor) -> torch.Tensor:
        """(...,d_v) -> (...) scalar mastery in (0,1)."""
        return torch.sigmoid(self.scalar_u(rows)).squeeze(-1)

    def read(
        self,
        M: torch.Tensor,        # (B,C,d_v)
        rel: torch.Tensor,      # (B,K_rel) related KC ids, -1 padded
        e_q: torch.Tensor,      # (B,d_emb) target question embedding
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Localized mastery read over the related-KC set. Returns
        (m_read (B,d_v), alpha (B,K_rel))."""
        valid = rel >= 0
        relc = rel.clamp(min=0)
        keys = self.kc_key(relc)                               # (B,K_rel,d_key)
        logits = torch.einsum("bkd,bd->bk", keys, self.W_read(e_q))
        alpha, _ = masked_softmax(logits, valid)
        rows = M.gather(1, relc.unsqueeze(-1).expand(-1, -1, M.size(-1)))
        read = torch.einsum("bk,bkv->bv", alpha, rows)
        # LayerNorm the read to control magnitude feedback through the
        # M_t -> read -> GRU -> patterns -> M_{t+1} recurrence without the
        # gradient saturation a tanh bound would introduce
        return self.read_norm(read), alpha

    def update(
        self,
        M: torch.Tensor,             # (B,C,d_v)
        rel: torch.Tensor,           # (B,K_rel)
        patterns: dict[str, dict],   # output of PatternOperators.forward
        step_valid: torch.Tensor,    # (B,) real (non-pad) steps only
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Erase-add update on the related rows.

        Returns (M_new, gates {name: (B,K_rel)}, rows_before, rows_after).
        """
        valid = (rel >= 0) & step_valid.unsqueeze(-1)
        relc = rel.clamp(min=0)
        keys = self.kc_key(relc)                               # (B,K,d_key)
        d_v = M.size(-1)
        rows = M.gather(1, relc.unsqueeze(-1).expand(-1, -1, d_v))  # (B,K,d_v)

        keep = torch.ones_like(rows)
        add = torch.zeros_like(rows)
        gates: dict[str, torch.Tensor] = {}
        for name in PATTERN_NAMES:
            v = patterns[name]["v"]                            # (B,d_z)
            logits = torch.einsum("bkd,bd->bk", keys, self.W_gate[name](v))
            A, _ = masked_softmax(logits, valid)               # (B,K_rel)
            gates[name] = A
            if not v.any():                                    # disabled operator
                continue
            e = torch.sigmoid(self.W_erase[name](v)).unsqueeze(1)  # (B,1,d_v)
            a = torch.tanh(self.W_add[name](v)).unsqueeze(1)
            keep = keep * (1.0 - A.unsqueeze(-1) * e)
            add = add + A.unsqueeze(-1) * a

        new_rows = torch.where(
            valid.unsqueeze(-1), rows * keep + add, rows
        )
        M_new = M.scatter(
            1, relc.unsqueeze(-1).expand(-1, -1, d_v), new_rows
        )
        # Gradient clipping THROUGH TIME: the erase-add recurrence composed
        # over many steps has a Jacobian whose norm compounds multiplicatively
        # (confirmed empirically: freezing M drops backward grad norm from
        # ~1e11 to ~0.2 at tbptt=25, isolating the recurrence as the sole
        # source). A post-hoc clip_grad_norm_ on the full parameter set can't
        # rescue this: under bf16 the compounded gradient already overflows to
        # inf/nan before it ever reaches the optimizer. Clamping the gradient
        # elementwise here, at the exact point it re-enters the recurrence,
        # keeps every step's local contribution bounded so the compounded norm
        # stays finite (~150 at tbptt=25, ~2e3 at tbptt=200) instead of
        # exploding — this is what lets tbptt run the FULL sequence length.
        if self.cfg.mastery_grad_clip > 0 and M_new.requires_grad:
            clip = self.cfg.mastery_grad_clip
            M_new.register_hook(lambda g, c=clip: g.clamp(-c, c))
        return M_new, gates, rows, new_rows
