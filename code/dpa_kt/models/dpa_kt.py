"""DPA-KT assembly: the four modules wired through the sequence time loop.

Causality per step t:
  1. Module 4 predicts y_hat_t from M_t (built from interactions < t) and the
     identity/difficulty of q_t — the response r_t is NOT visible here.
  2. Modules 1-3 then consume interaction t (including r_t) to produce the
     evidence z'_t and update M_t -> M_{t+1}.

Truncated BPTT: every cfg.tbptt steps the mastery state, branch-B hidden
state, and the distributional prefix are detached.

forward(..., return_trace=True) additionally returns the intrinsic
attribution trace: pattern pooling weights w^(i), pattern-to-KC gates A_i,
KC-to-prediction contributions beta, related-KC ids, scalar mastery
evolution, and the guess/slip scalars.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import Config
from .distribution import GaussianProjection, kl_to_standard_normal
from .embeddings import InteractionEmbeddings
from .interaction_encoder import BranchA, BranchBCell, Fusion
from .mastery import MasteryState
from .patterns import PATTERN_NAMES, PatternOperators, build_pattern_masks
from .predictor import PredictionHead


class DPAKT(nn.Module):
    def __init__(
        self,
        cfg: Config,
        q_rel: np.ndarray,       # (V_q,K_rel) graph-expanded related KCs, -1 pad
        P_rel: np.ndarray,       # (C,C) uint8
        N_rel: np.ndarray,       # (C,C) uint8
        q_diff_bin: np.ndarray,  # (V_q,) uint8
        kc_diff_bin: np.ndarray, # (C,) uint8
    ):
        super().__init__()
        self.cfg = cfg
        self.register_buffer("q_rel", torch.from_numpy(np.asarray(q_rel, dtype=np.int64)))
        self.register_buffer("P_rel", torch.from_numpy(np.asarray(P_rel)).bool())
        self.register_buffer("N_rel", torch.from_numpy(np.asarray(N_rel)).bool())
        self.register_buffer(
            "q_diff_bin", torch.from_numpy(np.asarray(q_diff_bin, dtype=np.int64))
        )
        self.register_buffer(
            "kc_diff_bin", torch.from_numpy(np.asarray(kc_diff_bin, dtype=np.int64))
        )

        self.emb = InteractionEmbeddings(cfg)
        self.branch_a = BranchA(cfg)
        self.branch_b = BranchBCell(cfg)
        self.fusion = Fusion(cfg)
        self.gauss = GaussianProjection(cfg)
        self.patterns = PatternOperators(cfg)
        self.mastery = MasteryState(cfg)
        self.predictor = PredictionHead(cfg)

    # ------------------------------------------------------------------
    def _related(self, q_t: torch.Tensor, kc_t: torch.Tensor) -> torch.Tensor:
        """Related-KC ids for the step: own KCs first, then graph expansion,
        truncated to K_rel. (B,) + (B,K_max) -> (B,K_rel)."""
        graph = self.q_rel[q_t][:, : self.cfg.k_rel - kc_t.size(1)]
        return torch.cat([kc_t, graph], dim=1)

    # ------------------------------------------------------------------
    def forward(
        self, batch: dict[str, torch.Tensor], return_trace: bool = False
    ) -> dict:
        cfg = self.cfg
        q, r, kc, selectmask = batch["q"], batch["r"], batch["kc"], batch["selectmask"]
        B, L = q.shape
        pad_mask = q > 0

        # --- Module 1, branch A (parallel) + shared step features ---
        e_q = self.emb.q(q)
        e_r = self.emb.r(r.clamp(0, 1))
        e_dq = self.emb.q_diff(self.q_diff_bin[q])
        h_a = self.branch_a(e_q, e_r, e_dq, pad_mask)
        e_c_mean, _ = self.emb.kc_mean(kc)
        e_dc_mean = self.emb.kc_diff_mean(kc, self.kc_diff_bin)

        # --- pattern relation masks, precomputed for the whole batch ---
        masks = build_pattern_masks(kc, pad_mask, self.P_rel, self.N_rel)

        # --- time loop ---
        M = self.mastery.init_state(B)
        h_b = h_a.new_zeros(B, cfg.d_model)
        mu_list: list[torch.Tensor] = []
        var_list: list[torch.Tensor] = []
        y_all: list[torch.Tensor] = []
        mono_terms: list[torch.Tensor] = []
        gs_terms: list[torch.Tensor] = []
        kl_mu: list[torch.Tensor] = []
        kl_logvar: list[torch.Tensor] = []

        trace: dict[str, list] = (
            {k: [] for k in ("pattern_w", "gates", "beta", "rel", "mastery", "alpha")}
            if return_trace
            else {}
        )

        kc_valid = kc >= 0
        kcc = kc.clamp(min=0)

        for t in range(L):
            step_valid = pad_mask[:, t]
            kc_t = kc[:, t]                       # (B,K_max)
            rel_t = self._related(q[:, t], kc_t)  # (B,K_rel)

            # Module 4: predict BEFORE seeing r_t
            y_t, beta_t = self.predictor(
                M, rel_t, self.mastery.kc_key, e_q[:, t], e_dq[:, t]
            )
            y_all.append(y_t)

            # Module 1 branch B: localized mastery read + GRU step
            m_read, alpha_t = self.mastery.read(M, rel_t, e_q[:, t])
            h_b = self.branch_b(
                m_read, e_c_mean[:, t], e_dc_mean[:, t], h_b, step_valid
            )
            h_b_use = h_b if cfg.dual_branch else torch.zeros_like(h_b)
            z_t = self.fusion(h_a[:, t], h_b_use)

            # Module 2: distribution projection + pattern pooling
            mu_t, logvar_t = self.gauss(z_t)
            mu_list.append(mu_t)
            var_list.append(logvar_t.exp())
            kl_mu.append(mu_t)
            kl_logvar.append(logvar_t)
            mu_prefix = torch.stack(mu_list, dim=1)
            var_prefix = torch.stack(var_list, dim=1)
            pats = self.patterns(t, mu_prefix, var_prefix, masks, pad_mask)

            # Module 3: gated erase-add mastery update
            M_new, gates, _, _ = self.mastery.update(M, rel_t, pats, step_valid)

            # --- alignment losses (Module 2.3) ---
            own_valid = kc_valid[:, t] & step_valid.unsqueeze(-1)  # (B,K_max)
            own_rows_pre = M.gather(
                1, kcc[:, t].unsqueeze(-1).expand(-1, -1, cfg.d_v)
            )
            own_rows_post = M_new.gather(
                1, kcc[:, t].unsqueeze(-1).expand(-1, -1, cfg.d_v)
            )
            m_pre = self.mastery.scalar_mastery(own_rows_pre)    # (B,K_max)
            m_post = self.mastery.scalar_mastery(own_rows_post)
            if cfg.use_align_mono:
                correct_mask = own_valid & (r[:, t] == 1).unsqueeze(-1)
                drop = F.relu(m_pre - m_post - cfg.mono_margin)
                mono_terms.append(
                    (drop * correct_mask).sum() / correct_mask.sum().clamp(min=1)
                )
            if cfg.use_align_gs:
                surprise = (r[:, t].float() - y_t).pow(2).detach()  # (B,)
                delta = (own_rows_post - own_rows_pre).pow(2).mean(-1)  # (B,K_max)
                delta = (delta * own_valid).sum(-1) / own_valid.sum(-1).clamp(min=1)
                gs_terms.append(
                    (surprise * delta * step_valid).sum()
                    / step_valid.sum().clamp(min=1)
                )

            if return_trace:
                # cast to fp32 on cpu so numpy conversion works under bf16 AMP
                trace["pattern_w"].append(
                    torch.stack([pats[n]["w"] for n in PATTERN_NAMES], 0)
                    .float().detach().cpu()
                )  # (4,B,t+1)
                trace["gates"].append(
                    torch.stack([gates[n] for n in PATTERN_NAMES], 0).float().detach().cpu()
                )
                trace["beta"].append(beta_t.float().detach().cpu())
                trace["rel"].append(rel_t.detach().cpu())
                trace["alpha"].append(alpha_t.float().detach().cpu())
                trace["mastery"].append(
                    self.mastery.scalar_mastery(M_new).float().detach().cpu()
                )

            M = M_new
            if cfg.tbptt > 0 and (t + 1) % cfg.tbptt == 0:
                M = M.detach()
                h_b = h_b.detach()
                mu_list = [m.detach() for m in mu_list]
                var_list = [v.detach() for v in var_list]

        y = torch.stack(y_all, dim=1).clamp(1e-6, 1 - 1e-6)  # (B,L)

        # --- losses ---
        # y is a probability mixture (guess/slip head), not sigmoid(logits), so
        # BCE is computed manually in fp32 (F.binary_cross_entropy is unsafe
        # under autocast).
        sm = selectmask.bool() & pad_mask
        if sm.any():
            yp = y[sm].float()
            tgt = r[sm].float()
            bce = -(tgt * yp.log() + (1 - tgt) * (1 - yp).log()).mean()
        else:
            bce = y.sum() * 0
        loss = bce
        aux = {"bce": bce.detach()}
        if cfg.use_align_mono and mono_terms:
            l_mono = torch.stack(mono_terms).mean()
            loss = loss + cfg.w_mono * l_mono
            aux["mono"] = l_mono.detach()
        if cfg.use_align_gs and gs_terms:
            l_gs = torch.stack(gs_terms).mean()
            loss = loss + cfg.w_gs * l_gs
            aux["gs"] = l_gs.detach()
        if cfg.use_distributional:
            l_kl = kl_to_standard_normal(
                torch.stack(kl_mu, 1), torch.stack(kl_logvar, 1), pad_mask
            )
            loss = loss + cfg.w_kl * l_kl
            aux["kl"] = l_kl.detach()

        out = {"y": y, "loss": loss, "aux": aux}
        if return_trace:
            out["trace"] = {
                "pattern_w": trace["pattern_w"],           # list of (4,B,t+1)
                "gates": torch.stack(trace["gates"], 2),   # (4,B,L,K_rel)
                "beta": torch.stack(trace["beta"], 1),     # (B,L,K_rel)
                "rel": torch.stack(trace["rel"], 1),       # (B,L,K_rel)
                "alpha": torch.stack(trace["alpha"], 1),   # (B,L,K_rel)
                "mastery": torch.stack(trace["mastery"], 1),  # (B,L,C)
                "guess": float(self.predictor.guess),
                "slip": float(self.predictor.slip),
                "pattern_names": PATTERN_NAMES,
            }
        return out


def build_model(cfg: Config, dataset: str | None = None) -> DPAKT:
    """Construct DPAKT from cached preprocessing artifacts."""
    from ..data.canonical import load_maps
    from ..data.kc_graph import graph_path
    from ..data.sequences import load_sequences

    dataset = dataset or cfg.dataset
    maps = load_maps(dataset)
    cfg.n_questions = maps["n_questions"]
    cfg.n_kcs = maps["n_kcs"]
    g = np.load(graph_path(dataset))
    seq = load_sequences(dataset)
    return DPAKT(
        cfg,
        q_rel=g["q_rel"],
        P_rel=g["P_rel"],
        N_rel=g["N_rel"],
        q_diff_bin=np.asarray(seq["q_diff_bin"]),
        kc_diff_bin=np.asarray(seq["kc_diff_bin"]),
    )
