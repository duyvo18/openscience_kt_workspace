"""Literature-reported AUC/ACC for KT models, for the comparison table.

Baseline model list matches the "Baselines" paragraph of
working/artifacts/main_20260723_post_review.pdf (Distributional Pedagogical
Alignment, Section V - Planned Evaluation): the foundational BKT; classic/
attention-based DKT, DKVMN, AKT; graph/representation-enriched GKT, CMDKT,
UKT, KeenKT, S2KT; reasoning/pattern-based PSI-KT, NSKT, PLKT; and
learning-process LPKT, MemoryKT (14 models total).

Models from an earlier literature pass that are NOT part of that paper's
baseline set (SAKT, simpleKT, sparseKT, qDKT, ATKT, Deep-IRT, AT-DKT, QIKT,
HawkesKT, DASKT, RKT, GRKT, DenoiseKT, MCKT, DIMKT, ReKT, ASIKT, NTKT, MERIT,
KCQRL) are commented out below rather than deleted, so their sourced values
are preserved if a future comparison needs them again - each still carries
its original source citation.

Values are collected under DIFFERENT preprocessing and splits than ours
(sequence length, fold scheme, question vs concept granularity), so the
table is INDICATIVE context, not a head-to-head result. Our own numbers are
the only ones produced on our exact splits.

Dataset key mapping (ours -> literature name):
  assist09  -> ASSISTments2009        algebra05 -> Algebra2005 (AL2005)
  bridge06  -> Bridge2006 (BD2006)     eedi      -> NeurIPS34 (Eedi)
  xes3g5m   -> XES3G5M (question lvl)  assist12  -> ASSISTments2012
  junyi     -> Junyi Academy

No usable published number could be found for two target baselines:
  S2KT - WWW 2026, DOI paywalled, no preprint indexed as of 2026-07-23.
  NSKT - 2026 preprint, evaluated only on a private Estonian-school
         dataset; no overlap with any of our 7 benchmarks.
"""
from __future__ import annotations

LITERATURE_AUC: dict[str, dict[str, float]] = {
    "assist09": {
        "BKT": 0.67,       # Piech et al. 2015 (DKT paper) Table 1, AS2009 row;
                           # replications range 0.73 (Khajah et al. 2016) to
                           # 0.76 (pyBKT) / 0.83 with an explicit forgetting term
        "DKT": 0.754, "DKVMN": 0.747, "GKT": 0.742, "AKT": 0.785,
        "CMDKT": 0.8051,   # CMDKT's own paper (Array 2025) Table 4
        "KeenKT": 0.8606,  # KeenKT's own paper (AAAI 2026) Table 2
        "PLKT": 0.8116,    # PLKT's own paper (2026) Table 2
        "UKT": 0.8563,     # UKT paper Table 1, AS2009 row
        # --- not in the paper's baseline list; kept for later use ---
        # "SAKT": 0.725, "ATKT": 0.747,
        # "Deep-IRT": 0.7465,  # QIKT / DenoiseKT papers, pyKT-protocol rerun
        # "simpleKT": 0.774,   # simpleKT paper (arXiv:2302.06881) Table 2
        # "qDKT": 0.762,       # qDKT's own paper, Table 3 (best variant)
    },
    "algebra05": {
        "DKT": 0.815, "DKVMN": 0.805, "AKT": 0.831,
        "KeenKT": 0.9346,    # KeenKT's own paper Table 2, AL2005 row
        "PLKT": 0.9645,      # PLKT's own paper Table 2
        "MemoryKT": 0.8437,  # MemoryKT's own paper (arXiv:2508.08122) Table 2
        "UKT": 0.9320,       # UKT paper Table 1, AL2005 row
        # --- not in the paper's baseline list; kept for later use ---
        # "SAKT": 0.7880,      # pyKT Table 2/8
        # "simpleKT": 0.825,   # simpleKT paper Table 2
        # "AT-DKT": 0.8246,    # AT-DKT's own paper (arXiv:2302.07942) Table 2
    },
    "bridge06": {
        "DKT": 0.801, "AKT": 0.821,
        "DKVMN": 0.7983,   # pyKT Table 2/8
        "KeenKT": 0.8265,  # KeenKT's own paper Table 2, BD2006 row
        "PLKT": 0.8534,    # PLKT's own paper Table 2
        "UKT": 0.8178,     # UKT paper Table 1, BD2006 row
        # --- not in the paper's baseline list; kept for later use ---
        # "SAKT": 0.7740,      # pyKT Table 2/8
        # "simpleKT": 0.816,   # simpleKT paper Table 2
    },
    "eedi": {
        "DKT": 0.769, "DKVMN": 0.767, "AKT": 0.803,
        "KeenKT": 0.8162,  # KeenKT's own paper Table 2, NIPS34 row
        "UKT": 0.8035,     # UKT paper Table 1, NIPS34 row
        # --- not in the paper's baseline list; kept for later use ---
        # "SAKT": 0.752,
        # "simpleKT": 0.804,   # simpleKT paper Table 2
        # "sparseKT": 0.7994,  # sparseKT's own paper Table 1, NIPS34 row
        # "QIKT": 0.8044,      # QIKT's own paper (arXiv:2302.06885) Table 2
        # "NTKT (LLaMA-1B)": 0.9335, "NTKT (LLaMA-3B)": 0.9572,
        # "NTKT (LLaMA-8B)": 0.9353, "MERIT (Gemini)": 0.7969,
        # "KCQRL": 0.7896,
        # "DIMKT": 0.8074,  # Table 3 DIMKT SIGIR'22, Eedi2020 row; ACC 0.7471
        # "ReKT": 0.7971,   # Table 2 ReKT ACM MM'24, Eedi row; ACC 0.7397
        # "ASIKT": 0.7754,  # Table 2 ASIKT SIGIR'25, Eedi row; ACC 0.7066*
    },
    "xes3g5m": {
        # No target baseline (BKT/CMDKT/KeenKT/S2KT/PSI-KT/NSKT/PLKT/
        # LPKT/MemoryKT) reports XES3G5M results - only DKT/DKVMN/AKT
        # survive from the earlier pass.
        "AKT": 0.8207,    # XES3G5M paper Table 2 / DenoiseKT paper Table 2
        "DKT": 0.7852,    # XES3G5M paper Table 2 / DenoiseKT paper Table 2
        "DKVMN": 0.7792,  # XES3G5M paper Table 2 / DenoiseKT paper Table 2
        # --- not in the paper's baseline list; kept for later use ---
        # "simpleKT": 0.8163,   # XES3G5M paper Table 2 / DenoiseKT paper Table 2
        # "DenoiseKT": 0.8282,  # DenoiseKT's own paper, Table 2 (best result)
        # "MCKT": 0.8451,       # MCKT's own paper, Table 5, XES3G5M row
    },
    "assist12": {
        "DKT": 0.712, "DKVMN": 0.701,  # RKT paper Table 4
        "PLKT": 0.7849,  # PLKT's own paper Table 2
        "LPKT": 0.7824,  # LPKT's own paper, corroborated by DASKT Table IV
        # PSI-KT reports only ACC (Within-learner 0.68), no AUC - see
        # LITERATURE_ACC below.
        # --- not in the paper's baseline list; kept for later use ---
        # "SAKT": 0.735,
        # "HawkesKT": 0.7676,  # HawkesKT's own paper / GitHub results table
        # "DASKT": 0.7925,     # DASKT's own paper (arXiv:2502.10396) Table IV
    },
    "junyi": {
        "DKT": 0.8003, "DKVMN": 0.8004, "AKT": 0.8161,  # GRKT paper Table 2
        "PLKT": 0.7226,  # PLKT's own paper Table 2
        # PSI-KT reports only ACC (Within-learner 0.83), no AUC - see
        # LITERATURE_ACC below.
        # --- not in the paper's baseline list; kept for later use ---
        # "SAKT": 0.7995,   # GRKT paper Table 2
        # "RKT": 0.860,     # RKT's own paper, Table 4, Junyi column
        # "GRKT": 0.8207,   # GRKT's own paper, Table 2, Junyi column
    },
}

LITERATURE_ACC: dict[str, dict[str, float]] = {
    ds: {m: float("nan") for m in models}
    for ds, models in LITERATURE_AUC.items()
}
LITERATURE_ACC["assist09"].update({
    "CMDKT": 0.7665,     # CMDKT's own paper Table 4
    "KeenKT": 0.7934,    # KeenKT's own paper Table 3
    "PLKT": 0.7657,      # PLKT's own paper Table 2
    "MemoryKT": 0.7974,  # MemoryKT's own paper Table 2
})
LITERATURE_ACC["algebra05"].update({
    "KeenKT": 0.8772,    # KeenKT's own paper Table 3
    "PLKT": 0.9072,      # PLKT's own paper Table 2
    "MemoryKT": 0.8117,  # MemoryKT's own paper Table 2
})
LITERATURE_ACC["bridge06"].update({
    "KeenKT": 0.8517,  # KeenKT's own paper Table 3
    "PLKT": 0.8653,    # PLKT's own paper Table 2
})
LITERATURE_ACC["eedi"].update({
    "KeenKT": 0.7378,  # KeenKT's own paper Table 3
    # --- not in the paper's baseline list; kept for later use ---
    # "DIMKT": 0.7471,   # Table 3 DIMKT SIGIR'22, Eedi2020 row
    # "ReKT": 0.7397,    # Table 2 ReKT ACM MM'24, Eedi row
    # "ASIKT": 0.7066,   # * p<0.05, Table 2 ASIKT SIGIR'25, Eedi row
})
LITERATURE_ACC["xes3g5m"].update({
    # --- not in the paper's baseline list; kept for later use ---
    # "MCKT": 0.8385,    # Table 5 MCKT IEEE Access 2026, XES3G5M row
})
LITERATURE_ACC["assist12"].update({
    "PLKT": 0.7613,   # PLKT's own paper Table 2
    "PSI-KT": 0.68,   # PSI-KT's own paper Table 2, Within-learner setting;
                      # this paper reports only accuracy, never AUC
})
LITERATURE_ACC["junyi"].update({
    "PLKT": 0.8333,  # PLKT's own paper Table 2
    "PSI-KT": 0.83,  # PSI-KT's own paper Table 2, Within-learner setting;
                     # this paper reports only accuracy, never AUC
})

CAVEAT = (
    "Literature AUC comes from each baseline's own paper (or the pyKT "
    "benchmark, for datasets it covers) under different preprocessing/"
    "splits; treat as indicative context, not a head-to-head comparison. "
    "Only 'DPA-KT (ours)' is on our exact splits."
)


def comparison_frame(our_results: dict[str, dict]):
    """Build a tidy DataFrame merging our test metrics with literature AUC.

    our_results: {dataset: {"auc": float, "acc": float, ...}}
    Returns a pandas DataFrame: rows = datasets, columns = models (AUC),
    with 'DPA-KT (ours)' first and a 'best_literature' summary column.
    """
    import pandas as pd

    rows = []
    datasets = sorted(set(LITERATURE_AUC) | set(our_results))
    for ds in datasets:
        row = {"dataset": ds}
        if ds in our_results and our_results[ds] is not None:
            row["DPA-KT (ours) AUC"] = round(our_results[ds].get("auc", float("nan")), 4)
            row["DPA-KT (ours) ACC"] = round(our_results[ds].get("acc", float("nan")), 4)
        lit = LITERATURE_AUC.get(ds, {})
        for model, auc in lit.items():
            row[model] = auc
        if lit:
            best_model = max(lit, key=lit.get)
            row["best_literature"] = f"{best_model} ({lit[best_model]:.3f})"
        rows.append(row)
    return pd.DataFrame(rows).set_index("dataset")
