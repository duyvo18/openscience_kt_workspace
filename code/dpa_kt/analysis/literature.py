"""Literature-reported AUC for KT models, for the comparison table.

Values are AUC reported in the pyKT benchmark (Liu et al., "pyKT", NeurIPS
2022 Datasets & Benchmarks) and in the models' original papers / recent
surveys. They are collected under DIFFERENT preprocessing and splits than
ours (sequence length, fold scheme, question vs concept granularity), so the
table is INDICATIVE context, not a head-to-head result. Our own numbers are
the only ones produced on our exact splits.

Dataset key mapping (ours -> literature name):
  assist09  -> ASSISTments2009        algebra05 -> Algebra2005 (AL2005)
  bridge06  -> Bridge2006 (BD2006)     eedi      -> NeurIPS34 (Eedi)
  xes3g5m   -> XES3G5M (question lvl)  assist12  -> ASSISTments2012
  junyi     -> Junyi Academy
"""
from __future__ import annotations

# dataset -> {model: AUC}. "~" in comments flags approximate values.
LITERATURE_AUC: dict[str, dict[str, float]] = {
    "assist09": {
        "DKT": 0.754, "DKVMN": 0.747, "SAKT": 0.725, "GKT": 0.742,
        "Deep-IRT": 0.747, "qDKT": 0.768, "ATKT": 0.747, "AKT": 0.785,
        "simpleKT": 0.774, "sparseKT": 0.778, "UKT": 0.791,
    },
    "algebra05": {
        "DKT": 0.815, "DKVMN": 0.805, "SAKT": 0.790, "AKT": 0.831,
        "simpleKT": 0.825, "FoLiBiKT": 0.827, "UKT": 0.836, "AT-DKT": 0.828,
    },
    "bridge06": {
        "DKT": 0.801, "DKVMN": 0.803, "SAKT": 0.792, "AKT": 0.821,
        "simpleKT": 0.816, "FoLiBiKT": 0.818, "UKT": 0.824,
    },
    "eedi": {
        "DKT": 0.769, "DKVMN": 0.767, "SAKT": 0.752, "AKT": 0.803,
        "simpleKT": 0.804, "sparseKT": 0.803, "QIKT": 0.806, "UKT": 0.810,
    },
    "xes3g5m": {
        "DKT": 0.783, "DKVMN": 0.788, "AKT": 0.821, "simpleKT": 0.816,
        "sparseKT": 0.824, "DenoiseKT": 0.826, "MCKT": 0.828,
    },
    "assist12": {
        "DKT": 0.729, "DKVMN": 0.701, "SAKT": 0.735, "AKT": 0.775,
        "HawkesKT": 0.775, "LPKT": 0.804, "DASKT": 0.783,
    },
    "junyi": {
        "DKT": 0.756, "DKVMN": 0.750, "SAKT": 0.752, "AKT": 0.795,
        "RKT": 0.802, "GRKT": 0.812,
    },
}

CAVEAT = (
    "Literature AUC comes from the pyKT benchmark and original papers under "
    "different preprocessing/splits; treat as indicative context, not a "
    "head-to-head comparison. Only 'DPA-KT (ours)' is on our exact splits."
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
