"""PSLC KDD Cup 2010 loaders: Algebra 2005-2006 and Bridge-to-Algebra 2006-2007.

Tab-separated. The KT item is the step (Problem Name + Step Name). KCs come
from KC(Default) (algebra) or KC(SubSkills) (bridge), multiple KCs joined by
'~~'. Target is 'Correct First Attempt'. Row order gives per-student ordering.
Steps without a KC are dropped.
"""
from __future__ import annotations

import polars as pl

from ...config import DATASETS_DIR
from . import register

BASE = DATASETS_DIR / "dataset PSLC KDD Cup 2010"


def _load_kdd(train_txt, kc_col: str) -> dict:
    cols = [
        "Row", "Anon Student Id", "Problem Name", "Step Name",
        "Correct First Attempt", kc_col,
    ]
    df = pl.read_csv(
        train_txt, separator="\t", columns=cols,
        schema_overrides={"Row": pl.Int64, "Correct First Attempt": pl.Int64},
        ignore_errors=True,
    )
    df = df.filter(
        pl.col(kc_col).is_not_null()
        & (pl.col(kc_col) != "")
        & pl.col("Correct First Attempt").is_in([0, 1])
    )
    out = df.select(
        pl.col("Anon Student Id").alias("user"),
        (pl.col("Problem Name") + "||" + pl.col("Step Name")).alias("question"),
        pl.col(kc_col).str.split("~~").alias("kcs"),
        pl.col("Correct First Attempt").alias("correct"),
        pl.col("Row").alias("ts"),
    )
    return {"df": out}


@register("algebra05")
def load_algebra05(**_) -> dict:
    return _load_kdd(
        BASE / "algebra_2005_2006" / "algebra_2005_2006_train.txt", "KC(Default)"
    )


@register("bridge06")
def load_bridge06(**_) -> dict:
    return _load_kdd(
        BASE / "bridge_to_algebra_2006_2007" / "bridge_to_algebra_2006_2007_train.txt",
        "KC(SubSkills)",
    )
