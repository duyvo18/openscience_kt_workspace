"""ASSISTments 2009-2010 skill-builder loader.

Source: skill_builder_data_corrected.csv (latin-1). Multi-skill problems are
stored as duplicated rows sharing order_id; we merge them into one interaction
with the union of skill ids. Rows without a skill are dropped (pyKT
convention). ts = order_id (dataset has ordering, no wall clock).
"""
from __future__ import annotations

import polars as pl

from ...config import DATASETS_DIR
from . import register

CSV = (
    DATASETS_DIR
    / "dataset ASSISTments"
    / "2009-2010"
    / "skill_builder_data_corrected.csv"
)


@register("assist09")
def load(**_) -> dict:
    df = pl.read_csv(
        CSV,
        encoding="latin-1",
        columns=["order_id", "user_id", "problem_id", "correct", "skill_id", "skill_name"],
        schema_overrides={
            "order_id": pl.Int64,
            "user_id": pl.Int64,
            "problem_id": pl.Int64,
            "correct": pl.Int64,
            "skill_id": pl.Utf8,
            "skill_name": pl.Utf8,
        },
        ignore_errors=True,
    )
    df = df.filter(
        pl.col("skill_id").is_not_null()
        & ~pl.col("skill_id").is_in(["", "NA"])  # missing skills = literal "NA"
        & pl.col("correct").is_in([0, 1])
    )

    kc_names = {
        r["skill_id"]: r["skill_name"]
        for r in df.select("skill_id", "skill_name").unique().to_dicts()
        if r["skill_name"]
    }

    # merge duplicated multi-skill rows: one interaction per order_id
    out = (
        df.group_by("order_id", maintain_order=False)
        .agg(
            pl.col("user_id").first().alias("user"),
            pl.col("problem_id").first().alias("question"),
            pl.col("skill_id").unique().alias("kcs"),
            pl.col("correct").first().alias("correct"),
        )
        .with_columns(pl.col("order_id").alias("ts"))
        .select("user", "question", "kcs", "correct", "ts")
    )
    return {"df": out, "kc_names": kc_names}
