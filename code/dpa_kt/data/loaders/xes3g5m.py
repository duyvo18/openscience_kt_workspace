"""XES3G5M loader (question level).

The dataset ships pre-sequenced in pyKT format (comma-joined per-student
lists). We explode those lists back to canonical interactions so the standard
sequencer + KC-graph builder apply uniformly. Multi-KC steps use '_' inside a
concept token (e.g. '62_65'); padding steps (response not in {0,1}) are
dropped. KC names come from metadata/kc_routes_map.json.
"""
from __future__ import annotations

import json

import polars as pl

from ...config import DATASETS_DIR
from . import register

BASE = DATASETS_DIR / "dataset XES3G5M (Google Drive)" / "XES3G5M"
QL = BASE / "question_level"


def _explode(path) -> pl.DataFrame:
    df = pl.read_csv(
        path,
        columns=["uid", "questions", "concepts", "responses", "timestamps"],
        schema_overrides={"uid": pl.Int64},
    )
    df = df.with_columns(
        pl.col("questions").str.split(","),
        pl.col("concepts").str.split(","),
        pl.col("responses").str.split(","),
        pl.col("timestamps").str.split(","),
    ).explode(["questions", "concepts", "responses", "timestamps"])
    df = df.filter(pl.col("responses").is_in(["0", "1"]))
    return df.select(
        pl.col("uid").alias("user"),
        pl.col("questions").alias("question"),
        pl.col("concepts").str.split("_").alias("kcs"),
        pl.col("responses").cast(pl.Int64).alias("correct"),
        pl.col("timestamps").cast(pl.Int64).alias("ts"),
    )


@register("xes3g5m")
def load(**_) -> dict:
    frames = [
        _explode(QL / "train_valid_sequences_quelevel.csv"),
        _explode(QL / "test_quelevel.csv"),
    ]
    df = pl.concat(frames)
    kc_names = {}
    kc_map_path = BASE / "metadata" / "kc_routes_map.json"
    if kc_map_path.exists():
        with open(kc_map_path) as f:
            kc_names = {str(k): v for k, v in json.load(f).items()}
    return {"df": df, "kc_names": kc_names}
