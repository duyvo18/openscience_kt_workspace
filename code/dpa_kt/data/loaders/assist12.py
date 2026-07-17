"""ASSISTments 2012-2013 loader.

full_data.csv is a ZIP wrapping a ~3GB CSV; we extract it once to
data_cache/raw/assist12/ then stream with polars. Single KC per row
(skill_id). problem_log_id (monotonic) serves as the ordering timestamp,
avoiding datetime parsing over millions of rows.
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import polars as pl

from ...config import DATA_CACHE, DATASETS_DIR
from . import register

ZIP = (
    DATASETS_DIR / "dataset ASSISTments"
    / "2012-13-school-data-with-affect" / "full_data.csv"
)
INNER = "2012-2013-data-with-predictions-4-final.csv"
RAW_DIR = DATA_CACHE / "raw" / "assist12"


def _extract_once() -> Path:
    out = RAW_DIR / INNER
    if not out.exists():
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ZIP) as z:
            z.extract(INNER, RAW_DIR)
    return out


@register("assist12")
def load(**_) -> dict:
    csv_path = _extract_once()
    lf = pl.scan_csv(
        csv_path,
        schema_overrides={"skill_id": pl.Utf8, "skill": pl.Utf8},
        infer_schema_length=10000,
        ignore_errors=True,
    ).select(
        pl.col("user_id").alias("user"),
        pl.col("problem_id").alias("question"),
        pl.col("skill_id"),
        pl.col("skill"),
        pl.col("correct"),
        pl.col("problem_log_id").alias("ts"),
    ).filter(
        pl.col("skill_id").is_not_null()
        & (pl.col("skill_id") != "")
        & pl.col("correct").is_in([0, 1])
    )
    df = lf.collect(engine="streaming")

    kc_names = {
        r["skill_id"]: r["skill"]
        for r in df.select("skill_id", "skill").unique().to_dicts()
        if r["skill"]
    }
    out = df.select(
        pl.col("user"),
        pl.col("question"),
        pl.concat_list(pl.col("skill_id")).alias("kcs"),  # single-KC -> 1-elem list
        pl.col("correct"),
        pl.col("ts"),
    )
    return {"df": out, "kc_names": kc_names}
