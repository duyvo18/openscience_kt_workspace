"""Junyi Academy loader.

Log_Problem.csv is ~2.9GB / 16M rows, streamed with polars. student=uuid,
question=upid, KC=ucid, correct=is_correct ('True'/'False'). timestamp_TW is
parsed to an epoch ordinal for per-student ordering. Info_Content.csv supplies
KC names (content_pretty_name).
"""
from __future__ import annotations

import polars as pl

from ...config import DATASETS_DIR
from . import register

BASE = DATASETS_DIR / "dataset Junyi Academy" / "Junyi"
LOG = BASE / "Log_Problem.csv"
CONTENT = BASE / "Info_Content.csv"


@register("junyi")
def load(**_) -> dict:
    kc_names: dict[str, str] = {}
    if CONTENT.exists():
        content = pl.read_csv(CONTENT, ignore_errors=True)
        if "content_pretty_name" in content.columns:
            kc_names = {
                str(r["ucid"]): r["content_pretty_name"]
                for r in content.iter_rows(named=True)
            }

    lf = pl.scan_csv(
        LOG,
        schema_overrides={"is_correct": pl.Utf8, "uuid": pl.Utf8,
                          "ucid": pl.Utf8, "upid": pl.Utf8},
        ignore_errors=True,
    ).select(
        pl.col("uuid").alias("user"),
        pl.col("upid").alias("question"),
        pl.col("ucid"),
        pl.col("is_correct"),
        pl.col("timestamp_TW"),
    ).filter(
        pl.col("is_correct").is_in(["True", "False", "true", "false"])
        & pl.col("ucid").is_not_null()
    ).with_columns(
        pl.col("is_correct").str.to_lowercase().replace_strict(
            {"true": 1, "false": 0}, return_dtype=pl.Int64
        ).alias("correct"),
        # timestamp_TW like "2018-08-03 07:29:36 UTC" -> epoch seconds
        pl.col("timestamp_TW").str.slice(0, 19)
        .str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        .dt.epoch("s").alias("ts"),
    )
    df = lf.collect(engine="streaming")
    df = df.filter(pl.col("ts").is_not_null())
    out = df.select(
        pl.col("user"),
        pl.col("question"),
        pl.concat_list(pl.col("ucid")).alias("kcs"),
        pl.col("correct"),
        pl.col("ts"),
    )
    return {"df": out, "kc_names": kc_names}
