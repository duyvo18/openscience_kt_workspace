"""Eedi NeurIPS 2020 loader (Task 1/2).

Interactions from train_task_1_2.csv (QuestionId, UserId, IsCorrect, AnswerId);
KCs are the SubjectId list of each question (question_metadata_task_1_2.csv,
a stringified int list). AnswerId is monotonic with time, so it serves as the
ordering timestamp — this avoids the ~1GB join to answer_metadata just for
DateAnswered.
"""
from __future__ import annotations

import ast

import polars as pl

from ...config import DATASETS_DIR
from . import register

BASE = DATASETS_DIR / "dataset Eedi NeurIPS 2020" / "data_extracted" / "data"
TRAIN = BASE / "train_data" / "train_task_1_2.csv"
QMETA = BASE / "metadata" / "question_metadata_task_1_2.csv"
SMETA = BASE / "metadata" / "subject_metadata.csv"


@register("eedi")
def load(**_) -> dict:
    qmeta = pl.read_csv(QMETA)
    q2kc = {
        int(r["QuestionId"]): [str(s) for s in ast.literal_eval(r["SubjectId"])]
        for r in qmeta.iter_rows(named=True)
    }

    kc_names: dict[str, str] = {}
    if SMETA.exists():
        smeta = pl.read_csv(SMETA)
        name_col = "Name" if "Name" in smeta.columns else smeta.columns[1]
        id_col = [c for c in smeta.columns if "SubjectId" in c][0]
        kc_names = {str(r[id_col]): r[name_col] for r in smeta.iter_rows(named=True)}

    df = pl.scan_csv(TRAIN).select(
        pl.col("UserId").alias("user"),
        pl.col("QuestionId").alias("question"),
        pl.col("IsCorrect").alias("correct"),
        pl.col("AnswerId").alias("ts"),
    ).filter(pl.col("correct").is_in([0, 1])).collect(engine="streaming")

    kcs = pl.Series(
        "kcs", [q2kc.get(q, []) for q in df["question"].to_list()], dtype=pl.List(pl.Utf8)
    )
    out = df.with_columns(
        pl.col("question").cast(pl.Utf8), kcs
    ).select("user", "question", "kcs", "correct", "ts")
    out = out.filter(pl.col("kcs").list.len() > 0)
    return {"df": out, "kc_names": kc_names}
