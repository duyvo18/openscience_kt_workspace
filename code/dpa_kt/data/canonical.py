"""Canonical interaction format shared by all dataset loaders.

Every loader produces a polars DataFrame with columns
    user: str | int   raw student id
    question: str     raw question id
    kcs: list[str]    raw KC ids (>= 1 entries)
    correct: int      0/1
    ts: int           sortable timestamp (epoch seconds or ordinal)
plus optionally a list of raw hierarchy edges between KCs.

`build_canonical` assigns integer ids and writes
    data_cache/canonical/<ds>.parquet   uid,qid,kcs(list[i32]),correct,ts
    data_cache/maps/<ds>.json           vocab maps + stats + hierarchy edges

Question id space: 0 = padding, 1 = rare bucket (freq < min_q_freq),
real questions start at 2. KC ids are 0-based (padding slot uses -1 in
sequence arrays). Huge question vocabs are hash-bucketed to max_questions.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from ..config import Config, DATA_CACHE

Q_PAD, Q_RARE, Q_OFFSET = 0, 1, 2


def canonical_paths(dataset: str) -> tuple[Path, Path]:
    return (
        DATA_CACHE / "canonical" / f"{dataset}.parquet",
        DATA_CACHE / "maps" / f"{dataset}.json",
    )


def build_canonical(
    dataset: str,
    df: pl.DataFrame,
    cfg: Config,
    hierarchy_edges: list[tuple[str, str]] | None = None,
    kc_names: dict[str, str] | None = None,
) -> tuple[Path, Path]:
    """Assign integer ids and persist the canonical parquet + maps sidecar."""
    pq_path, maps_path = canonical_paths(dataset)
    pq_path.parent.mkdir(parents=True, exist_ok=True)
    maps_path.parent.mkdir(parents=True, exist_ok=True)

    df = df.with_columns(
        pl.col("user").cast(pl.Utf8),
        pl.col("question").cast(pl.Utf8),
        pl.col("correct").cast(pl.Int8),
        pl.col("ts").cast(pl.Int64),
    ).sort("user", "ts", maintain_order=True)

    # --- user ids ---
    users = df["user"].unique(maintain_order=True).to_list()
    user_map = {u: i for i, u in enumerate(users)}

    # --- question ids with frequency cap / hash bucketing ---
    q_counts = df["question"].value_counts()
    frequent = q_counts.filter(pl.col("count") >= cfg.min_q_freq)["question"].to_list()
    if len(frequent) > cfg.max_questions - Q_OFFSET:
        q_map = {
            q: Q_OFFSET + (hash(q) % (cfg.max_questions - Q_OFFSET)) for q in frequent
        }
        n_questions = cfg.max_questions
        hashed = True
    else:
        q_map = {q: Q_OFFSET + i for i, q in enumerate(sorted(frequent))}
        n_questions = Q_OFFSET + len(frequent)
        hashed = False

    # --- KC ids (no cap; vocabulary is bounded) ---
    kcs_all = sorted(df["kcs"].explode().drop_nulls().unique().to_list())
    kc_map = {c: i for i, c in enumerate(kcs_all)}

    out = df.with_columns(
        pl.col("user").replace_strict(user_map, return_dtype=pl.Int32).alias("uid"),
        pl.col("question")
        .replace_strict(q_map, default=Q_RARE, return_dtype=pl.Int32)
        .alias("qid"),
        pl.col("kcs")
        .list.eval(pl.element().replace_strict(kc_map, return_dtype=pl.Int32))
        .alias("kcs"),
    ).select("uid", "qid", "kcs", "correct", "ts")
    out.write_parquet(pq_path)

    maps = {
        "dataset": dataset,
        "n_users": len(users),
        "n_questions": n_questions,
        "n_kcs": len(kcs_all),
        "n_interactions": len(out),
        "pos_rate": float(out["correct"].mean()),
        "question_hashed": hashed,
        "kc_map": kc_map,
        "kc_names": {str(kc_map[k]): v for k, v in (kc_names or {}).items() if k in kc_map},
        "hierarchy_edges": [
            [kc_map[a], kc_map[b]]
            for a, b in (hierarchy_edges or [])
            if a in kc_map and b in kc_map
        ],
    }
    with open(maps_path, "w") as f:
        json.dump(maps, f)
    return pq_path, maps_path


def load_maps(dataset: str) -> dict:
    with open(canonical_paths(dataset)[1]) as f:
        return json.load(f)
