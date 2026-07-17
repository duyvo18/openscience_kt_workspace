"""KC prerequisite/neighbor graph construction from the train split.

The paper treats the knowledge graph G as a given, validated input; our
datasets ship no explicit prerequisite graph, so we estimate one from data
(train_valid students only, so the test split never leaks in):

  Neighbors  N_rel: PMI over (a) KC co-membership within a question and
             (b) KC adjacency within a 3-interaction window per student;
             top-5 per KC, symmetrized; dataset hierarchy edges merged in.
  Prereqs    P_rel: directed c -> c' when first_encounter(c) precedes
             first_encounter(c') for >= 65% of students who met both
             (min support 10 students); top-5 per KC by support.
             Hierarchy edges (parent -> child) are added as prerequisites.

Also emits q_rel (V_q, K_rel): per-question GRAPH-EXPANDED related-KC index
table (prerequisites of the question's KCs, then neighbors; the question's
own KCs are EXCLUDED — the model prepends them from the per-step kc array,
which stays correct even for rare-bucketed or hash-collided question ids).
Rows for the padding/rare buckets stay -1.

Output data_cache/graphs/<ds>.npz: P_rel (C,C) uint8, N_rel (C,C) uint8,
q_rel (V_q,K_rel) int16, plus float score matrices for visualization.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, DATA_CACHE
from .canonical import canonical_paths, load_maps
from .sequences import load_sequences

WINDOW = 3
TOP_K = 5
PREREQ_RATIO = 0.65
PREREQ_SUPPORT = 10


def graph_path(dataset: str) -> Path:
    return DATA_CACHE / "graphs" / f"{dataset}.npz"


def build_kc_graph(dataset: str, cfg: Config) -> Path:
    out_path = graph_path(dataset)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    maps = load_maps(dataset)
    C, V_q = maps["n_kcs"], maps["n_questions"]

    seq = load_sequences(dataset)
    test_uids = set(np.asarray(seq["uid"])[np.asarray(seq["split"]) == 1].tolist())

    df = pl.read_parquet(canonical_paths(dataset)[0])
    df = df.filter(~pl.col("uid").is_in(list(test_uids))).sort(
        "uid", "ts", maintain_order=True
    )

    cooc = np.zeros((C, C), dtype=np.float64)   # co-membership + window adjacency
    order = np.zeros((C, C), dtype=np.int32)    # order[i,j]: first(i) < first(j)
    both = np.zeros((C, C), dtype=np.int32)     # students who met both i and j
    kc_freq = np.zeros(C, dtype=np.float64)

    q_kcs: dict[int, list[int]] = {}

    for (uid,), g in df.group_by("uid", maintain_order=True):
        kc_lists = g["kcs"].to_list()
        qids = g["qid"].to_list()
        # per-interaction unique KC sets
        step_kcs = [list(dict.fromkeys(lst)) for lst in kc_lists]
        for qid, lst in zip(qids, step_kcs):
            q_kcs.setdefault(qid, lst)
        flat_first: dict[int, int] = {}
        for t, lst in enumerate(step_kcs):
            for c in lst:
                if c not in flat_first:
                    flat_first[c] = t
                kc_freq[c] += 1
            # (a) co-membership within a question
            arr = np.array(lst)
            if len(arr) > 1:
                cooc[np.ix_(arr, arr)] += 1.0
            # (b) adjacency within WINDOW consecutive interactions
            for dt in range(1, WINDOW):
                if t - dt < 0:
                    break
                prev = np.array(step_kcs[t - dt])
                if len(prev) and len(arr):
                    cooc[np.ix_(prev, arr)] += 0.5
                    cooc[np.ix_(arr, prev)] += 0.5
        # first-encounter ordering over this student's KC set
        ks = np.array(list(flat_first.keys()))
        ts_ = np.array(list(flat_first.values()))
        if len(ks) > 1:
            earlier = ts_[:, None] < ts_[None, :]
            order[np.ix_(ks, ks)] += earlier
            both[np.ix_(ks, ks)] += 1

    np.fill_diagonal(cooc, 0.0)

    # --- neighbors via PMI, top-K per row, symmetrized ---
    total = cooc.sum() + 1e-9
    p_i = cooc.sum(1) / total
    pmi = np.log((cooc / total + 1e-12) / (p_i[:, None] * p_i[None, :] + 1e-12))
    pmi[cooc == 0] = -np.inf
    N_rel = np.zeros((C, C), dtype=np.uint8)
    for i in range(C):
        nz = np.where(np.isfinite(pmi[i]))[0]
        if len(nz):
            top = nz[np.argsort(pmi[i, nz])[-TOP_K:]]
            N_rel[i, top] = 1
    N_rel |= N_rel.T

    # --- prerequisites via first-encounter ordering ---
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(both > 0, order / np.maximum(both, 1), 0.0)
    cand = (ratio >= PREREQ_RATIO) & (both >= PREREQ_SUPPORT)
    P_rel = np.zeros((C, C), dtype=np.uint8)
    for j in range(C):  # keep top-K prerequisites per TARGET KC j
        pre = np.where(cand[:, j])[0]
        if len(pre):
            top = pre[np.argsort(both[pre, j])[-TOP_K:]]
            P_rel[top, j] = 1
    np.fill_diagonal(P_rel, 0)

    # --- merge dataset hierarchy edges ---
    for a, b in maps.get("hierarchy_edges", []):
        N_rel[a, b] = N_rel[b, a] = 1
        P_rel[a, b] = 1  # parent/earlier-level -> child

    # --- per-question graph-expanded related-KC table (own KCs excluded) ---
    q_rel = np.full((V_q, cfg.k_rel), -1, dtype=np.int16)
    for qid, own in q_kcs.items():
        if qid <= 1:  # padding / rare bucket: no reliable KC identity
            continue
        rel: list[int] = []
        for c in own:
            rel.extend(np.where(P_rel[:, c])[0].tolist())
        for c in own:
            rel.extend(np.where(N_rel[c])[0].tolist())
        rel = [c for c in dict.fromkeys(rel) if c not in set(own)][: cfg.k_rel]
        q_rel[qid, : len(rel)] = rel

    np.savez_compressed(
        out_path, P_rel=P_rel, N_rel=N_rel, q_rel=q_rel,
        pmi=np.nan_to_num(pmi, neginf=0.0).astype(np.float32),
        prereq_ratio=ratio.astype(np.float32), kc_freq=kc_freq.astype(np.float32),
    )
    return out_path
