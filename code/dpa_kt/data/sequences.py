"""pyKT-convention sequencing: canonical parquet -> fixed-length npz sequences.

Conventions (mirroring pykt-toolkit so the literature table stays meaningful):
  * students with < min_interactions dropped
  * per-student interactions sorted by ts, chunked to seq_len=200
  * student-level 80/20 train_valid/test split (seeded), 5 folds on train_valid
  * selectmask: loss positions (t >= 1, non-pad)

Output data_cache/sequences/<ds>/ as one raw .npy per array (memmap-friendly;
compressed npz cannot be memmapped and breaks across DataLoader forks):
  q (N,L) int32 | r (N,L) int8 | kc (N,L,K_max) int16 (-1 pad) | ts (N,L) int64
  selectmask (N,L) int8 | fold (N,) int8 (-1 for test) | split (N,) int8 (0 tv/1 test)
  uid (N,) int32
  q_diff_bin (V_q,) uint8 | kc_diff_bin (C,) uint8  (empirical difficulty bins,
      computed on train_valid students only)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from ..config import Config, DATA_CACHE
from .canonical import canonical_paths, load_maps


def sequences_path(dataset: str) -> Path:
    """Directory holding one raw .npy file per sequence array."""
    return DATA_CACHE / "sequences" / dataset


def save_sequence_arrays(dataset: str, arrays: dict[str, np.ndarray]) -> Path:
    out_dir = sequences_path(dataset)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(out_dir / f"{name}.npy", arr)
    return out_dir


def load_sequences(dataset: str, mmap: bool = True) -> dict[str, np.ndarray]:
    out_dir = sequences_path(dataset)
    mode = "r" if mmap else None
    return {
        p.stem: np.load(p, mmap_mode=mode) for p in sorted(out_dir.glob("*.npy"))
    }


def _difficulty_bins(p_correct: np.ndarray, counts: np.ndarray, n_bins: int) -> np.ndarray:
    """Quantile-bin empirical p-correct; unseen ids get the middle bin."""
    bins = np.full(len(p_correct), n_bins // 2, dtype=np.uint8)
    seen = counts > 0
    if seen.sum() > n_bins:
        edges = np.quantile(p_correct[seen], np.linspace(0, 1, n_bins + 1)[1:-1])
        bins[seen] = np.searchsorted(edges, p_correct[seen]).astype(np.uint8)
    return bins


def build_sequences(dataset: str, cfg: Config) -> Path:
    pq_path, _ = canonical_paths(dataset)
    maps = load_maps(dataset)
    n_questions, n_kcs = maps["n_questions"], maps["n_kcs"]
    assert n_kcs < 32_000, "kc dtype int16 would overflow"

    df = pl.read_parquet(pq_path)

    # --- filter short students, then split at STUDENT level ---
    counts = df.group_by("uid").len()
    keep = counts.filter(pl.col("len") >= cfg.min_interactions)["uid"].to_numpy()
    rng = np.random.default_rng(cfg.seed)
    keep = rng.permutation(np.sort(keep))
    n_test = int(round(0.2 * len(keep)))
    test_uids = set(keep[:n_test].tolist())
    tv_uids = keep[n_test:]
    fold_of = {int(u): int(i % 5) for i, u in enumerate(tv_uids)}

    df = df.filter(pl.col("uid").is_in(keep)).sort("uid", "ts", maintain_order=True)

    # --- difficulty from train_valid rows only ---
    tv_rows = df.filter(~pl.col("uid").is_in(list(test_uids)))
    q_stats = tv_rows.group_by("qid").agg(pl.col("correct").mean(), pl.len())
    q_p = np.full(n_questions, 0.5, dtype=np.float32)
    q_n = np.zeros(n_questions, dtype=np.int64)
    q_p[q_stats["qid"].to_numpy()] = q_stats["correct"].to_numpy()
    q_n[q_stats["qid"].to_numpy()] = q_stats["len"].to_numpy()
    kc_stats = (
        tv_rows.explode("kcs").group_by("kcs").agg(pl.col("correct").mean(), pl.len())
    )
    kc_p = np.full(n_kcs, 0.5, dtype=np.float32)
    kc_n = np.zeros(n_kcs, dtype=np.int64)
    kc_p[kc_stats["kcs"].to_numpy()] = kc_stats["correct"].to_numpy()
    kc_n[kc_stats["kcs"].to_numpy()] = kc_stats["len"].to_numpy()

    # --- chunk per student into fixed-length windows ---
    L, K = cfg.seq_len, cfg.k_max
    qs, rs, kcs_, tss, sms, folds, splits, uids = [], [], [], [], [], [], [], []

    for (uid,), g in df.group_by("uid", maintain_order=True):
        uid = int(uid)
        q_arr = g["qid"].to_numpy()
        r_arr = g["correct"].to_numpy()
        t_arr = g["ts"].to_numpy()
        kc_lists = g["kcs"].to_list()
        is_test = uid in test_uids
        for s in range(0, len(q_arr), L):
            e = min(s + L, len(q_arr))
            if e - s < cfg.min_interactions:
                continue
            n = e - s
            q = np.zeros(L, dtype=np.int32)
            r = np.zeros(L, dtype=np.int8)
            t = np.zeros(L, dtype=np.int64)
            kc = np.full((L, K), -1, dtype=np.int16)
            sm = np.zeros(L, dtype=np.int8)
            q[:n], r[:n], t[:n] = q_arr[s:e], r_arr[s:e], t_arr[s:e]
            for j, lst in enumerate(kc_lists[s:e]):
                kc[j, : min(len(lst), K)] = lst[:K]
            sm[1:n] = 1
            qs.append(q); rs.append(r); tss.append(t); kcs_.append(kc); sms.append(sm)
            folds.append(-1 if is_test else fold_of[uid])
            splits.append(1 if is_test else 0)
            uids.append(uid)

    return save_sequence_arrays(
        dataset,
        {
            "q": np.stack(qs), "r": np.stack(rs), "kc": np.stack(kcs_),
            "ts": np.stack(tss), "selectmask": np.stack(sms),
            "fold": np.array(folds, dtype=np.int8),
            "split": np.array(splits, dtype=np.int8),
            "uid": np.array(uids, dtype=np.int32),
            "q_diff_bin": _difficulty_bins(q_p, q_n, cfg.n_difficulty_bins),
            "kc_diff_bin": _difficulty_bins(kc_p, kc_n, cfg.n_difficulty_bins),
        },
    )
