#!/usr/bin/env python
"""Preprocess a dataset end-to-end: loader -> canonical -> sequences -> KC graph.

Usage: python scripts/preprocess.py --dataset assist09 [--force]
       python scripts/preprocess.py --dataset all
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpa_kt.config import load_config
from dpa_kt.data import canonical, kc_graph, sequences
from dpa_kt.data.loaders import LOADERS

ALL = ["assist09", "algebra05", "bridge06", "xes3g5m", "assist12", "eedi", "junyi"]


def preprocess(name: str, force: bool = False) -> None:
    cfg = load_config(name)
    t0 = time.time()

    seq_path = sequences.sequences_path(name)
    g_path = kc_graph.graph_path(name)
    if not force and seq_path.exists() and g_path.exists():
        print(f"[{name}] cached, skipping (use --force to rebuild)")
        return

    if name not in LOADERS:
        raise SystemExit(f"[{name}] loader not implemented/importable")

    print(f"[{name}] loading raw data ...")
    res = LOADERS[name](cfg=cfg)

    if "df" in res:  # standard path; xes3g5m writes sequences directly
        print(f"[{name}] building canonical parquet ...")
        canonical.build_canonical(
            name, res["df"], cfg,
            hierarchy_edges=res.get("hierarchy_edges"),
            kc_names=res.get("kc_names"),
        )
        del res
        print(f"[{name}] building sequences ...")
        sequences.build_sequences(name, cfg)

    print(f"[{name}] building KC graph ...")
    kc_graph.build_kc_graph(name, cfg)
    print(f"[{name}] done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help=f"one of {ALL} or 'all'")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    targets = ALL if args.dataset == "all" else [args.dataset]
    for t in targets:
        preprocess(t, force=args.force)
