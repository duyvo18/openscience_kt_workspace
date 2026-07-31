#!/usr/bin/env python3
"""Standalone renderer for the "2e" KC-graph dashboard figure.

Lets you iterate on the figure layout (fonts, spacing, panel sizes) by
looking at a PNG on disk, without going through the notebook. The actual
layout logic lives in ``dpa_kt.analysis.visualize.plot_kc_graph_dashboard``
-- this script just calls it and saves the result, so fixes made here
apply to the notebook too once re-run.

Usage:
    python scripts/render_kc_graph_dashboard.py [dataset] [--out PATH]
    python scripts/render_kc_graph_dashboard.py assist09 --out /tmp/kc_dash.png
    python scripts/render_kc_graph_dashboard.py --all   # one PNG per dataset
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import numpy as np

from dpa_kt.data.canonical import load_maps
from dpa_kt.data.kc_graph import graph_path
from dpa_kt.analysis.visualize import plot_kc_graph_dashboard

ALL_DATASETS = ["assist09", "algebra05", "bridge06", "xes3g5m", "assist12", "eedi"]


def render_one(ds: str, out_path: str) -> None:
    gp = graph_path(ds)
    if not gp.exists():
        print(f"{ds}: no KC graph artifact at {gp}, skipping")
        return
    gg = np.load(gp)
    if "prereq_ratio" not in gg.files:
        print(f"{ds}: prereq_ratio missing, skipping")
        return
    maps = load_maps(ds)
    kc_names = maps.get("kc_names") or {}
    if not kc_names and "kc_map" in maps:
        kc_names = {str(k): str(v)[:40] for k, v in maps["kc_map"].items()}

    fig = plot_kc_graph_dashboard(
        ds, gg["P_rel"], gg["N_rel"], gg["prereq_ratio"], kc_names,
    )
    fig.savefig(out_path, dpi=110)
    print(f"{ds}: saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", nargs="?", default="assist09",
                    help="dataset name (default: assist09)")
    ap.add_argument("--out", default=None,
                    help="output PNG path (default: <script_dir>/_render_out/<ds>.png)")
    ap.add_argument("--all", action="store_true",
                    help="render every dataset in ALL_DATASETS instead of just one")
    args = ap.parse_args()

    out_dir = os.path.join(os.path.dirname(__file__), "_render_out")
    os.makedirs(out_dir, exist_ok=True)

    datasets = ALL_DATASETS if args.all else [args.dataset]
    for ds in datasets:
        out_path = args.out or os.path.join(out_dir, f"kc_dashboard_{ds}.png")
        render_one(ds, out_path)


if __name__ == "__main__":
    main()
