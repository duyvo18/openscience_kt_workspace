#!/usr/bin/env python
"""Probe the largest batch_size that fits under RAM_CAP_GB for each dataset.

Runs 1 short train epoch + 1 val pass per (dataset, batch_size) candidate,
records peak GPU memory, and prints the recommended batch_size per dataset
(the largest candidate whose peak stays <= RAM_CAP_GB * 0.95).

Usage:
  python scripts/probe_batch.py                 # all 7 datasets, default candidates
  python scripts/probe_batch.py --datasets assist09 xes3g5m
  python scripts/probe_batch.py --cap 32        # tighten the cap
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PY = "./venv/bin/python"

# Conservative starting points (the current config) and an aggressive upper
# bound. The script will pick the largest size whose peak stays under cap.
DEFAULT_CANDIDATES = [64, 128, 192, 256, 320, 384, 448, 512]
DATASETS = ["assist09", "algebra05", "bridge06", "xes3g5m",
            "assist12", "eedi", "junyi"]


def _peak_gb() -> float | None:
    """Currently-irrelevant helper kept for backwards compatibility.

    The probe reads the peak from the training subprocess's log.csv
    (the trainer already records `peak_mem_gb` per epoch).
    """
    return None


def probe(ds: str, batch_size: int, cap: float, run_root: Path,
          epochs: int = 1, eval_batch_size: int = 128) -> tuple[bool, float | None, str]:
    """Return (ok, peak_gb, log_excerpt)."""
    import shutil
    run_name = f"_probe_{ds}_b{batch_size}"
    log_path = run_root / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY, "scripts/train.py",
        "--dataset", ds,
        "--ablation", "full",
        "--epochs", str(epochs),
        "--overrides",
        f"run_name={run_name}",
        f"batch_size={batch_size}",
        f"eval_batch_size={eval_batch_size}",
        "patience=9999",         # no early stop on a 1-epoch probe
        "num_workers=0",         # simpler, less RAM variance
    ]
    env = os.environ.copy()
    env["DPA_KT_RUNS_200"] = str(run_root)
    env["PYTHONUNBUFFERED"] = "1"

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0

    # Peak = max of `peak_mem_gb` over all rows of log.csv (if present)
    peak = None
    log_csv = run_root / run_name / "log.csv"
    if log_csv.exists():
        import csv
        with open(log_csv) as f:
            rdr = csv.DictReader(f)
            vals = [float(r["peak_mem_gb"]) for r in rdr
                    if r.get("peak_mem_gb")]
        if vals:
            peak = max(vals)

    log = log_path.read_text() if log_path.exists() else ""
    excerpt = ""
    for line in log.splitlines():
        if "epoch" in line and "loss" in line and "GB" in line:
            excerpt = line.strip()

    # clean up the probe run dir (keep log file)
    rd = run_root / run_name
    if rd.exists():
        shutil.rmtree(rd, ignore_errors=True)

    ok = (proc.returncode == 0) and (peak is not None) and (peak <= cap)
    return ok, peak, f"{excerpt}  ({dt:.0f}s, rc={proc.returncode})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=DATASETS)
    ap.add_argument("--candidates", nargs="*", type=int,
                    default=DEFAULT_CANDIDATES,
                    help="batch sizes to try, ascending")
    ap.add_argument("--cap", type=float, default=33.0,
                    help="peak VRAM ceiling (GB). Use ~cap*0.95 to leave headroom.")
    ap.add_argument("--run-root", type=Path,
                    default=ROOT / "runs-200-epochs")
    args = ap.parse_args()

    args.run_root.mkdir(parents=True, exist_ok=True)
    rec: dict[str, dict] = {}
    for ds in args.datasets:
        print(f"\n=== probing {ds} (cap={args.cap} GB) ===", flush=True)
        best = None
        for b in sorted(set(args.candidates)):
            gc.collect()
            ok, peak, msg = probe(ds, b, args.cap, args.run_root)
            print(f"  b={b:4d}  ok={ok}  peak={peak}  {msg}", flush=True)
            if not ok:
                break
            best = (b, peak)
        rec[ds] = {
            "best_batch_size": best[0] if best else None,
            "best_peak_gb":    round(best[1], 2) if best else None,
            "cap_gb":          args.cap,
        }

    out = args.run_root / "batch_probe.json"
    out.write_text(json.dumps(rec, indent=2))
    print("\n=== recommended batch_size per dataset ===")
    print(json.dumps(rec, indent=2))
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()