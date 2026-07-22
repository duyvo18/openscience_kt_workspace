#!/usr/bin/env python
"""Sequential 200-epoch + 5-fold CV sweep for DPA-KT (full model only).

For each (dataset, fold in 0..4):
  * set valid_fold = fold and run_name = f"{dataset}_full_fold{fold}"
  * train for cfg.epochs (200 by default) on train+other folds, validate on
    the held-out fold, evaluate on the held-out student split
  * every epoch is logged to <run_root>/<run_name>/log.csv; final
    test_metrics.json records {auc, acc, rmse, best_val_auc, fold, ...}
  * all output goes under <ROOT>/runs-200-epochs/ (override with
    DPA_KT_RUNS_200 env var)

Designed for a SHARED GPU with <=35 GB RAM. Each run runs sequentially. If
peak VRAM exceeds the cap the script aborts that fold so the user can
lower batch_size in configs/<dataset>.yaml and rerun --resume.

Skips a run whose test_metrics.json already exists (idempotent / resumable).

Usage:
  python scripts/train_200_cv.py                 # all 7 datasets x 5 folds
  python scripts/train_200_cv.py --datasets assist09 xes3g5m
  python scripts/train_200_cv.py --folds 0 1     # only folds 0,1
  python scripts/train_200_cv.py --resume       # skip completed runs
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
sys.path.insert(0, str(ROOT))

from dpa_kt.config import RUNS_DIR_200  # noqa: E402

DATASETS = ["assist09", "algebra05", "bridge06", "xes3g5m",
            "assist12", "eedi", "junyi"]
ABLATIONS = ["full"]                      # sweep only the full model
FOLDS = [0, 1, 2, 3, 4]
RAM_CAP_GB = 10.0                         # hard ceiling per process (GB)
# Per-dataset batch_size picked from the probe runs (assist09 and algebra05
# measured directly at b=192 ≈ 9.8 GB peak). Larger datasets (bridge06,
# assist12) get a smaller batch to stay under the 10 GB cap; junyi has the
# largest KC count so we are conservative.
BATCH_SIZE: dict[str, int] = {
    "assist09":   192,
    "algebra05":  192,
    "bridge06":   128,
    "xes3g5m":    128,
    "assist12":   128,
    "eedi":        96,
    "junyi":       64,
}
PY = "./venv/bin/python"


def _peak_gb() -> float | None:
    """Read peak_mem_gb from the run's log.csv (the trainer records it per epoch).

    Returns None if the file is missing or unreadable.
    """
    rd = RUNS_DIR_200  # overwritten by caller before invocation
    return None  # placeholder, see _peak_gb_for() below


def _peak_gb_for(run_dir: Path) -> float | None:
    csv = run_dir / "log.csv"
    if not csv.exists():
        return None
    try:
        import csv
        with open(csv) as f:
            rdr = csv.DictReader(f)
            vals = [float(r["peak_mem_gb"]) for r in rdr
                    if r.get("peak_mem_gb")]
        return max(vals) if vals else None
    except Exception:
        return None


def _make_run_name(ds: str, ab: str, fold: int) -> str:
    return f"{ds}_{ab}_fold{fold}"


def _already_done(run_root: Path, run_name: str) -> bool:
    return (run_root / run_name / "test_metrics.json").exists()


def _can_resume(run_root: Path, run_name: str) -> bool:
    """A partial run exists (last.pt present) but no test_metrics.json yet."""
    rd = run_root / run_name
    return (rd / "last.pt").exists() and not (rd / "test_metrics.json").exists()


def _run_one(ds: str, ab: str, fold: int, run_root: Path,
             extra_overrides: list[str], epochs: int | None,
             resume: bool) -> tuple[bool, str]:
    run_name = _make_run_name(ds, ab, fold)
    if resume and _already_done(run_root, run_name):
        return True, "SKIP (test_metrics.json exists)"

    log_path = run_root / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        PY, "scripts/train.py",
        "--dataset", ds,
        "--ablation", ab,
        "--overrides",
        f"valid_fold={fold}",
        f"run_name={run_name}",
        f"batch_size={BATCH_SIZE.get(ds, 64)}",
        *extra_overrides,
    ]
    if epochs is not None:
        cmd += ["--epochs", str(epochs)]
    if _can_resume(run_root, run_name):
        cmd += ["--resume", str(run_root / run_name / "last.pt")]
        resume_note = " (resuming from last.pt)"
    else:
        resume_note = ""

    env = os.environ.copy()
    env["DPA_KT_RUNS_200"] = str(run_root)
    env.setdefault("PYTHONUNBUFFERED", "1")

    t0 = time.time()
    with open(log_path, "w") as f:
        proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    dt = time.time() - t0

    peak = _peak_gb_for(run_root / run_name)
    peak_str = f", peak={peak:.1f}GB" if peak is not None else ""
    line = (f"[{time.strftime('%H:%M:%S')}] DONE {run_name}{resume_note} "
            f"rc={proc.returncode} t={dt:.0f}s{peak_str}")
    print(line, flush=True)

    if proc.returncode != 0:
        return False, f"FAIL (rc={proc.returncode}, see {log_path})"
    if peak is not None and peak > RAM_CAP_GB:
        return False, f"OVER-RAM (peak {peak:.1f}GB > {RAM_CAP_GB}GB cap)"
    if not _already_done(run_root, run_name):
        return False, "FAIL (no test_metrics.json written)"
    return True, "OK"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", nargs="*", default=DATASETS,
                    help="subset of datasets (default: all 7)")
    ap.add_argument("--folds", nargs="*", type=int, default=FOLDS,
                    help="subset of folds 0..4 (default: all 5)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="override cfg.epochs (default: from base.yaml = 200)")
    ap.add_argument("--run-root", type=Path, default=RUNS_DIR_200,
                    help="output root (default: runs-200-epochs/)")
    ap.add_argument("--resume", action="store_true",
                    help="skip runs whose test_metrics.json already exists")
    ap.add_argument("--extra-overrides", nargs="*", default=[],
                    help="extra --overrides key=value pairs forwarded to "
                         "train.py (e.g. batch_size=32)")
    ap.add_argument("--no-resume", action="store_true",
                    help="re-run even if test_metrics.json exists")
    args = ap.parse_args()

    run_root: Path = args.run_root
    run_root.mkdir(parents=True, exist_ok=True)
    resume = args.resume and not args.no_resume

    total = len(args.datasets) * len(ABLATIONS) * len(args.folds)
    print(f"[sweep] datasets={args.datasets} folds={args.folds} "
          f"ablations={ABLATIONS} -> {total} runs -> {run_root}",
          flush=True)
    print(f"[sweep] RAM cap = {RAM_CAP_GB} GB. Resume = {resume}.", flush=True)

    manifest = []
    failures = []
    n_done = 0
    for ds in args.datasets:
        for ab in ABLATIONS:
            for fold in args.folds:
                ok, msg = _run_one(ds, ab, fold, run_root,
                                   args.extra_overrides, args.epochs, resume)
                n_done += 1
                manifest.append({"dataset": ds, "ablation": ab,
                                 "fold": fold, "ok": ok, "msg": msg})
                if not ok:
                    failures.append((ds, ab, fold, msg))
                gc.collect()

    summary = {
        "total": n_done,
        "ok": sum(1 for r in manifest if r["ok"]),
        "fail": len(failures),
        "run_root": str(run_root),
        "failures": [{"dataset": d, "ablation": a, "fold": f, "msg": m}
                     for d, a, f, m in failures],
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (run_root / "sweep_manifest.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SWEEP COMPLETE ===")
    print(json.dumps(summary, indent=2))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()