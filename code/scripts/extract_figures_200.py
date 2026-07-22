#!/usr/bin/env python
"""Wait for the 200-epoch sweep to finish, then extract figures + embed them.

Produces:
  * `code/notebooks/figures-200/<dataset>_fold<i>_curves.png` per fold
  * `code/notebooks/figures-200/cv_summary.png`              (one big CV plot)
  * `code/notebooks/figures-200/throughput_memory.png`      (perf table plot)
  * `code/notebooks/figures-200/per_dataset_composite.png`   (5x3 grid)
  * Updated `notebooks/DPA_KT_200_epochs_ENG.ipynb` and
    `notebooks/DPA_KT_200_epochs_VN.ipynb` with embedded images.

Usage:
  python scripts/extract_figures_200.py [--wait] [--no-embed]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"
FIG_DIR = NB_DIR / "figures-200"
RUNS = NB_DIR.parent / "runs-200-epochs"
SWEEP_LOG = RUNS / "sweep.log"
SWEEP_MANIFEST = RUNS / "sweep_manifest.json"

DATASETS = ["assist09", "algebra05", "bridge06", "xes3g5m",
            "assist12", "eedi", "junyi"]
FOLDS = [0, 1, 2, 3, 4]


# ---------------------------------------------------------------- wait ----
def _is_sweep_done() -> bool:
    """True iff the sweep has written its manifest (success or otherwise)."""
    if not SWEEP_MANIFEST.exists():
        return False
    try:
        m = json.loads(SWEEP_MANIFEST.read_text())
        return m.get("total", 0) > 0 and m.get("finished_at")
    except Exception:
        return False


def wait_for_sweep(poll_s: int = 60, timeout_s: int | None = None) -> bool:
    """Block until the sweep finishes. Returns True if finished."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    while True:
        if _is_sweep_done():
            return True
        if timeout_s is not None and time.time() - t0 > timeout_s:
            print(f"[wait] timeout after {timeout_s}s; continuing with partial results")
            return False
        n_done = sum(
            1 for ds in DATASETS for f in FOLDS
            if (RUNS / f"{ds}_full_fold{f}" / "test_metrics.json").exists()
        )
        print(f"[wait] {n_done}/35 folds done — sleeping {poll_s}s "
              f"(manifest={'yes' if SWEEP_MANIFEST.exists() else 'no'})",
              flush=True)
        time.sleep(poll_s)


# ----------------------------------------------------------------- figures -
def _per_fold_curves(ds: str, fold: int, out: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import pandas as pd

    log_csv = RUNS / f"{ds}_full_fold{fold}" / "log.csv"
    if not log_csv.exists():
        return None
    d = pd.read_csv(log_csv)

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(d["epoch"], d["train_loss"], "-", color="steelblue", label="train loss")
    ax[0].plot(d["epoch"], d["train_bce"], "--", color="steelblue", alpha=0.5,
                label="train BCE")
    if "val_auc" in d:
        ax0b = ax[0].twinx()
        ax0b.plot(d["epoch"], d["val_auc"], "-", color="seagreen", label="val AUC")
        ax0b.set_ylabel("val AUC", color="seagreen")
    ax[0].set_xlabel("epoch"); ax[0].set_title(f"{ds} fold{fold} — loss & val AUC")
    ax[0].legend(loc="upper left")

    if "val_acc" in d:
        ax[1].plot(d["epoch"], d["val_acc"], "-", color="seagreen", label="val ACC")
    ax[1].set_xlabel("epoch")
    ax[1].set_ylabel("val ACC")
    if "val_auc" in d:
        ax[1].set_ylim(0.5, 1.0)
        best = d.loc[d["val_auc"].idxmax()]
        ax[1].scatter([best["epoch"]], [best["val_auc"]],
                      color="red", s=30, zorder=5,
                      label=f"best e{int(best['epoch'])}={best['val_auc']:.3f}")
    ax[1].set_title(f"{ds} fold{fold} — val ACC")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _cv_summary_plot(out: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    rows = []
    for ds in DATASETS:
        aucs, accs, rmses = [], [], []
        for f in FOLDS:
            tm = RUNS / f"{ds}_full_fold{f}" / "test_metrics.json"
            if not tm.exists():
                continue
            m = json.loads(tm.read_text())
            aucs.append(m["auc"]); accs.append(m["acc"]); rmses.append(m["rmse"])
        if not aucs:
            continue
        rows.append({
            "dataset": ds,
            "n": len(aucs),
            "auc_mean": np.mean(aucs),
            "auc_std":  np.std(aucs, ddof=1) if len(aucs) > 1 else 0.0,
            "acc_mean": np.mean(accs),
            "acc_std":  np.std(accs, ddof=1) if len(accs) > 1 else 0.0,
            "rmse_mean": np.mean(rmses),
        })
    if not rows:
        return None
    s = pd.DataFrame(rows).set_index("dataset")

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].bar(s.index, s["auc_mean"], yerr=s["auc_std"],
              capsize=4, color="steelblue")
    ax[0].set_title(f"5-fold CV test AUC — mean ± std (n={int(s['n'].iloc[0])} folds)")
    ax[0].set_ylabel("AUC"); ax[0].set_ylim(0.5, 1.0)
    ax[0].tick_params(axis="x", rotation=30)
    ax[1].bar(s.index, s["acc_mean"], yerr=s["acc_std"],
              capsize=4, color="seagreen")
    ax[1].set_title("5-fold CV test ACC — mean ± std")
    ax[1].set_ylabel("ACC"); ax[1].set_ylim(0.5, 1.0)
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _per_dataset_composite(out: Path) -> Path | None:
    """7x5 grid: rows = datasets, cols = folds. Each cell shows the fold's
    val-AUC curve with the best epoch highlighted."""
    import matplotlib.pyplot as plt
    import pandas as pd

    n_ds = len(DATASETS); n_f = len(FOLDS)
    fig, axes = plt.subplots(n_ds, n_f, figsize=(2.6 * n_f, 1.8 * n_ds),
                             squeeze=False)
    for r, ds in enumerate(DATASETS):
        for c, f in enumerate(FOLDS):
            ax = axes[r, c]
            lc = RUNS / f"{ds}_full_fold{f}" / "log.csv"
            if not lc.exists():
                ax.set_title(f"{ds} f{f}\n(pending)", fontsize=7); ax.axis("off")
                continue
            d = pd.read_csv(lc)
            ax.plot(d["epoch"], d["val_auc"], "-", color="seagreen", lw=1)
            best = d.loc[d["val_auc"].idxmax()]
            ax.scatter([best["epoch"]], [best["val_auc"]],
                       color="red", s=15, zorder=5)
            ax.set_title(f"{ds} f{f} ({int(best['epoch'])}:{best['val_auc']:.3f})",
                         fontsize=7)
            ax.tick_params(labelsize=6)
    fig.suptitle("Validation AUC per (dataset, fold) — red dot = best epoch",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _throughput_memory_plot(out: Path) -> Path | None:
    import matplotlib.pyplot as plt
    import pandas as pd

    rows = []
    for ds in DATASETS:
        for f in FOLDS:
            lc = RUNS / f"{ds}_full_fold{f}" / "log.csv"
            if not lc.exists():
                continue
            d = pd.read_csv(lc)
            rows.append({
                "dataset": ds, "fold": f,
                "sec/epoch": float(d["train_epoch_seconds"].mean()),
                "peak_mem_GB": float(d["peak_mem_gb"].max()),
            })
    if not rows:
        return None
    perf = pd.DataFrame(rows)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    pv = perf.pivot(index="dataset", columns="fold", values="peak_mem_GB")
    pv.plot(kind="bar", ax=ax[0], legend=False)
    ax[0].set_title("Peak GPU memory (GB) per (dataset, fold)")
    ax[0].set_ylabel("GB")
    ax[0].axhline(10, ls="--", color="red", label="10 GB cap (shared GPU)")
    ax[0].legend(loc="best", fontsize=8)
    ax[0].tick_params(axis="x", rotation=30)

    pv2 = perf.pivot(index="dataset", columns="fold", values="sec/epoch")
    pv2.plot(kind="bar", ax=ax[1], legend=False)
    ax[1].set_title("Epoch seconds (mean) per (dataset, fold)")
    ax[1].set_ylabel("sec / epoch")
    ax[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def extract_all_figures() -> dict[str, Path]:
    """Generate every per-fold figure + the three summaries.

    Returns a dict mapping figure-name -> path (missing figures are omitted).
    """
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path] = {}

    # 1. per-fold curves
    for ds in DATASETS:
        for f in FOLDS:
            p = FIG_DIR / f"{ds}_fold{f}_curves.png"
            r = _per_fold_curves(ds, f, p)
            if r:
                figs[f"{ds}_fold{f}_curves"] = r

    # 2. CV summary
    p = FIG_DIR / "cv_summary.png"
    r = _cv_summary_plot(p)
    if r:
        figs["cv_summary"] = r

    # 3. composite 7x5 grid
    p = FIG_DIR / "per_dataset_composite.png"
    r = _per_dataset_composite(p)
    if r:
        figs["per_dataset_composite"] = r

    # 4. throughput + memory
    p = FIG_DIR / "throughput_memory.png"
    r = _throughput_memory_plot(p)
    if r:
        figs["throughput_memory"] = r

    return figs


# ----------------------------------------------------------- embed in NB ---
def _to_data_uri(path: Path) -> str:
    """Inline-embed a PNG file as a base64 data URI."""
    import base64
    b64 = base64.b64encode(path.read_bytes()).decode()
    return (f'<img alt="{path.name}" '
            f'src="data:image/png;base64,{b64}" '
            f'style="max-width:100%"/>')


def embed_in_notebook(nb_path: Path, figs: dict[str, Path]) -> None:
    """Inject a markdown cell with the saved PNG (data URI) before the
    conclusions cell. The existing matplotlib cells stay for live refresh."""
    import nbformat

    nb = nbformat.read(nb_path, as_version=4)

    # find the conclusions heading markdown cell
    concl_idx = None
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            if src.startswith("## 7."):
                concl_idx = i
                break
    if concl_idx is None:
        print(f"[embed] no conclusions cell in {nb_path.name}; appending instead")
        concl_idx = len(nb["cells"])

    md_lines = ["## 8. Embedded figures (PNG, generated from sweep)\n",
                "All figures below are saved to `code/notebooks/figures-200/` "
                "and inlined into this notebook for offline viewing.\n"]
    for name, path in figs.items():
        if not path.exists():
            continue
        md_lines.append(f"\n### {name}\n")
        md_lines.append(_to_data_uri(path))
        md_lines.append("\n")
    md_cell = nbformat.v4.new_markdown_cell("\n".join(md_lines))

    # remove any pre-existing embedded-figures section (idempotent)
    to_remove = []
    for i, c in enumerate(nb["cells"]):
        if c["cell_type"] == "markdown":
            src = "".join(c["source"])
            if src.startswith("## 8. Embedded figures"):
                to_remove.append(i)
    for i in reversed(to_remove):
        nb["cells"].pop(i)
        if i < concl_idx:
            concl_idx -= 1

    nb["cells"].insert(concl_idx, md_cell)
    nbformat.write(nb, nb_path)
    print(f"[embed] updated {nb_path}")


# ---------------------------------------------------------------- main ----
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait", action="store_true",
                    help="block until the sweep finishes (polls sweep_manifest.json)")
    ap.add_argument("--poll", type=int, default=60,
                    help="seconds between polls when --wait is set")
    ap.add_argument("--timeout", type=int, default=None,
                    help="max seconds to wait (default: forever)")
    ap.add_argument("--no-embed", action="store_true",
                    help="generate figures but do not modify the notebooks")
    ap.add_argument("--no-figures", action="store_true",
                    help="do not generate figures; only embed existing ones")
    args = ap.parse_args()

    if args.wait:
        ok = wait_for_sweep(poll_s=args.poll, timeout_s=args.timeout)
        if not ok:
            print("[main] proceeding with partial results")

    if not args.no_figures:
        figs = extract_all_figures()
        print(f"[main] generated {len(figs)} figures under {FIG_DIR}")
        for n, p in figs.items():
            print(f"  - {n}: {p.relative_to(ROOT)}")
    else:
        figs = {p.stem: p for p in FIG_DIR.glob("*.png")}
        print(f"[main] reusing {len(figs)} existing figures under {FIG_DIR}")

    if not args.no_embed:
        for lang in ("ENG", "VN"):
            nb = NB_DIR / f"DPA_KT_200_epochs_{lang}.ipynb"
            if nb.exists():
                embed_in_notebook(nb, figs)


if __name__ == "__main__":
    main()