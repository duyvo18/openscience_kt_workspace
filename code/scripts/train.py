#!/usr/bin/env python
"""Train DPA-KT on a dataset.

Usage:
  python scripts/train.py --dataset assist09
  python scripts/train.py --dataset assist09 --ablation no_prereq
  python scripts/train.py --dataset assist09 --resume runs/assist09_full/last.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dpa_kt.config import load_config
from dpa_kt.data.dataset import make_loader
from dpa_kt.models.dpa_kt import build_model
from dpa_kt.training import Trainer
from dpa_kt.utils import set_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--ablation", default="full")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--overrides", nargs="*", default=[],
                    help="key=value config overrides (int/float/bool/str)")
    args = ap.parse_args()

    def _cast(v: str):
        for fn in (int, float):
            try:
                return fn(v)
            except ValueError:
                pass
        if v.lower() in ("true", "false"):
            return v.lower() == "true"
        return v

    overrides = dict(o.split("=", 1) for o in args.overrides)
    overrides = {k: _cast(v) for k, v in overrides.items()}

    cfg = load_config(args.dataset, ablation=args.ablation, **overrides)
    set_seed(cfg.seed)

    model = build_model(cfg)
    train_dl = make_loader(cfg.dataset, "train", cfg)
    val_dl = make_loader(cfg.dataset, "valid", cfg)
    test_dl = make_loader(cfg.dataset, "test", cfg)

    trainer = Trainer(model, cfg)
    if args.resume:
        trainer.resume(args.resume)
    trainer.fit(train_dl, val_dl, epochs=args.epochs)

    from dpa_kt.training.checkpoint import load_checkpoint

    best = trainer.run_dir / "best.pt"
    if best.exists():
        load_checkpoint(best, trainer.model, map_location=str(trainer.device),
                        restore_rng=False)
    test_metrics = trainer.evaluate(test_dl)
    print(f"[{cfg.run_name}] TEST {test_metrics}")
    import json
    with open(trainer.run_dir / "test_metrics.json", "w") as f:
        json.dump({**test_metrics, "best_val_auc": trainer.best_val_auc,
                   "run_name": cfg.run_name, "dataset": cfg.dataset,
                   "ablation": args.ablation}, f, indent=2)


if __name__ == "__main__":
    main()
