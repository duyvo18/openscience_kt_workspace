"""Training harness: fit / evaluate / predict, AMP, early stopping, resume.

Instrumentation (per epoch, written to runs-50-epochs/<run_name>/log.csv by
default, or runs-200-epochs/<run_name>/log.csv when DPA_KT_RUNS_200 is set):
  loss + component losses, train/val AUC/ACC/RMSE, lr, epoch seconds,
  throughput (interactions/s), peak GPU memory.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import torch

from ..config import RUNS_DIR, RUNS_DIR_200, Config
from ..utils import Timer, count_parameters
from .checkpoint import load_checkpoint, save_checkpoint
from .csv_logger import CSVLogger
from .metrics import compute_metrics


def _default_run_root() -> Path:
    """Pick the output root for a run.

    The 200-epoch + 5-fold sweep sets DPA_KT_RUNS_200 so every fold lands
    under runs-200-epochs/. Everything else (master notebook, ad-hoc
    training, ablation grid) keeps using runs-50-epochs/.
    """
    override = os.environ.get("DPA_KT_RUNS_200")
    return Path(override) if override else RUNS_DIR


class Trainer:
    def __init__(self, model: torch.nn.Module, cfg: Config, run_dir: Path | None = None):
        self.cfg = cfg
        self.device = torch.device(
            cfg.device if torch.cuda.is_available() else "cpu"
        )
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=2
        )
        self.amp = cfg.amp and self.device.type == "cuda"
        self.run_dir = Path(run_dir or (_default_run_root() / cfg.run_name))
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.logger = CSVLogger(self.run_dir / "log.csv")
        self.epoch = 0
        self.global_step = 0
        self.best_val_auc = -1.0
        self._no_improve = 0

    # ------------------------------------------------------------------
    def _forward(self, batch, return_trace=False):
        batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
        ctx = (
            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            if self.amp
            else torch.autocast(device_type="cpu", enabled=False)
        )
        with ctx:
            return self.model(batch, return_trace=return_trace), batch

    # ------------------------------------------------------------------
    def train_epoch(self, loader) -> tuple[dict, float]:
        self.model.train()
        agg: dict[str, float] = {}
        n_inter = 0
        with Timer() as timer:
            for batch in loader:
                out, batch = self._forward(batch)
                loss = out["loss"]
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                gnorm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.cfg.grad_clip
                )
                # skip corrupt steps rather than poisoning weights with NaN/inf
                if not torch.isfinite(gnorm):
                    self._skipped = getattr(self, "_skipped", 0) + 1
                    continue
                self.optimizer.step()
                self.global_step += 1
                bs = batch["selectmask"].sum().item()
                n_inter += int(bs)
                agg["loss"] = agg.get("loss", 0.0) + float(loss) * bs
                for k, v in out["aux"].items():
                    agg[k] = agg.get(k, 0.0) + float(v) * bs
        agg = {k: v / max(n_inter, 1) for k, v in agg.items()}
        throughput = n_inter / max(timer.elapsed, 1e-9)
        agg["epoch_seconds"] = timer.elapsed
        agg["throughput"] = throughput
        return agg, timer.elapsed

    # ------------------------------------------------------------------
    @torch.no_grad()
    def evaluate(self, loader) -> dict:
        self.model.eval()
        ys, ps = [], []
        for batch in loader:
            out, batch = self._forward(batch)
            sm = (batch["selectmask"].bool() & (batch["q"] > 0))
            ys.append(batch["r"][sm].float().cpu().numpy())
            ps.append(out["y"].float()[sm].cpu().numpy())
        y = np.concatenate(ys)
        p = np.concatenate(ps)
        return compute_metrics(y, p)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, loader, return_trace: bool = False) -> list[dict]:
        """Return per-batch outputs (probabilities and optional trace)."""
        self.model.eval()
        results = []
        for batch in loader:
            out, batch = self._forward(batch, return_trace=return_trace)
            item = {"y": out["y"].float().cpu(), "batch": {k: v.cpu() for k, v in batch.items()}}
            if return_trace:
                item["trace"] = out["trace"]
            results.append(item)
        return results

    # ------------------------------------------------------------------
    def fit(self, train_loader, val_loader, epochs: int | None = None) -> dict:
        epochs = epochs or self.cfg.epochs
        print(
            f"[{self.cfg.run_name}] params={count_parameters(self.model):,} "
            f"device={self.device} amp={self.amp}"
        )
        start = self.epoch
        for _ in range(start, epochs):
            self.epoch += 1
            torch.cuda.reset_peak_memory_stats() if self.device.type == "cuda" else None
            train_stats, _ = self.train_epoch(train_loader)
            val = self.evaluate(val_loader)
            self.scheduler.step(val["auc"])
            peak = (
                torch.cuda.max_memory_allocated() / 1e9
                if self.device.type == "cuda"
                else 0.0
            )
            row = {
                "epoch": self.epoch,
                **{f"train_{k}": round(v, 5) for k, v in train_stats.items()},
                **{f"val_{k}": round(v, 5) for k, v in val.items()},
                "lr": self.optimizer.param_groups[0]["lr"],
                "peak_mem_gb": round(peak, 3),
            }
            self.logger.log(row)
            print(
                f"  epoch {self.epoch:3d}  loss {train_stats['loss']:.4f}  "
                f"val_auc {val['auc']:.4f}  val_acc {val['acc']:.4f}  "
                f"{train_stats['epoch_seconds']:.1f}s  {peak:.2f}GB"
            )

            save_checkpoint(
                self.run_dir / "last.pt", self.model, self.optimizer,
                self.scheduler, self.epoch, self.global_step, self.best_val_auc, self.cfg,
            )
            if val["auc"] > self.best_val_auc:
                self.best_val_auc = val["auc"]
                self._no_improve = 0
                save_checkpoint(
                    self.run_dir / "best.pt", self.model, self.optimizer,
                    self.scheduler, self.epoch, self.global_step,
                    self.best_val_auc, self.cfg,
                )
            else:
                self._no_improve += 1
                if self._no_improve >= self.cfg.patience:
                    print(f"  early stopping (no val AUC gain for {self.cfg.patience})")
                    break
        return {"best_val_auc": self.best_val_auc, "epochs_run": self.epoch}

    # ------------------------------------------------------------------
    def resume(self, path: str | Path) -> None:
        """Restore model/optimizer/scheduler/epoch/RNG and continue training."""
        ckpt = load_checkpoint(
            path, self.model, self.optimizer, self.scheduler,
            map_location=str(self.device),
        )
        self.epoch = ckpt["epoch"]
        self.global_step = ckpt["global_step"]
        self.best_val_auc = ckpt["best_val_auc"]
        print(f"  resumed from {path} at epoch {self.epoch} (best_val_auc {self.best_val_auc:.4f})")
