# Knowledge Index

- `knowledge/system.md`: `evolve-agent` model, duties, workspace, and memory structure.
- `knowledge/current-state.md`: current goal, stage, and task status.

## Project layout (DPA-KT)
- Code root: `code/`. Package: `code/dpa_kt/`. Scripts: `code/scripts/`.
- Datasets: `code/datasets/` (raw) + `code/data_cache/sequences/<ds>/` (canonical).
- Runs: `code/runs-50-epochs/` (original 50-epoch sweep + ablations) and
  `code/runs-200-epochs/` (200-epoch + 5-fold CV; set via `DPA_KT_RUNS_200`).
- Config: `code/configs/base.yaml` overrides `epochs=200`, `patience=10`,
  `tbptt=5`. Per-dataset files tweak `batch_size` and `d_v`.
- Trainer output root resolution:
  `_default_run_root()` in `code/dpa_kt/training/trainer.py` honors the
  `DPA_KT_RUNS_200` env var; default is `RUNS_DIR` (`runs-50-epochs/`).
- Sweep driver: `code/scripts/train_200_cv.py` (sequential, 35 GB RAM gate,
  idempotent via `test_metrics.json`).