# DPA-KT — Distributional Pedagogical Alignment for Knowledge Tracing

A reference implementation of the four-module Knowledge Tracing framework
described in `main.pdf` (*Distributional Pedagogical Alignment*), trained
full-scale on five dataset families (seven dataset configs), with per-module
visualizations, an ablation study, a literature comparison table, training-time
instrumentation, and checkpoint-resume.

The paper is conceptual (no equations); this repo makes concrete, pragmatic
implementation choices for each module and documents them inline.

## Model architecture

![Main model architecture](Main%20model%20architecture.png)

*Reference design diagram (`Main model architecture.png`) for the four
modules and the attribution trace.* The diagram specifies Mamba blocks and a
Beta(α, β) distribution; this implementation substitutes trainable-at-this-scale
equivalents documented in the table below (causal Transformer + GRU instead of
Mamba SSM blocks, a Gaussian instead of Beta, and a DKVMN-style multiplicative
erase-add instead of the diagram's additive `M_{t+1} = M_t + ΔM`) — the module
boundaries, data flow, and attribution trace match the diagram exactly.

## The model (`dpa_kt/models/`)

| Module | File | What it does |
|--------|------|--------------|
| 1. Interaction encoding | `interaction_encoder.py` | Dual branch: a causal **Transformer** over (question ⊕ response ⊕ difficulty) and a layer-normalised **GRU** that reads localized mastery; fused into `z_t`. |
| 2. Distributional alignment | `distribution.py`, `patterns.py`, `alignment.py` | Project `z_t → N(μ, diag σ²)`; four fixed **pattern operators** (temporal / same-KC / prerequisite / neighbor) pool the prefix by **moment matching**; alignment losses (monotonicity, guess/slip, KL). |
| 3. Mastery tracking | `mastery.py` | Explicit mastery memory `M`, **pattern→KC gating** `A_i`, DKVMN-style erase-add update on the related KCs. |
| 4. Prediction | `predictor.py` | KC→prediction contributions `β` + learnable guess/slip head. |
| Assembly | `dpa_kt.py` | Time loop (truncated BPTT), total loss `BCE + 0.1·mono + 0.1·gs + 1e-4·kl`, and the intrinsic **attribution trace**. |

~1.3 M trainable parameters. Loss combines BCE with the three alignment terms.

## Hardware used

| Component | Spec |
|-----------|------|
| GPU | NVIDIA **GB10** (Grace-Blackwell, unified memory), driver 580.142, **CUDA 13.0** — shared with other workloads (e.g. a resident vLLM process); training peaks well under 8 GB |
| CPU | ARM **Cortex-X925 / Cortex-A725**, aarch64, 20 cores |
| RAM | 121 GiB unified CPU+GPU memory |
| OS | Ubuntu 24.04.4 LTS (aarch64) |
| Software | Python 3.12, PyTorch 2.13 (`+cu130`), Triton (needs `python3.12-dev` headers to JIT-compile kernels) |

The GPU being shared and unified-memory is why the data/training code is
memory-frugal by design: memmapped `.npy` sequences, `pin_memory=False`,
uint8 relation matrices, and modest per-dataset batch sizes (see `configs/`).

## Setup

```bash
bash scripts/setup_venv.sh          # venv + deps + Jupyter kernel "dpa_kt"
```
Requires PyTorch with CUDA (tested on an NVIDIA GB10 / CUDA 13, aarch64). The
GPU may be shared; training peaks well under 10 GB. Triton kernel compilation
needs the Python dev headers (`python3.12-dev`).

## Model reference — modules, I/O, and roles

The table below is the canonical reference for the four-module architecture;
the same content appears in the notebooks under §2d.

| # | Module | File | Role | Key input | Key output |
|---|--------|------|------|-----------|------------|
| — | `InteractionEmbeddings` | `embeddings.py` | Shared embedding tables for question id, response, KC id, question & KC difficulty bins | `q` (B,L), `r` (B,L), `kc` (B,L,K_max), `q_diff_bin` (V_q), `kc_diff_bin` | `e_q` (B,L,d_emb), `e_r` (B,L,d_emb), `e_dq` (B,L,d_emb), `e_c_mean` (B,L,d), `e_dc_mean` (B,L,d) |
| 1 | `BranchA` | `interaction_encoder.py` | Parallel 1-layer causal Transformer over (question ⊕ response ⊕ difficulty); produces interaction-context representation independent of mastery state | `e_q, e_r, e_dq` (B,L,d_emb), `pad_mask` (B,L) | `h_a` (B,L,d_model) |
| 1 | `BranchBCell` | `interaction_encoder.py` | GRU cell over localized mastery read + concept/difficulty embedding; stepped once per interaction inside the time loop | `m_read` (B,d_v), `e_c` (B,d), `e_dc` (B,d), `h_prev` (B,d_model), `step_valid` (B,) | `h_b` (B,d_model) |
| 1 | `Fusion` | `interaction_encoder.py` | Projects the concatenation of branch-A and branch-B outputs into `z_t`, the unified interaction representation | `h_a` (B,d_model), `h_b` (B,d_model) | `z_t` (B,d_model) |
| 2 | `GaussianProjection` | `distribution.py` | "Statistical-descriptor decoder": linear heads map `z_t → N(μ, diag σ²)` | `z_t` (B,d_model) | `mu` (B,d_z), `logvar` (B,d_z) |
| 2 | `PatternOperators` | `patterns.py` | Four fixed pooling operators (temporal, same-KC, prerequisite, neighbor) that produce per-step weights `w` over the prefix and pool the prefix Gaussians by moment matching | `t`, `mu_prefix` (B,t+1,d_z), `var_prefix` (B,t+1,d_z), `masks`, `pad_mask` (B,L) | `{name: {mu (B,d_z), var (B,d_z), v (B,d_z), w (B,t+1)}}` |
| 3 | `MasteryState` | `mastery.py` | Explicit mastery memory `M_t` (B,C,d_v); DKVMN-style erase-add update gate-controlled by pattern outputs; scalar mastery readout in (0,1) | `M` (B,C,d_v), `rel` (B,K_rel), `patterns` (Module 2 output), `step_valid` (B,) | `M_new` (B,C,d_v), `gates {name: (B,K_rel)}`, scalar mastery (B,C) |
| 3 | `MasteryState.read` | `mastery.py` | Localized attention read over only the related KCs of the current question | `M` (B,C,d_v), `rel` (B,K_rel), `e_q` (B,d_emb) | `m_read` (B,d_v), `alpha` (B,K_rel) |
| 4 | `PredictionHead` | `predictor.py` | KC→prediction contribution weights `β` (attribution trace) + MLP over mastery read + question/difficulty embedding; corrected by soft-capped guess/slip scalars | `M` (B,C,d_v), `rel` (B,K_rel), `kc_key`, `e_q` (B,d_emb), `e_dq` (B,d_emb) | `y_hat` (B,), `beta` (B,K_rel) |
| — | `DPAKT` (assembly) | `dpa_kt.py` | Wires all modules into a truncated-BPTT time loop; Module 4 predicts before seeing `r_t`, then Modules 1-3 update; returns `y`, `loss`, and the full attribution trace | `batch: q,r,kc,selectmask` (all B,L) or (B,L,K_max) | `y` (B,L), `loss` (scalar), `trace: {pattern_w, gates, beta, rel, alpha, mastery, guess, slip}` |

## Data

Place the datasets under `datasets/` (already present here). Preprocess to
cached sequences + KC graph:

```bash
python scripts/preprocess.py --dataset assist09      # one dataset
python scripts/preprocess.py --dataset all           # all seven
```

Dataset keys: `assist09`, `algebra05`, `bridge06`, `xes3g5m`, `assist12`,
`eedi`, `junyi`. Each is loaded to a canonical parquet, cut into pyKT-style
length-200 sequences (drop smaller than 3 interactions, student-level 80/20 split + 5
folds), and a prerequisite/neighbor KC graph is estimated from the train split.
Artifacts cache under `data_cache/`.

## Training

```bash
python scripts/train.py --dataset assist09                       # full model
python scripts/train.py --dataset assist09 --ablation no_prereq  # an ablation
python scripts/train.py --dataset assist09 --resume runs-50-epochs/assist09_full/last.pt
```

Checkpoints (`best.pt`, `last.pt`), a per-epoch `log.csv`, and `test_metrics.json`
are written to `runs-50-epochs/<dataset>_<ablation>/`. The trainer uses AdamW, bf16 AMP,
gradient clipping, `ReduceLROnPlateau` on val AUC, and early stopping; checkpoints
store model+optimizer+scheduler+epoch+RNG for seamless resume.

Ablations (recommended on `assist09` + `xes3g5m`):
```bash
for A in full no_temporal no_samekc no_prereq no_neighbor no_mono no_gs no_distributional single_branch; do
  python scripts/train.py --dataset assist09 --ablation $A
  python scripts/train.py --dataset xes3g5m  --ablation $A
done
```

## The notebook

`notebooks/DPA_KT_master.ipynb` orchestrates everything and renders all results
(13 sections: setup → data → KC graph → per-module demos → full training →
results vs literature → ablation matrix → checkpoint-resume → attribution case
study → conclusions). It holds orchestration/display only; all logic is in the
package. Regenerate it with `python scripts/build_notebook.py`.

## 200-epoch + 5-fold CV sweep

`scripts/train_200_cv.py` retrains the **full** model for up to **200
epochs** with **5-fold cross-validation** on every dataset, writing the
artefacts to `runs-200-epochs/<dataset>_full_fold<i>/`. The sweep runs
sequentially, honours a 35 GB shared-GPU RAM cap, and is idempotent (skips
runs whose `test_metrics.json` already exists).

```bash
./venv/bin/python scripts/train_200_cv.py --resume         # all 7 datasets x 5 folds
./venv/bin/python scripts/train_200_cv.py --datasets assist09 xes3g5m
DPA_KT_RUNS_200=/path/to/root ./venv/bin/python scripts/train_200_cv.py --resume
```

The aggregated results are surfaced in two notebooks — one English, one
Vietnamese:

```bash
./venv/bin/python scripts/build_notebook_200.py             # writes both
./venv/bin/python scripts/build_notebook_200.py eng         # English only
./venv/bin/python scripts/build_notebook_200.py vn          # Vietnamese only
```

`notebooks/DPA_KT_200_epochs_ENG.ipynb` and
`notebooks/DPA_KT_200_epochs_VN.ipynb` mirror the structure of the master
notebook and add 5-fold CV aggregation. They contain:

1. Setup & environment
2. Datasets in the sweep (status table) + 2b. dataset descriptions + 2c. KC-graph
   inspection (node graphs + degree charts per dataset)
3. 5-fold CV per-dataset summary (mean / best / worst + per-fold long table)
4. Test metrics vs pyKT literature (toggleable `AGG = 'auc_mean' / 'auc_best'`)
5. Per-fold training curves + 5b. mastery spider + beta contributions
   (per dataset, fold-0 weights)
6. Throughput & peak memory
7. Conclusions + 7b. attribution case study (fold-0 weights)

## Workspace folder tree

```
.
├── .claude/
│   └── settings.local.json
├── .env
├── configs/
│   ├── ablations.yaml
│   ├── algebra05.yaml
│   ├── assist09.yaml
│   ├── assist12.yaml
│   ├── base.yaml
│   ├── bridge06.yaml
│   ├── eedi.yaml
│   ├── junyi.yaml
│   └── xes3g5m.yaml
├── data_cache/                   # generated, gitignored
├── datasets/                     # raw dataset files (gitignored)
├── dpa_kt/
│   ├── __init__.py
│   ├── config.py
│   ├── utils.py
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── attribution.py
│   │   ├── literature.py
│   │   └── visualize.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── canonical.py
│   │   ├── dataset.py
│   │   ├── kc_graph.py
│   │   ├── loaders/
│   │   │   ├── __init__.py
│   │   │   ├── assist09.py
│   │   │   ├── assist12.py
│   │   │   ├── eedi.py
│   │   │   ├── junyi.py
│   │   │   ├── kddcup.py
│   │   │   └── xes3g5m.py
│   │   └── sequences.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── distribution.py
│   │   ├── dpa_kt.py
│   │   ├── embeddings.py
│   │   ├── interaction_encoder.py
│   │   ├── mastery.py
│   │   ├── patterns.py
│   │   └── predictor.py
│   └── training/
│       ├── __init__.py
│       ├── checkpoint.py
│       ├── csv_logger.py
│       ├── metrics.py
│       └── trainer.py
├── dpa_kt_vs_pykt_baselines_report.md
├── Main model architecture.png
├── main.pdf
├── notebooks/
│   ├── DPA_KT_master.ipynb
│   ├── DPA_KT_200_epochs_ENG.ipynb
│   ├── DPA_KT_200_epochs_VN.ipynb
│   └── figures/                  # generated plots, gitignored
├── README.md
├── requirements.txt
├── runs-50-epochs/               # generated, gitignored (50-epoch original runs)
├── runs-200-epochs/              # generated, gitignored (200-epoch + 5-fold CV runs)
├── scripts/
│   ├── build_notebook.py
│   ├── build_notebook_200.py
│   ├── preprocess.py
│   ├── queue_run.sh
│   ├── run_all.sh
│   ├── setup_venv.sh
│   ├── train.py
│   └── train_200_cv.py
└── venv/                          # local virtualenv, gitignored
```

## Notes & caveats

- **Literature comparison is indicative, not head-to-head:** reported AUC
  (pyKT benchmark + original papers) uses different preprocessing/splits. Only
  *DPA-KT (ours)* rows are on our exact splits.
- Truncated-BPTT window is 5 steps: the forward pass still spans all 200 steps,
  but the recurrent Jacobian's gain requires a short gradient window for
  stability (see the inline note in `dpa_kt.py`).
- The KC graph is **estimated from data** (the paper assumes a given graph);
  a learnable graph is a natural extension.
