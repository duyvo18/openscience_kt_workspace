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
GPU may be shared; training peaks well under 8 GB. Triton kernel compilation
needs the Python dev headers (`python3.12-dev`).

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
python scripts/train.py --dataset assist09 --resume runs/assist09_full/last.pt
```

Checkpoints (`best.pt`, `last.pt`), a per-epoch `log.csv`, and `test_metrics.json`
are written to `runs/<dataset>_<ablation>/`. The trainer uses AdamW, bf16 AMP,
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
│   └── figures/                  # generated plots, gitignored
├── README.md
├── requirements.txt
├── runs/                         # generated, gitignored
├── scripts/
│   ├── build_notebook.py
│   ├── preprocess.py
│   ├── queue_run.sh
│   ├── run_all.sh
│   ├── setup_venv.sh
│   └── train.py
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
