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
The repository/
├── README.md, PLAN.md              this file / progress checklist
├── main.pdf                        the conceptual DPA-KT paper
├── Main model architecture.png     reference architecture diagram (above)
├── requirements.txt
├── .env                            local secrets (gitignored, e.g. sudo pw)
│
├── dpa_kt/                         the package — all logic lives here
│   ├── config.py                   Config dataclass, YAML load/merge, ablation presets
│   ├── utils.py                    seeding, RNG save/restore, timers, param count
│   ├── data/
│   │   ├── canonical.py            canonical parquet schema + vocab/difficulty maps
│   │   ├── loaders/                one file per dataset family
│   │   │   ├── assist09.py  assist12.py  junyi.py
│   │   │   ├── eedi.py  xes3g5m.py  kddcup.py   (algebra05 + bridge06)
│   │   ├── sequences.py            pyKT-style sequencing -> memmap .npy
│   │   ├── kc_graph.py             prerequisite/neighbor KC graph estimation
│   │   └── dataset.py              memmap Dataset + DataLoader factory
│   ├── models/
│   │   ├── embeddings.py           embedding tables + masked_softmax
│   │   ├── interaction_encoder.py  Module 1 (Transformer branch + GRU branch)
│   │   ├── distribution.py         Module 2.1 (Gaussian projection + KL)
│   │   ├── patterns.py             Module 2.2-2.4 (4 pattern operators + readout)
│   │   ├── mastery.py              Module 3 (mastery state + gating)
│   │   ├── predictor.py            Module 4 (KC contributions + guess/slip)
│   │   └── dpa_kt.py                assembly: time loop, losses, attribution trace
│   ├── training/
│   │   ├── trainer.py              fit/evaluate/predict, AMP, early stopping, resume
│   │   ├── checkpoint.py           save/load model+optimizer+scheduler+epoch+RNG
│   │   ├── metrics.py              AUC/ACC/RMSE, param count
│   │   └── csv_logger.py           per-epoch CSV logging
│   └── analysis/
│       ├── literature.py           static literature AUC table + comparison frame
│       ├── attribution.py          per-student attribution trace extraction
│       └── visualize.py            every plotting function used by the notebook
│
├── configs/                        base.yaml + one YAML per dataset + ablations.yaml
├── scripts/
│   ├── setup_venv.sh                venv + deps + Jupyter kernel
│   ├── preprocess.py                CLI: loader -> canonical -> sequences -> KC graph
│   ├── train.py                     CLI: train / ablate / resume
│   ├── build_notebook.py            (re)generates the master notebook
│   └── run_all.sh                   trains every dataset + the 18-run ablation grid
├── notebooks/
│   └── DPA_KT_master.ipynb          the orchestration notebook (13 sections)
│
├── datasets/                        raw input data
│   ├── dataset ASSISTments/          2009-2010, 2012-13-with-affect
│   ├── dataset Junyi Academy/
│   ├── dataset XES3G5M (Google Drive)/
│   ├── dataset Eedi NeurIPS 2020/
│   └── dataset PSLC KDD Cup 2010/    algebra_2005_2006, bridge_to_algebra_2006_2007
│
├── data_cache/                      generated, gitignored
│   ├── canonical/  maps/            per-dataset parquet + vocab/difficulty JSON
│   ├── sequences/  graphs/          memmap .npy sequences + KC graph .npz
│   └── raw/assist12/                one-time ZIP extraction
│
└── runs/                            generated, gitignored
    └── <dataset>_<ablation>/         last.pt, best.pt, log.csv, test_metrics.json
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
