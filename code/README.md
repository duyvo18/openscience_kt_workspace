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
├── .gitignore
├── Main model architecture.png
├── README.md
├── dpa_kt_vs_pykt_baselines_report.md
├── main.pdf
├── requirements.txt
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
│   ├── canonical/
│   │   ├── algebra05.parquet
│   │   ├── assist09.parquet
│   │   ├── assist12.parquet
│   │   ├── bridge06.parquet
│   │   ├── eedi.parquet
│   │   ├── junyi.parquet
│   │   └── xes3g5m.parquet
│   ├── graphs/
│   │   ├── algebra05.npz
│   │   ├── assist09.npz
│   │   ├── assist12.npz
│   │   ├── bridge06.npz
│   │   ├── eedi.npz
│   │   ├── junyi.npz
│   │   └── xes3g5m.npz
│   ├── maps/
│   │   ├── algebra05.json
│   │   ├── assist09.json
│   │   ├── assist12.json
│   │   ├── bridge06.json
│   │   ├── eedi.json
│   │   ├── junyi.json
│   │   └── xes3g5m.json
│   ├── raw/
│   │   └── assist12/
│   └── sequences/
│       ├── algebra05/
│       ├── assist09/
│       ├── assist12/
│       ├── bridge06/
│       ├── eedi/
│       ├── junyi/
│       └── xes3g5m/
├── datasets/
│   ├── dataset ASSISTments/
│   │   ├── 2009-2010/
│   │   └── 2012-13-school-data-with-affect/
│   ├── dataset Eedi NeurIPS 2020/
│   │   ├── data_extracted/
│   │   └── starter_kit_extracted/
│   ├── dataset Junyi Academy/
│   │   └── Junyi/
│   ├── dataset PSLC KDD Cup 2010/
│   │   ├── algebra_2005_2006/
│   │   └── bridge_to_algebra_2006_2007/
│   └── dataset XES3G5M (Google Drive)/
│       └── XES3G5M/
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
├── notebooks/
│   ├── DPA_KT_master.ipynb
│   └── figures/
│       ├── algebra05_beta_student0_first.png
│       ├── algebra05_beta_student0_last.png
│       ├── algebra05_beta_student_last_first.png
│       ├── algebra05_beta_student_last_last.png
│       ├── algebra05_kc_graph.png
│       ├── algebra05_mastery_spider_student0.png
│       ├── algebra05_mastery_spider_student_last.png
│       ├── assist09_beta_student0_first.png
│       ├── assist09_beta_student0_last.png
│       ├── assist09_beta_student_last_first.png
│       ├── assist09_beta_student_last_last.png
│       ├── assist09_kc_graph.png
│       ├── assist09_mastery_spider_student0.png
│       ├── assist09_mastery_spider_student_last.png
│       ├── assist12_beta_student0_first.png
│       ├── assist12_beta_student0_last.png
│       ├── assist12_kc_graph.png
│       ├── assist12_mastery_spider_student0.png
│       ├── bridge06_beta_student0_first.png
│       ├── bridge06_beta_student0_last.png
│       ├── bridge06_beta_student_last_first.png
│       ├── bridge06_beta_student_last_last.png
│       ├── bridge06_kc_graph.png
│       ├── bridge06_mastery_spider_student0.png
│       ├── bridge06_mastery_spider_student_last.png
│       ├── composite_first_student.png
│       ├── composite_last_student.png
│       ├── eedi_kc_graph.png
│       ├── junyi_kc_graph.png
│       ├── xes3g5m_beta_student0_first.png
│       ├── xes3g5m_beta_student0_last.png
│       ├── xes3g5m_beta_student_last_first.png
│       ├── xes3g5m_beta_student_last_last.png
│       ├── xes3g5m_kc_graph.png
│       ├── xes3g5m_mastery_spider_student0.png
│       └── xes3g5m_mastery_spider_student_last.png
├── runs/                         # generated, gitignored
│   ├── algebra05_full/
│   ├── assist09_full/
│   ├── assist09_no_distributional/
│   ├── assist09_no_gs/
│   ├── assist09_no_mono/
│   ├── assist09_no_neighbor/
│   ├── assist09_no_prereq/
│   ├── assist09_no_samekc/
│   ├── assist09_no_temporal/
│   ├── assist09_single_branch/
│   ├── assist12_full/
│   ├── bridge06_full/
│   ├── eedi_full/
│   ├── exp_nodist/
│   ├── exp_reg/
│   ├── exp_reg2/
│   ├── exp_reg3/
│   ├── exp_small/
│   ├── exp_smallreg/
│   ├── exp_wd/
│   ├── junyi_full/
│   ├── xes3g5m_full/
│   ├── xes3g5m_no_distributional/
│   ├── xes3g5m_no_gs/
│   ├── xes3g5m_no_mono/
│   ├── xes3g5m_no_neighbor/
│   ├── xes3g5m_no_prereq/
│   ├── xes3g5m_no_samekc/
│   ├── xes3g5m_no_temporal/
│   └── xes3g5m_single_branch/
├── scripts/
│   ├── build_notebook.py
│   ├── preprocess.py
│   ├── queue_run.sh
│   ├── run_all.sh
│   ├── setup_venv.sh
│   └── train.py
└── venv/                          # local virtualenv, gitignored
    └── ...
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
