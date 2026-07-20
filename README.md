# evolve-agent

A self-evolving single agent. It serves its own goals independently, with no manager or peers. In this repo, it delivers work, reviews itself, and revises itself.

## Core Idea

After each work cycle, ask: what could be better? Save reusable lessons into memory. Promote repeatedly verified lessons into principles. The agent updates its own rule file and keeps improving.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `AGENTS.md` | Rule core: identity, mission, principles, and self-evolution loop. |
| `KNOWLEDGE.md` | Knowledge index. |
| `knowledge/` | Current facts: `system.md` covers the system model, and `current-state.md` covers goals and progress. |
| `notes/` | Daily logs appended by date. Old entries are not edited after their day ends. |
| `code/` | DPA-KT project root (model, scripts, data, runs, notebooks). |
| `sources/` | Reference papers and proposal drafts. |
| `working/` | Active artifacts, diagrams, references, and paper source files. |

## Project File Tree

```
.
├── AGENTS.md
├── KNOWLEDGE.md
├── README.md
├── knowledge/
│   ├── current-state.md
│   └── system.md
├── notes/
│   └── .gitkeep
├── code/
│   ├── .claude/
│   │   └── settings.local.json
│   ├── .env
│   ├── .gitignore
│   ├── Main model architecture.png
│   ├── README.md
│   ├── dpa_kt_vs_pykt_baselines_report.md
│   ├── main.pdf
│   ├── requirements.txt
│   ├── configs/
│   │   ├── ablations.yaml
│   │   ├── algebra05.yaml
│   │   ├── assist09.yaml
│   │   ├── assist12.yaml
│   │   ├── base.yaml
│   │   ├── bridge06.yaml
│   │   ├── eedi.yaml
│   │   ├── junyi.yaml
│   │   └── xes3g5m.yaml
│   ├── data_cache/                   # generated, gitignored
│   │   ├── canonical/
│   │   │   ├── algebra05.parquet
│   │   │   ├── assist09.parquet
│   │   │   ├── assist12.parquet
│   │   │   ├── bridge06.parquet
│   │   │   ├── eedi.parquet
│   │   │   ├── junyi.parquet
│   │   │   └── xes3g5m.parquet
│   │   ├── graphs/
│   │   │   ├── algebra05.npz
│   │   │   ├── assist09.npz
│   │   │   ├── assist12.npz
│   │   │   ├── bridge06.npz
│   │   │   ├── eedi.npz
│   │   │   ├── junyi.npz
│   │   │   └── xes3g5m.npz
│   │   ├── maps/
│   │   │   ├── algebra05.json
│   │   │   ├── assist09.json
│   │   │   ├── assist12.json
│   │   │   ├── bridge06.json
│   │   │   ├── eedi.json
│   │   │   ├── junyi.json
│   │   │   └── xes3g5m.json
│   │   ├── raw/
│   │   │   └── assist12/
│   │   └── sequences/
│   │       ├── algebra05/
│   │       ├── assist09/
│   │       ├── assist12/
│   │       ├── bridge06/
│   │       ├── eedi/
│   │       ├── junyi/
│   │       └── xes3g5m/
│   ├── datasets/
│   │   ├── dataset ASSISTments/
│   │   │   ├── 2009-2010/
│   │   │   └── 2012-13-school-data-with-affect/
│   │   ├── dataset Eedi NeurIPS 2020/
│   │   │   ├── data_extracted/
│   │   │   └── starter_kit_extracted/
│   │   ├── dataset Junyi Academy/
│   │   │   └── Junyi/
│   │   ├── dataset PSLC KDD Cup 2010/
│   │   │   ├── algebra_2005_2006/
│   │   │   └── bridge_to_algebra_2006_2007/
│   │   └── dataset XES3G5M (Google Drive)/
│   │       └── XES3G5M/
│   ├── dpa_kt/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── utils.py
│   │   ├── analysis/
│   │   │   ├── __init__.py
│   │   │   ├── attribution.py
│   │   │   ├── literature.py
│   │   │   └── visualize.py
│   │   ├── data/
│   │   │   ├── __init__.py
│   │   │   ├── canonical.py
│   │   │   ├── dataset.py
│   │   │   ├── kc_graph.py
│   │   │   ├── loaders/
│   │   │   └── sequences.py
│   │   ├── models/
│   │   │   ├── __init__.py
│   │   │   ├── distribution.py
│   │   │   ├── dpa_kt.py
│   │   │   ├── embeddings.py
│   │   │   ├── interaction_encoder.py
│   │   │   ├── mastery.py
│   │   │   ├── patterns.py
│   │   │   └── predictor.py
│   │   └── training/
│   │       ├── __init__.py
│   │       ├── checkpoint.py
│   │       ├── csv_logger.py
│   │       ├── metrics.py
│   │       └── trainer.py
│   ├── notebooks/
│   │   ├── DPA_KT_master.ipynb
│   │   └── figures/
│   │       ├── algebra05_beta_student0_first.png
│   │       ├── algebra05_beta_student0_last.png
│   │       ├── algebra05_beta_student_last_first.png
│   │       ├── algebra05_beta_student_last_last.png
│   │       ├── algebra05_kc_graph.png
│   │       ├── algebra05_mastery_spider_student0.png
│   │       ├── algebra05_mastery_spider_student_last.png
│   │       ├── assist09_beta_student0_first.png
│   │       ├── assist09_beta_student0_last.png
│   │       ├── assist09_beta_student_last_first.png
│   │       ├── assist09_beta_student_last_last.png
│   │       ├── assist09_kc_graph.png
│   │       ├── assist09_mastery_spider_student0.png
│   │       ├── assist09_mastery_spider_student_last.png
│   │       ├── assist12_beta_student0_first.png
│   │       ├── assist12_beta_student0_last.png
│   │       ├── assist12_kc_graph.png
│   │       ├── assist12_mastery_spider_student0.png
│   │       ├── bridge06_beta_student0_first.png
│   │       ├── bridge06_beta_student0_last.png
│   │       ├── bridge06_beta_student_last_first.png
│   │       ├── bridge06_beta_student_last_last.png
│   │       ├── bridge06_kc_graph.png
│   │       ├── bridge06_mastery_spider_student0.png
│   │       ├── bridge06_mastery_spider_student_last.png
│   │       ├── composite_first_student.png
│   │       ├── composite_last_student.png
│   │       ├── eedi_kc_graph.png
│   │       ├── junyi_kc_graph.png
│   │       ├── xes3g5m_beta_student0_first.png
│   │       ├── xes3g5m_beta_student0_last.png
│   │       ├── xes3g5m_beta_student_last_first.png
│   │       ├── xes3g5m_beta_student_last_last.png
│   │       ├── xes3g5m_kc_graph.png
│   │       ├── xes3g5m_mastery_spider_student0.png
│   │       └── xes3g5m_mastery_spider_student_last.png
│   ├── runs/                         # generated, gitignored
│   │   ├── algebra05_full/
│   │   ├── assist09_full/
│   │   ├── assist09_no_distributional/
│   │   ├── assist09_no_gs/
│   │   ├── assist09_no_mono/
│   │   ├── assist09_no_neighbor/
│   │   ├── assist09_no_prereq/
│   │   ├── assist09_no_samekc/
│   │   ├── assist09_no_temporal/
│   │   ├── assist09_single_branch/
│   │   ├── assist12_full/
│   │   ├── bridge06_full/
│   │   ├── eedi_full/
│   │   ├── exp_nodist/
│   │   ├── exp_reg/
│   │   ├── exp_reg2/
│   │   ├── exp_reg3/
│   │   ├── exp_small/
│   │   ├── exp_smallreg/
│   │   ├── exp_wd/
│   │   ├── junyi_full/
│   │   ├── xes3g5m_full/
│   │   ├── xes3g5m_no_distributional/
│   │   ├── xes3g5m_no_gs/
│   │   ├── xes3g5m_no_mono/
│   │   ├── xes3g5m_no_neighbor/
│   │   ├── xes3g5m_no_prereq/
│   │   ├── xes3g5m_no_samekc/
│   │   ├── xes3g5m_no_temporal/
│   │   └── xes3g5m_single_branch/
│   ├── scripts/
│   │   ├── build_notebook.py
│   │   ├── preprocess.py
│   │   ├── queue_run.sh
│   │   ├── run_all.sh
│   │   ├── setup_venv.sh
│   │   └── train.py
│   └── venv/                          # local virtualenv, gitignored
│       └── ...
├── .opencode/
│   └── skills/
│       ├── academic-plotting/
│       ├── archify/
│       ├── ml-paper-writing/
│       ├── presenting-conference-talks/
│       └── systems-paper-writing/
├── .openscience/
│   ├── provenance.jsonl
│   └── runs.jsonl
├── sources/
│   ├── CMDKT.pdf
│   ├── CMKT.pdf
│   ├── FINER.pdf
│   ├── GKT.pdf
│   ├── KeenKT.pdf
│   ├── MCKT.pdf
│   ├── NSKT.pdf
│   ├── PLKT.pdf
│   ├── PSIKT.pdf
│   ├── RAGKT.pdf
│   ├── SSKT.pdf
│   ├── TBKT.pdf
│   ├── UKT.pdf
│   ├── proposal_final.md
│   ├── proposal_final.pdf
│   ├── proposal_v0.1.pdf
│   ├── proposal_v0.2.txt
│   └── proposal_v0.3.txt
└── working/
    ├── artifacts/
    │   ├── outline_short_paper.md
    │   ├── paper_plan.md.bk
    │   ├── proposal_final.md
    │   ├── review_KT_defer.md
    │   └── review_KT_in_scope.md
    ├── diagrams/
    │   ├── framework-overview.architecture.json
    │   ├── framework-overview.html
    │   ├── src.pdf
    │   └── src.tex
    ├── ref/
    │   ├── CMDKT.pdf
    │   ├── CMKT.pdf
    │   ├── ECD.pdf
    │   ├── Emerging_2022_2025.pdf
    │   ├── FINER.pdf
    │   ├── formative_assessment.pdf
    │   ├── GKT.pdf
    │   ├── KeenKT.pdf
    │   ├── MCKT.pdf
    │   ├── NSKT.pdf
    │   ├── PLKT.pdf
    │   ├── PSIKT.pdf
    │   ├── RAGKT.pdf
    │   ├── SSKT.pdf
    │   ├── TBKT.pdf
    │   ├── UKT.pdf
    │   └── paper_summary.pdf
    ├── sample/
    │   ├── 2004.07606v1.pdf
    │   ├── CEU-KT_Conference_Paper_ENG.docx
    │   └── foqua_kt_model_design.pdf
    └── src/
        ├── designspace.png
        ├── framework.png
        ├── main.tex
        └── references.bib
```

## Startup Order

1. Read `AGENTS.md`.
2. Read `KNOWLEDGE.md`.
3. Read the latest 2-3 files in `notes/`.
4. Check the goal, worktree, code, data, and logs.

## Memory Rules

- `knowledge/` stores current facts only; update it when facts change.
- `notes/` stores daily logs; append during the day and do not edit old entries.
- Principle changes go directly into `AGENTS.md`.
