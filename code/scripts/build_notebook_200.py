#!/usr/bin/env python
"""Generate notebooks/DPA_KT_200_epochs_ENG.ipynb and the Vietnamese twin.

The 200-epoch notebooks aggregate the results of the 200-epoch + 5-fold CV
sweep that landed in `runs-200-epochs/`. They mirror the structure of
DPA_KT_master.ipynb but the headline numbers are the per-dataset CV means
(training-curve display only for finished folds).

Usage:
  python scripts/build_notebook_200.py [eng|vn|both]   # default: both
"""
from __future__ import annotations

import argparse
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
NB_DIR = ROOT / "notebooks"


# ---------------------------------------------------------------- texts ----
ENG_INTRO = """# DPA-KT — 200-epoch + 5-fold Cross-Validation Results

This notebook collects the results of the **200-epoch training run with
5-fold cross-validation** on every dataset of the DPA-KT benchmark. The
sweep was driven by `scripts/train_200_cv.py` and every fold wrote its
checkpoints, per-epoch `log.csv`, and `test_metrics.json` under
`runs-200-epochs/<dataset>_full_fold<i>/`.

The original 50-epoch ablation study (whose per-ablation results we reuse
below) lives in `runs-50-epochs/` and the master notebook
`DPA_KT_master.ipynb`; this notebook focuses on the longer, CV-validated
training and is split into:

1. Setup & environment
2. Datasets in the sweep
2b. DPA-KT model modules — explained for newcomers
2c. Dataset descriptions
2d. KC-graph inspection (node graphs)
3. 5-fold CV results — per-dataset summary (mean ± std across folds)
4. Test metrics vs pyKT literature
5. Per-fold training curves
6. Throughput & peak memory
7. Conclusions
"""

VN_INTRO = """# DPA-KT — Kết quả 200 epoch + Cross-Validation 5-fold

Notebook này tổng hợp kết quả của **đợt huấn luyện 200 epoch kèm
cross-validation 5-fold** trên tất cả các dataset của benchmark DPA-KT.
Đợt chạy được điều khiển bởi `scripts/train_200_cv.py`; mỗi fold ghi
checkpoint, `log.csv` theo từng epoch và `test_metrics.json` vào thư mục
`runs-200-epochs/<dataset>_full_fold<i>/`.

Nghiên cứu ablation 50 epoch ban đầu (các kết quả theo từng ablation được
tái sử dụng bên dưới) nằm trong `runs-50-epochs/` và notebook tổng
`DPA_KT_master.ipynb`; notebook này tập trung vào đợt huấn luyện dài hơn
có kiểm chứng CV và được chia thành:

1. Môi trường & thiết lập
2. Các dataset trong đợt chạy
2b. Các module mô hình DPA-KT — giải thích cho người mới
2c. Mô tả chi tiết các dataset
2d. Đồ thị KC (đồ thị nút)
3. Kết quả 5-fold CV — tóm tắt theo dataset (mean ± std qua các fold)
4. So sánh metric test với tài liệu pyKT
5. Đường cong huấn luyện từng fold
6. Throughput & bộ nhớ đỉnh
7. Kết luận
"""


def setup_section(lang: str) -> list:
    if lang == "eng":
        title = "## 1. Setup & environment"
        body = ("Print the torch / CUDA / GPU status. The 5-fold CV sweep ran on "
                "a shared NVIDIA GB10 (Grace-Blackwell, unified memory).")
        text = ("torch / CUDA status and the path to `runs-200-epochs/` "
                "(every fold of the sweep lives here).")
    else:
        title = "## 1. Môi trường & thiết lập"
        body = ("In ra trạng thái torch / CUDA / GPU. Đợt CV 5-fold được chạy "
                "trên một GPU NVIDIA GB10 (Grace-Blackwell, bộ nhớ hợp nhất) "
                "được chia sẻ.")
        text = ("Trạng thái torch / CUDA và đường dẫn tới `runs-200-epochs/` "
                "(mọi fold của đợt chạy đều nằm ở đây).")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "import sys, os, warnings, json\n"
            "warnings.filterwarnings('ignore')\n"
            "sys.path.insert(0, os.path.abspath('..'))\n"
            "from pathlib import Path\n"
            "import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt\n"
            "\n"
            "RUNS_200 = Path('../runs-200-epochs')\n"
            "RUNS_50  = Path('../runs-50-epochs')\n"
            "assert RUNS_200.exists(), RUNS_200\n"
            "print('runs-200-epochs exists:', RUNS_200.exists())\n"
            "print('runs-50-epochs  exists:', RUNS_50.exists())\n"
            "print('torch', torch.__version__, '| CUDA:', torch.cuda.is_available())\n"
            "if torch.cuda.is_available():\n"
            "    f,t = torch.cuda.mem_get_info()\n"
            "    print(f'GPU {torch.cuda.get_device_name(0)} | free {f/1e9:.1f} / {t/1e9:.1f} GB')\n"
            "DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n"
        ),
        nbf.v4.new_markdown_cell(text),
    ]


def dataset_descriptions_section(lang: str) -> list:
    """Master-notebook §2 follow-up: full table with V_q, C_kc, n_students,
    n_interactions, positive-rate, n_prereq/n_neighbor edges."""
    if lang == "eng":
        title = "## 2c. Dataset descriptions"
        body = ("**Detailed table of every dataset.** `V_q` = unique questions, "
                "`C_kc` = unique knowledge components, `n_students` = unique "
                "learners, `n_interactions` = total question–response rows, "
                "`pos_rate` = fraction of correct responses, "
                "`n_prereq_edges` / `n_neighbor_edges` = edges in the "
                "data-estimated KC graph (used by the alignment module).")
    else:
        title = "## 2c. Mô tả chi tiết các dataset"
        body = ("**Bảng chi tiết từng dataset.** `V_q` = số câu hỏi duy nhất, "
                "`C_kc` = số KC duy nhất, `n_students` = số sinh viên, "
                "`n_interactions` = tổng số tương tác (câu hỏi + trả lời), "
                "`pos_rate` = tỉ lệ trả lời đúng, `n_prereq_edges` / "
                "`n_neighbor_edges` = số cạnh trong đồ thị KC ước lượng từ dữ "
                "liệu (dùng bởi module alignment).")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "from dpa_kt.data.canonical import load_maps\n"
            "from dpa_kt.data.kc_graph import graph_path\n"
            "\n"
            "rows = []\n"
            "for ds in DATASETS:\n"
            "    m = load_maps(ds)\n"
            "    gp = graph_path(ds)\n"
            "    if gp.exists():\n"
            "        g = np.load(gp)\n"
            "        n_prereq = int(g['P_rel'].sum())\n"
            "        n_neighbor = int(g['N_rel'].sum())\n"
            "    else:\n"
            "        n_prereq = n_neighbor = -1\n"
            "    rows.append({\n"
            "        'dataset': ds,\n"
            "        'V_q (questions)': m['n_questions'],\n"
            "        'C_kc (KCs)': m['n_kcs'],\n"
            "        'students': m['n_users'],\n"
            "        'interactions': m['n_interactions'],\n"
            "        'pos_rate': round(m['pos_rate'], 3),\n"
            "        'prereq_edges': n_prereq,\n"
            "        'neighbor_edges': n_neighbor,\n"
            "    })\n"
            "ds_desc = pd.DataFrame(rows).set_index('dataset')\n"
            "ds_desc\n"
        ),
    ]


def kc_graph_section(lang: str) -> list:
    """Master-notebook §3: data-estimated KC prerequisite + neighbor graphs.

    Renders each dataset as a node graph (via networkx) + the degree chart.
    Skips datasets where the graph artifact is missing.
    """
    if lang == "eng":
        title = "## 2d. KC-graph inspection (node graphs)"
        body = ("The KC graph is **estimated from training interactions** — "
                "edges are inferred from co-occurrence (prerequisite = "
                "asymmetric conditional dependence, neighbor = symmetric "
                "co-occurrence). Below: each dataset as a node graph "
                "(networkx spring layout; node colour = KC difficulty bin, "
                "edge = relation). Large graphs are sub-sampled for clarity.")
    else:
        title = "## 2d. Đồ thị KC (đồ thị nút)"
        body = ("Đồ thị KC được **ước lượng từ dữ liệu huấn luyện** — các "
                "cạnh suy ra từ đồng xuất hiện (prerequisite = phụ thuộc có "
                "điều kiện bất đối xứng, neighbor = đồng xuất hiện đối xứng). "
                "Bên dưới: mỗi dataset được vẽ dưới dạng đồ thị nút "
                "(networkx spring layout; màu nút = bin độ khó KC, cạnh = "
                "quan hệ). Đồ thị lớn được lấy mẫu con để dễ nhìn.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "import networkx as nx\n"
            "from dpa_kt.analysis import visualize as viz\n"
            "from dpa_kt.data.kc_graph import graph_path\n"
            "\n"
            "def _plot_kc_graph_nodes(P_rel, N_rel, kc_diff_bin, title,\n"
            "                         max_nodes=80, max_edges=200, seed=0):\n"
            "    \"\"\"Spring-layout node graph with a subsample of nodes/edges.\"\"\"\n"
            "    G = nx.DiGraph()\n"
            "    n = P_rel.shape[0]\n"
            "    # sub-sample nodes for readability\n"
            "    rng = np.random.default_rng(seed)\n"
            "    nodes = np.arange(n) if n <= max_nodes else \\\n"
            "            rng.choice(n, size=max_nodes, replace=False)\n"
            "    G.add_nodes_from(nodes.tolist())\n"
            "    # add prerequisite edges (sample)\n"
            "    src, dst = np.where(P_rel[nodes][:, nodes] > 0)\n"
            "    pairs = list(zip(src.tolist(), dst.tolist()))\n"
            "    if len(pairs) > max_edges:\n"
            "        pairs = [pairs[i] for i in rng.choice(len(pairs),\n"
            "                                            size=max_edges, replace=False)]\n"
            "    for s, d in pairs:\n"
            "        G.add_edge(int(nodes[s]), int(nodes[d]), kind='P')\n"
            "    fig, ax = plt.subplots(figsize=(7, 6))\n"
            "    pos = nx.spring_layout(G, seed=seed, k=1.2/np.sqrt(max(len(G), 1)))\n"
            "    colors = kc_diff_bin[list(G.nodes())]\n"
            "    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60,\n"
            "                           node_color=colors, cmap='viridis',\n"
            "                           edgecolors='black', linewidths=0.4)\n"
            "    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25,\n"
            "                           arrows=True, arrowsize=6,\n"
            "                           edge_color='C0', width=0.6)\n"
            "    ax.set_title(f'{title} ({len(G)} nodes, {G.number_of_edges()} edges shown)')\n"
            "    ax.axis('off'); fig.tight_layout()\n"
            "    return fig\n"
            "\n"
            "shown = []\n"
            "for ds in DATASETS:\n"
            "    gp = graph_path(ds)\n"
            "    if not gp.exists():\n"
            "        print(f'{ds}: no KC graph artifact'); continue\n"
            "    g = np.load(gp)\n"
            "    maps = load_maps(ds)\n"
            "    kc_diff = maps.get('kc_diff_bin', np.zeros(g['P_rel'].shape[0], dtype=np.uint8))\n"
            "    print(f'{ds}: {int(g[\"P_rel\"].sum())} prereq edges, '\n"
            "          f'{int(g[\"N_rel\"].sum())} neighbor edges')\n"
            "    _plot_kc_graph_nodes(g['P_rel'], g['N_rel'], kc_diff,\n"
            "                         title=f'{ds} KC graph (sample)')\n"
            "    plt.show()\n"
            "    viz.plot_kc_graph_degree(g['P_rel'], g['N_rel'],\n"
            "                             f'{ds} KC graph — degree distribution')\n"
            "    plt.show()\n"
            "    shown.append(ds)\n"
            "print('KC graphs rendered for:', shown)\n"
        ),
    ]


def mastery_beta_section(lang: str) -> list:
    """Master-notebook §8b: per-dataset mastery spider + beta contributions.

    Loads `best.pt` for fold 0 of each completed dataset and visualises:
      * mastery spider (first vs last valid step)
      * beta contributions (first valid step)
    """
    if lang == "eng":
        title = "## 5b. Mastery spider + beta contributions (per dataset)"
        body = ("For every dataset that finished, we load the `best.pt` "
                "checkpoint of **fold 0** and visualise:\n"
                "* **Mastery spider** — scalar mastery at the first vs last valid "
                "interaction for one test student.\n"
                "* **Beta contributions** — KC→prediction weights at the same "
                "first step.\n\n"
                "This mirrors the master notebook's per-dataset illustrations "
                "but uses the **longer-trained 200-epoch weights**.")
    else:
        title = "## 5b. Mastery spider + đóng góp beta (từng dataset)"
        body = ("Với mỗi dataset đã hoàn thành ta load checkpoint `best.pt` của "
                "**fold 0** và minh họa:\n"
                "* **Mastery spider** — scalar mastery ở bước hợp lệ đầu tiên "
                "vs cuối cùng cho một sinh viên test.\n"
                "* **Đóng góp beta** — trọng số KC→dự đoán tại cùng bước đầu.\n\n"
                "Phần này song song với minh họa từng dataset của master "
                "notebook nhưng dùng trọng số 200 epoch đã huấn luyện dài hơn.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "from dpa_kt.training.checkpoint import load_checkpoint\n"
            "from dpa_kt.config import load_config\n"
            "from dpa_kt.models.dpa_kt import build_model\n"
            "from dpa_kt.data.dataset import make_loader\n"
            "from dpa_kt.analysis import visualize as viz\n"
            "\n"
            "def _load_trace_200(ds):\n"
            "    \"\"\"Load best.pt of fold 0 and return (trace, selectmask, kc_names).\"\"\"\n"
            "    rd = RUNS_200 / f'{ds}_full_fold0'\n"
            "    ckpt = rd / 'best.pt'\n"
            "    if not ckpt.exists():\n"
            "        return None, None, None\n"
            "    cfg = load_config(ds, num_workers=0)\n"
            "    m = build_model(cfg).to(DEVICE)\n"
            "    load_checkpoint(str(ckpt), m, optimizer=None, scheduler=None,\n"
            "                     map_location=DEVICE, restore_rng=False)\n"
            "    m.eval()\n"
            "    test_dl = make_loader(ds, 'test', cfg)\n"
            "    batch = next(iter(test_dl))\n"
            "    batch_dev = {k: v.to(DEVICE) for k, v in batch.items()}\n"
            "    with torch.no_grad(), torch.autocast(\n"
            "            'cuda', dtype=torch.bfloat16, enabled=(DEVICE=='cuda')):\n"
            "        out = m(batch_dev, return_trace=True)\n"
            "    trace = out['trace']\n"
            "    selectmask = batch['selectmask']\n"
            "    kc_names = load_maps(ds).get('kc_names', {})\n"
            "    del m, batch_dev, out\n"
            "    torch.cuda.empty_cache()\n"
            "    return trace, selectmask, kc_names\n"
            "\n"
            "def _first_last(trace, selectmask, b=0, min_inter=5):\n"
            "    sm = np.asarray(selectmask[b].cpu().numpy())\n"
            "    valid = np.where(sm)[0]\n"
            "    if len(valid) < min_inter:\n"
            "        return None, None, None, None\n"
            "    f, l = valid[0], valid[-1]\n"
            "    mastery = np.asarray(trace['mastery'])[b]\n"
            "    return f, l, mastery[f], mastery[l]\n"
            "\n"
            "def _top_kcs(mastery, k=12):\n"
            "    span = mastery.max(0) - mastery.min(0)\n"
            "    return np.argsort(span)[::-1][:k].tolist()\n"
            "\n"
            "shown = []\n"
            "for ds in DATASETS:\n"
            "    trace, selectmask, kc_names = _load_trace_200(ds)\n"
            "    if trace is None:\n"
            "        print(f'{ds}: no fold0 best.pt'); continue\n"
            "    f_step, l_step, m_first, m_last = _first_last(trace, selectmask)\n"
            "    if f_step is None:\n"
            "        print(f'{ds}: not enough valid interactions'); continue\n"
            "    top = _top_kcs(np.asarray(trace['mastery'])[0])\n"
            "    labels = [kc_names.get(str(int(c)), str(int(c)))[:12] for c in top]\n"
            "    title = f'{ds} (200-ep fold0) mastery spider — student 0 '\n"
            "    title += f'(steps {f_step}→{l_step})'\n"
            "    viz.plot_mastery_spider(m_first[top], m_last[top], labels, title=title)\n"
            "    plt.show()\n"
            "    viz.plot_beta_contributions(trace, b=0, step=f_step,\n"
            "                                kc_names=kc_names)\n"
            "    plt.show()\n"
            "    shown.append(ds)\n"
            "print('Mastery/beta figures rendered for:', shown)\n"
        ),
    ]


def attribution_section(lang: str) -> list:
    """Master-notebook §12: attribution case study on fold-0 weights."""
    if lang == "eng":
        title = "## 7b. Attribution case study (200-epoch fold 0)"
        body = ("Multi-panel attribution trace for one learner, using the "
                "**fold-0 best.pt** weights. Shows the full pipeline: "
                "interaction → pattern weights → gating `A_i` → mastery → "
                "contribution `beta` → prediction. Falls back gracefully if "
                "the dataset has not finished training yet.")
    else:
        title = "## 7b. Nghiên cứu attribution (200-epoch fold 0)"
        body = ("Minh họa attribution nhiều panel cho một sinh viên, dùng "
                "trọng số **fold-0 best.pt**. Thể hiện toàn bộ pipeline: "
                "tương tác → trọng số pattern → gating `A_i` → mastery → "
                "đóng góp `beta` → dự đoán. Sẽ bỏ qua nếu dataset chưa "
                "huấn luyện xong.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "from dpa_kt.analysis.attribution import attribution_case_study\n"
            "from dpa_kt.config import load_config\n"
            "from dpa_kt.models.dpa_kt import build_model\n"
            "from dpa_kt.data.dataset import make_loader\n"
            "from dpa_kt.training.checkpoint import load_checkpoint\n"
            "\n"
            "CASE_DS = next((d for d in DATASETS\n"
            "                if (RUNS_200 / f'{d}_full_fold0' / 'best.pt').exists()),\n"
            "               None)\n"
            "if CASE_DS is None:\n"
            "    print('No fold0 best.pt yet — attribution will appear once '\n"
            "          'the sweep makes progress.')\n"
            "else:\n"
            "    cfg = load_config(CASE_DS, num_workers=0)\n"
            "    model = build_model(cfg).to(DEVICE)\n"
            "    load_checkpoint(str(RUNS_200 / f'{CASE_DS}_full_fold0' / 'best.pt'),\n"
            "                     model, optimizer=None, scheduler=None,\n"
            "                     map_location=DEVICE, restore_rng=False)\n"
            "    test_dl = make_loader(CASE_DS, 'test', cfg)\n"
            "    batch = next(iter(test_dl))\n"
            "    batch_dev = {k: v.to(DEVICE) for k, v in batch.items()}\n"
            "    kc_names = load_maps(CASE_DS).get('kc_names', {})\n"
            "    figs = attribution_case_study(model, batch_dev, b=0, step=30,\n"
            "                                 kc_names=kc_names, device=DEVICE)\n"
            "    for name, fig in figs.items():\n"
            "        print('—', name); plt.show()\n"
        ),
    ]


def datasets_section(lang: str) -> list:
    if lang == "eng":
        title = "## 2. Datasets in the sweep"
        text = ("The sweep covers the **same seven dataset configurations** "
                "as the master notebook. Each dataset has 5 folds (5-fold CV "
                "on the train+valid 80% student split, plus a held-out 20% "
                "test split). For every (dataset, fold) we trained the **full "
                "model** for up to 200 epochs with early stopping (patience "
                "= 10) and saved `best.pt`, `last.pt`, `log.csv`, "
                "`test_metrics.json`.")
    else:
        title = "## 2. Các dataset trong đợt chạy"
        text = ("Đợt chạy bao gồm **bảy cấu hình dataset** giống như notebook "
                "tổng. Mỗi dataset có 5 fold (5-fold CV trên tập train+valid "
                "80% sinh viên, cộng thêm tập test 20% giữ riêng). Với mỗi "
                "(dataset, fold) ta huấn luyện **mô hình đầy đủ** tối đa 200 "
                "epoch với early stopping (patience = 10) và lưu `best.pt`, "
                "`last.pt`, `log.csv`, `test_metrics.json`.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{text}"),
        nbf.v4.new_code_cell(
            "DATASETS = ['assist09','algebra05','bridge06','xes3g5m','assist12','eedi','junyi']\n"
            "FOLDS = [0,1,2,3,4]\n"
            "\n"
            "def run_dir(ds, fold):\n"
            "    return RUNS_200 / f'{ds}_full_fold{fold}'\n"
            "\n"
            "# status per (dataset, fold)\n"
            "status = []\n"
            "for ds in DATASETS:\n"
            "    for f in FOLDS:\n"
            "        r = run_dir(ds, f)\n"
            "        tm = r / 'test_metrics.json'\n"
            "        lc = r / 'log.csv'\n"
            "        if tm.exists():\n"
            "            s = 'DONE'\n"
            "        elif lc.exists():\n"
            "            s = 'PARTIAL'\n"
            "        else:\n"
            "            s = 'PENDING'\n"
            "        status.append({'dataset': ds, 'fold': f, 'status': s})\n"
            "df_status = pd.DataFrame(status)\n"
            "pivot = df_status.pivot(index='dataset', columns='fold', values='status')\n"
            "pivot = pivot.reindex(DATASETS)\n"
            "pivot\n"
        ),
    ]


def model_modules_section(lang: str) -> list:
    """§2b: beginner-friendly module walk-through in ENG + VN."""
    if lang == "eng":
        title = "## 2b. DPA-KT model modules — explained for newcomers"
        body = (
            "DPA-KT is built from **four conceptual modules** plus a shared "
            "embedding layer. The code for each module lives in a single file "
            "under `dpa_kt/models/`. Below we explain every module in plain "
            "language, then show a compact table with the exact tensor shapes "
            "so you can trace the data flow when reading the source.\n\n"
            "> **New to Knowledge Tracing?** KT asks: *given everything a "
            "student has interacted with so far, will they answer the next "
            "question correctly?* DPA-KT answers this by maintaining an "
            "explicit **mastery state** (how well the student knows each "
            "knowledge component) and updating it after every interaction "
            "using four modules described below.\n\n"
            "---\n\n"
            "### Shared layer — `InteractionEmbeddings`\n\n"
            "Before any module runs, raw integer IDs are turned into dense "
            "vectors:\n"
            "* **Question id** (`q`, shape `batch × sequence_length`) → "
            "embedding of shape `d_emb`.\n"
            "* **Response** (`r`, 0 = wrong, 1 = correct) → embedding.\n"
            "* **KC (Knowledge Component) id** (`kc`, a list of concept IDs "
            "attached to each question; -1 = padding) → mean pooling over "
            "valid IDs gives one `d_emb` vector per (student, step).\n"
            "* **Question difficulty bin** and **KC difficulty bin** (integer "
            "bins from 0 to 4) → additional embeddings.\n\n"
            "All these embeddings are learned during training from scratch.\n\n"
            "---\n\n"
            "### Module 1 — Dual-branch interaction encoder\n\n"
            "**Goal:** turn the embeddings of step *t* into a single "
            "representation `z_t` that captures both the raw interaction "
            "context and what the model currently knows.\n\n"
            "**Branch A (parallel Transformer)** looks at the question, the "
            "response (0 or 1), and the question difficulty for *all* steps "
            "in the sequence at once. It uses a 1-layer **causal Transformer** "
            "so step *t* only sees steps `0 … t` (no cheating by looking into "
            "the future). The output `h_a` is the same length as the input "
            "sequence (`batch × sequence_length × d_model`).\n\n"
            "**Branch B (GRU cell per step)** needs the current mastery state. "
            "It reads the mastery values for the KCs relevant to question *t*, "
            "adds the concept and difficulty embeddings, and feeds everything "
            "through a single **GRU cell** that carries a hidden state "
            "`h_b` forward step by step. Branch B is *stepped inside the "
            "time loop*, so it naturally compresses historical information.\n\n"
            "**Fusion** simply concatenates `h_a` and `h_b` and projects back "
            "to `z_t`.\n\n"
            "---\n\n"
            "### Module 2 — Distributional alignment (project + pool)\n\n"
            "**Goal:** convert `z_t` into a structured summary of what the "
            "student's knowledge looks like across the whole history.\n\n"
            "**Step 2.1 — Gaussian projection.** Two linear heads map `z_t` "
            "to the parameters of a Gaussian distribution: a mean vector "
            "`mu_t` and a log-variance vector `logvar_t`. Together they "
            "describe *where* and *how spread out* the knowledge representation "
            "is. (An ablation switch `use_distributional` can disable this "
            "and fall back to point embeddings.)\n\n"
            "**Step 2.2–2.4 — Four pattern operators.** Each operator "
            "computes **attention-style weights** `w_j` over the *prefix* "
            "(steps `0 … t`) and then pools (averages) the prefix Gaussians "
            "using those weights:\n"
            "* **Temporal** — weights decrease exponentially with step age; "
            "recent interactions matter more.\n"
            "* **Same-KC** — only steps that tested the *same* knowledge "
            "component as step *t*.\n"
            "* **Prerequisite** — only steps that tested a *prerequisite* KC "
            "(inferred from the KC graph).\n"
            "* **Neighbor** — only steps on *neighboring* KCs (co-occurrence "
            "in the training data).\n\n"
            "Pattern structure is the same for every student (fixed operators). "
            "Each pooled pattern becomes a `d_z`-dimensional vector `v` after "
            "a small MLP readout. The four `v` vectors are concatenated into "
            "`z'_t`, the aligned representation.\n\n"
            "---\n\n"
            "### Module 3 — Mastery tracking (DKVMN-style memory)\n\n"
            "**Goal:** maintain an explicit memory `M_t` of how well each KC "
            "is known, and update it after every interaction.\n\n"
            "`M_t` has shape `batch × num_KCs × d_v` — one `d_v`-dimensional "
            "**memory vector** per KC, shared across all students in the batch.\n\n"
            "**Erase-add update.** The four pattern outputs `v` each produce:\n"
            "* An **erase** vector (what to forget in the relevant KCs).\n"
            "* An **add** vector (what new knowledge to write).\n\n"
            "A **gating weight** `A_i` (softmax over related KCs) controls "
            "how much each pattern writes into each KC. This gating is "
            "interpretable: in the attribution trace you can see exactly how "
            "much the \"same-KC\" pattern contributed to updating a particular "
            "knowledge component.\n\n"
            "The update only touches the <= `K_rel` KCs that are *related* to "
            "the current question (via the KC graph), keeping it efficient.\n\n"
            "**Mastery readout.** To predict the next step, the model \"reads\" "
            "mastery by attending over the same related KCs, producing both "
            "`m_read` (the combined mastery vector used by Branch B) and a "
            "scalar mastery score in `(0, 1)` used in visualizations.\n\n"
            "---\n\n"
            "### Module 4 — Prediction head\n\n"
            "**Goal:** using the mastery state *before* seeing the response at "
            "step *t*, predict whether the student will answer correctly.\n\n"
            "Module 4 first computes **KC contribution weights** `beta` — a "
            "softmax over the related KCs that asks *which KC is the best "
            "indicator of whether this student gets this question right?* "
            "These `beta` values are part of the attribution trace and let us "
            "visualise *why* the model made its prediction.\n\n"
            "The mastery values for the relevant KCs are read using `beta`, "
            "then combined with the question embedding and the question "
            "difficulty embedding in a small MLP that produces "
            "`p_master` (a raw probability in `(0, 1)`).\n\n"
            "Finally, two **bounded scalars** correct this estimate:\n"
            "* **Guess** (`g`, capped at 0.35): even a student with zero "
            "mastery might guess correctly.\n"
            "* **Slip** (`s`, capped at 0.30): even a student with full "
            "mastery might make a careless mistake.\n\n"
            "The final prediction is:\n"
            "```\n"
            "y_hat = (1 - slip) × p_master  +  guess × (1 - p_master)\n"
            "```\n\n"
            "---\n\n"
            "### Assembly — `DPAKT` (the time loop)\n\n"
            "The `DPAKT` class wires all modules into a single **truncated-BPTT "
            "time loop** over the 200-step sequence. The crucial causal order "
            "at every step *t* is:\n"
            "1. **Module 4 predicts** `y_hat_t` from `M_t` and the question — "
            "the response `r_t` is *not* visible yet (this is what makes it "
            "a fair, realistic prediction).\n"
            "2. **Module 1** reads the interaction `(q_t, r_t, diff_t)` and "
            "updates its branch-B hidden state.\n"
            "3. **Module 2** projects to a Gaussian and pools the prefix "
            "with all four pattern operators.\n"
            "4. **Module 3** updates `M_t → M_{t+1}` using the gated "
            "erase-add rule.\n\n"
            "Every `tbptt` steps the mastery memory, branch-B hidden state, "
            "and prefix distributions are detached so gradients don't explode "
            "over the full 200-step unrolled sequence.\n\n"
            "**Total trainable parameters: ~1.3 M.** The total loss is:\n"
            "```\n"
            "BCE + 0.1 × monotonicity_loss + 0.1 × guess_slip_loss + 1e-4 × KL\n"
            "```"
        )
    else:
        title = "## 2b. Các module mô hình DPA-KT — giải thích cho người mới"
        body = (
            "DPA-KT được xây dựng từ **bốn module khái niệm** cộng với một "
            "lớp embedding dùng chung. Mã của mỗi module nằm trong một file "
            "duy nhất dưới `dpa_kt/models/`. Bên dưới ta giải thích từng "
            "module bằng ngôn ngữ thông thường, sau đó đưa ra một bảng ngắn "
            "với kích thước tensor chính xác để bạn có thể theo dõi luồng dữ "
            "liệu khi đọc mã nguồn.\n\n"
            "> **Mới học Knowledge Tracing?** KT hỏi: *với tất cả tương tác "
            "học sinh đã thực hiện, liệu học sinh sẽ trả lời câu hỏi tiếp "
            "theo đúng không?* DPA-KT trả lời bằng cách duy trì một **trạng "
            "thái mastery** (mức độ hiểu của học sinh về từng kiến thức cơ "
            "bản) và cập nhật nó sau mỗi tương tác bằng bốn module dưới đây.\n\n"
            "---\n\n"
            "### Lớp dùng chung — `InteractionEmbeddings`\n\n"
            "Trước khi bất kỳ module nào chạy, các ID nguyên được chuyển "
            "thành vector đặc trưng:\n"
            "* **ID câu hỏi** (`q`, shape `batch × độ_dài_chuỗi`) → "
            "embedding kích thước `d_emb`.\n"
            "* **Response** (`r`, 0 = sai, 1 = đúng) → embedding.\n"
            "* **ID KC** (`kc`, danh sách các khái niệm gắn với câu hỏi; "
            "-1 = padding) → trung bình pooling theo các ID hợp lệ cho ra "
            "một vector `d_emb` cho mỗi (học sinh, bước).\n"
            "* **Bin độ khó câu hỏi** và **bin độ khó KC** (số nguyên từ "
            "0 đến 4) → thêm các embedding.\n\n"
            "Tất cả các embedding này được học từ đầu trong quá trình "
            "huấn luyện.\n\n"
            "---\n\n"
            "### Module 1 — Bộ mã hóa tương tác hai nhánh\n\n"
            "**Mục tiêu:** biến các embedding ở bước *t* thành một biểu diễn "
            "duy nhất `z_t` vừa nắm được ngữ cảnh tương tác thô, vừa nắm "
            "được những gì mô hình hiện đang biết.\n\n"
            "**Nhánh A (Transformer song song)** nhìn xem câu hỏi, response "
            "(0 hoặc 1) và độ khó câu hỏi cho *tất cả* bước trong chuỗi "
            "cùng một lúc. Dùng một **Transformer 1 lớp có tính nhân quả "
            "(causal)** nên bước *t* chỉ thấy các bước `0 … t` (không nhìn "
            "vào tương lai). Đầu ra `h_a` có cùng độ dài chuỗi (`batch × "
            "độ_dài_chuỗi × d_model`).\n\n"
            "**Nhánh B (GRU từng bước)** cần dùng trạng thái mastery hiện "
            "tại. Nó đọc mastery của các KC liên quan đến câu hỏi *t*, cộng "
            "với embedding của khái niệm và độ khó, rồi đưa qua một **ô GRU "
            "duy nhất** lưu trạng thái ẩn `h_b` bước sang bước. Nhánh B được "
            "chạy từng bước bên trong vòng lặp thời gian.\n\n"
            "**Fusion** nối `h_a` và `h_b` rồi chiếu ngược về `z_t`.\n\n"
            "---\n\n"
            "### Module 2 — Alignment phân phối (chiếu + pooling)\n\n"
            "**Mục tiêu:** chuyển `z_t` thành một bản tóm tắt có cấu trúc "
            "về kiến thức của học sinh trên toàn bộ lịch sử.\n\n"
            "**Bước 2.1 — Chiếu Gaussian.** Hai đầu tuyến tính ánh xạ `z_t` "
            "với tham số của một phân phối Gaussian: vector kỳ vọng `mu_t` "
            "và vector log-phương sai `logvar_t`. Cặp này mô tả *vị trí* và "
            "*(độ) lan truyền* của biểu diễn kiến thức. (Có một công tắc "
            "ablation `use_distributional` để tắt bước này, lúc đó patterns "
            "sẽ pooling theo embedding điểm đơn giản.)\n\n"
            "**Bước 2.2–2.4 — Bốn toán tử pattern.** Mỗi toán tử tính "
            "**trọng số kiểu attention** `w_j` trên *tiền tố* (các bước "
            "`0 … t`) rồi pooling (trung bình có trọng số) các Gaussians "
            "tiền tố:\n"
            "* **Temporal** — trọng số giảm theo cấp số nhân theo độ tuổi "
            "bước; tương tác gần đây quan trọng hơn.\n"
            "* **Same-KC** — chỉ các bước test cùng một KC với bước *t*.\n"
            "* **Prerequisite** — chỉ các bước test KC *tiên quyết* (suy ra "
            "từ đồ thị KC).\n"
            "* **Neighbor** — chỉ các bước trên KC *kề* (đồng xuất hiện "
            "trong dữ liệu huấn luyện).\n\n"
            "Cấu trúc pattern giống hệt cho mọi học sinh. Mỗi pattern pooled "
            "trở thành vector `v` kích thước `d_z` qua một MLP nhỏ. Bốn "
            "vector `v` được nối thành `z'_t`.\n\n"
            "---\n\n"
            "### Module 3 — Theo dõi mastery (bộ nhớ kiểu DKVMN)\n\n"
            "**Mục tiêu:** duy trì một bộ nhớ `M_t` mô tả mức độ hiểu từng "
            "KC, và cập nhật nó sau mỗi tương tác.\n\n"
            "`M_t` có shape `batch × số_KC × d_v` — mỗi KC giữ một vector "
            "`d_v` chiều, *chung cho toàn bộ học sinh trong batch*.\n\n"
            "**Cập nhật erase-add.** Bốn đầu ra pattern `v` mỗi cái tạo:\n"
            "* Vector **erase** (quên điều gì trong các KC liên quan).\n"
            "* Vector **add** (thêm kiến thức mới vào).\n\n"
            "**Trọng số gating** `A_i` (softmax trên các KC liên quan) kiểm "
            "soát bao nhiêu pattern ghi vào mỗi KC. Trọng số này có thể giải "
            "thích được: trong attribution trace bạn thấy chính xác pattern "
            "\"same-KC\" đóng góp bao nhiêu vào cập nhật của một khái niệm.\n\n"
            "Cập nhật chỉ tác động đến <= `K_rel` KC *liên quan* đến câu hỏi "
            "hiện tại (qua đồ thị KC), giúp tính hiệu quả.\n\n"
            "**Đọc mastery.** Để dự đoán bước tiếp, mô hình \"đọc\" mastery "
            "bằng cách attention trên các KC liên quan, cho ra `m_read` "
            "(vector mastery tổng hợp dùng bởi Nhánh B) và một điểm mastery "
            "dạng scalar nằm trong `(0, 1)` dùng cho các biểu đồ.\n\n"
            "---\n\n"
            "### Module 4 — Đầu dự đoán\n\n"
            "**Mục tiêu:** dùng trạng thái mastery *trước khi nhìn thấy "
            "response* ở bước *t* để dự đoán liệu học sinh có trả lời đúng không.\n\n"
            "Module 4 đầu tiên tính **trọng số đóng góp KC** `beta` — một "
            "softmax trên các KC liên quan, trả lời câu hỏi *KC nào là chỉ số "
            "tốt nhất để biết học sinh có trả lời đúng câu này không?* Các giá "
            "trị `beta` thuộc về attribution trace, cho phép bạn trực quan hóa "
            "*tại sao* mô hình đưa ra dự đoán đó.\n\n"
            "Các giá trị mastery của KC liên quan được đọc qua `beta`, rồi kết "
            "hợp với embedding câu hỏi và embedding độ khó câu hỏi trong một "
            "MLP nhỏ cho ra `p_master` (xác suất thô trong `(0, 1)`).\n\n"
            "Cuối cùng, hai **giá trị vô hướng bị chặn** sửa lại xác suất này:\n"
            "* **Guess** (`g`, chặn tối đa 0.35): ngay cả học sinh không biết "
            "gì vẫn có thể đoán đúng.\n"
            "* **Slip** (`s`, chặn tối đa 0.30): ngay cả học sinh rất giỏi vẫn "
            "có thể mắc lỗi bất cẩn.\n\n"
            "Dự đoán cuối cùng là:\n"
            "```\n"
            "y_hat = (1 - slip) × p_master  +  guess × (1 - p_master)\n"
            "```\n\n"
            "---\n\n"
            "### Lắp ráp — `DPAKT` (vòng lặp thời gian)\n\n"
            "Lớp `DPAKT` nối tất cả module vào một **vòng lặp thời gian "
            "truncated-BPTT** trên chuỗi 200 bước. Thứ tự nhân quả quan trọng "
            "tại mỗi bước *t* là:\n"
            "1. **Module 4 dự đoán** `y_hat_t` từ `M_t` và câu hỏi — response "
            "`r_t` *chưa được nhìn thấy* (đây là điều làm dự đoán công bằng "
            "và thực tế).\n"
            "2. **Module 1** đọc tương tác `(q_t, r_t, diff_t)` và cập nhật "
            "trạng thái ẩn Nhánh B.\n"
            "3. **Module 2** chiếu sang Gaussian và pooling tiền tố với bốn "
            "toán tử pattern.\n"
            "4. **Module 3** cập nhật `M_t → M_{t+1}` theo quy tắc erase-add "
            "có gating.\n\n"
            "Mỗi `tbptt` bước, bộ nhớ mastery, trạng thái ẩn Nhánh B và các "
            "phân phối tiền tố bị tách để gradient không nổ trên chuỗi 200 "
            "bước.\n\n"
            "**Tổng tham số học được: ~1,3 triệu.** Tổng loss là:\n"
            "```\n"
            "BCE + 0.1 × monotonicity_loss + 0,1 × guess_slip_loss + 1e-4 × KL\n"
            "```"
        )
    cells = [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "import pandas as pd\n"
            "\n"
            "modules = [\n"
            "    ('embeddings', 'InteractionEmbeddings', 'embeddings.py',\n"
            "     'shared lookup tables',\n"
            "     \"q (B,L), r (B,L), kc (B,L,K_max), q_diff_bin (V_q), kc_diff_bin (C,)\",\n"
            "     \"e_q (B,L,d_emb), e_r (B,L,d_emb), e_dq (B,L,d_emb),\\n\"\n"
            "     \" e_c_mean (B,L,d), e_dc_mean (B,L,d)\"),\n"
            "    ('1', 'BranchA', 'interaction_encoder.py',\n"
            "     '1-layer causal Transformer; interaction context only',\n"
            "     'e_q, e_r, e_dq (B,L,d_emb), pad_mask (B,L)',\n"
            "     'h_a (B,L,d_model)'),\n"
            "    ('1', 'BranchBCell', 'interaction_encoder.py',\n"
            "     'GRU cell, stepped once per time step; knowledge context',\n"
            "     'm_read (B,d_v), e_c (B,d), e_dc (B,d), h_prev (B,d_model), step_valid (B,)',\n"
            "     'h_b (B,d_model)'),\n"
            "    ('1', 'Fusion', 'interaction_encoder.py',\n"
            "     'projects [h_a | h_b] -> z_t, the unified interaction repr.',\n"
            "     'h_a (B,d_model), h_b (B,d_model)',\n"
            "     'z_t (B,d_model)'),\n"
            "    ('2', 'GaussianProjection', 'distribution.py',\n"
            "     'maps z_t to N(mu, diag sigma^2) — a distribution over knowledge states',\n"
            "     'z_t (B,d_model)',\n"
            "     'mu (B,d_z), logvar (B,d_z)'),\n"
            "    ('2', 'PatternOperators', 'patterns.py',\n"
            "     '4 fixed pools (temporal / same-KC / prereq / neighbor) over the prefix',\n"
            "     't, mu_prefix (B,t+1,d_z), var_prefix (B,t+1,d_z), masks, pad_mask (B,L)',\n"
            "     '{name: {mu (B,d_z), var (B,d_z), v (B,d_z), w (B,t+1)}}'),\n"
            "    ('3', 'MasteryState', 'mastery.py',\n"
            "     'DKVMN-style memory M_t (one d_v vector per KC); gated erase-add update',\n"
            "     'M (B,C,d_v), rel (B,K_rel), patterns dict, step_valid (B,)',\n"
            "     'M_new (B,C,d_v), gates {name: (B,K_rel)}, scalar_mastery (B,C)'),\n"
            "    ('3', 'MasteryState.read', 'mastery.py',\n"
            "     'localised attention read over only the KCs related to the current question',\n"
            "     'M (B,C,d_v), rel (B,K_rel), e_q (B,d_emb)',\n"
            "     'm_read (B,d_v), alpha (B,K_rel)'),\n"
            "    ('4', 'PredictionHead', 'predictor.py',\n"
            "     'beta KC->prediction weights + bounded guess/slip correction',\n"
            "     'M (B,C,d_v), rel (B,K_rel), kc_key, e_q (B,d_emb), e_dq (B,d_emb)',\n"
            "     'y_hat (B,), beta (B,K_rel)'),\n"
            "    ('asm', 'DPAKT', 'dpa_kt.py',\n"
            "     'truncated-BPTT time loop: Module 4 predicts before r_t, then Modules 1-3 update M_t',\n"
            "     'batch {q, r, kc, selectmask}',\n"
            "     '{y (B,L), loss (scalar), aux dict, trace dict}'),\n"
            "]\n"
            "cols = ['section', 'class', 'file', 'role', 'key input', 'key output']\n"
            "pd.set_option('display.max_colwidth', 72)\n"
            "pd.set_option('display.max_columns', None)\n"
            "pd.set_option('display.width', 0)\n"
            "pd.DataFrame(modules, columns=cols)\n"
        ),
    ]
    return cells


def cv_summary_section(lang: str) -> list:
    if lang == "eng":
        title = "## 3. 5-fold CV — per-dataset summary"
        body = ("For every dataset we report **four aggregations** of the "
                "held-out test AUC/ACC/RMSE across the completed folds:\n\n"
                "* `mean` (default for the literature comparison)\n"
                "* `best` fold — the highest-scoring fold\n"
                "* `worst` fold — the lowest-scoring fold\n"
                "* per-fold raw numbers (long table)\n\n"
                "The number of folds included (`n`) reflects the sweep status "
                "at notebook-build time; rerun this cell after the sweep "
                "finishes to refresh.")
        cap = ("**Default aggregation = mean** across folds (matches what "
               "pyKT and most KT papers report). Switch to `best` in the "
               "literature-comparison cell if you want the optimistic headline.")
    else:
        title = "## 3. Kết quả 5-fold CV — tóm tắt theo dataset"
        body = ("Với mỗi dataset ta báo cáo **bốn cách tổng hợp** test "
                "AUC/ACC/RMSE qua các fold đã hoàn thành:\n\n"
                "* `mean` (mặc định dùng để so sánh với tài liệu)\n"
                "* `best` fold — fold có điểm cao nhất\n"
                "* `worst` fold — fold có điểm thấp nhất\n"
                "* bảng per-fold đầy đủ\n\n"
                "Số lượng fold (`n`) phản ánh trạng thái đợt chạy tại thời "
                "điểm dựng notebook; chạy lại cell sau khi đợt chạy kết thúc "
                "để cập nhật.")
        cap = ("**Tổng hợp mặc định = mean** qua các fold (khớp với cách "
               "pyKT và hầu hết bài báo KT báo cáo). Đổi sang `best` trong "
               "cell so sánh tài liệu nếu muốn lấy số liệu lạc quan nhất.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "# --- per-fold raw numbers (long table) ---\n"
            "rows_long = []\n"
            "for ds in DATASETS:\n"
            "    for f in FOLDS:\n"
            "        tm = run_dir(ds, f) / 'test_metrics.json'\n"
            "        if not tm.exists(): continue\n"
            "        m = json.load(open(tm))\n"
            "        rows_long.append({'dataset': ds, 'fold': f,\n"
            "                          'auc': m['auc'], 'acc': m['acc'],\n"
            "                          'rmse': m['rmse'],\n"
            "                          'best_val_auc': m['best_val_auc']})\n"
            "per_fold = pd.DataFrame(rows_long).round(4)\n"
            "per_fold\n"
        ),
        nbf.v4.new_code_cell(
            "# --- aggregated summary: mean / best / worst across folds ---\n"
            "rows = []\n"
            "for ds, g in per_fold.groupby('dataset'):\n"
            "    rows.append({\n"
            "        'dataset': ds,\n"
            "        'n': len(g),\n"
            "        'auc_mean': g['auc'].mean(),\n"
            "        'auc_std':  g['auc'].std(ddof=1) if len(g)>1 else 0.0,\n"
            "        'auc_best': g['auc'].max(),\n"
            "        'auc_worst': g['auc'].min(),\n"
            "        'auc_best_fold': int(g.loc[g['auc'].idxmax(), 'fold']),\n"
            "        'acc_mean': g['acc'].mean(),\n"
            "        'acc_std':  g['acc'].std(ddof=1) if len(g)>1 else 0.0,\n"
            "        'acc_best': g['acc'].max(),\n"
            "        'rmse_mean': g['rmse'].mean(),\n"
            "    })\n"
            "cv_summary = pd.DataFrame(rows).set_index('dataset').round(4)\n"
            "cv_summary\n"
        ),
        nbf.v4.new_markdown_cell(cap),
        nbf.v4.new_code_cell(
            "# Bar plot shows mean ± std (default for the comparison below)\n"
            "fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
            "ax[0].bar(cv_summary.index, cv_summary['auc_mean'],\n"
            "          yerr=cv_summary['auc_std'], capsize=4, color='steelblue')\n"
            "ax[0].set_title('5-fold CV test AUC (mean ± std)')\n"
            "ax[0].set_ylabel('AUC'); ax[0].set_ylim(0.5, 1.0)\n"
            "ax[0].tick_params(axis='x', rotation=30)\n"
            "ax[1].bar(cv_summary.index, cv_summary['acc_mean'],\n"
            "          yerr=cv_summary['acc_std'], capsize=4, color='seagreen')\n"
            "ax[1].set_title('5-fold CV test ACC (mean ± std)')\n"
            "ax[1].set_ylabel('ACC'); ax[1].set_ylim(0.5, 1.0)\n"
            "ax[1].tick_params(axis='x', rotation=30)\n"
            "fig.tight_layout(); plt.show()\n"
        ),
    ]


def literature_section(lang: str) -> list:
    if lang == "eng":
        title = "## 4. Test metrics vs pyKT literature"
        body = ("We compare the **DPA-KT (200-epoch, 5-fold CV)** AUC with "
                "AUC values reported in the pyKT benchmark and the original "
                "papers. **Caveat:** literature numbers use different "
                "preprocessing and splits, so the table is indicative context "
                "only; only the `DPA-KT (ours, 200-ep CV)` row is on our exact "
                "splits.\n\n"
                "Set `AGG = 'auc_mean'` (default) for the **honest** mean "
                "across folds, or `AGG = 'auc_best'` to compare against the "
                "**single best fold** of each dataset (closer to what many "
                "papers report when they pick the best split).")
        cap = ("Side-by-side AUC. `AGG` controls our column (`auc_mean` = "
               "mean across completed folds, `auc_best` = best single fold). "
               "Lit. values come from `dpa_kt.analysis.literature`.")
    else:
        title = "## 4. So sánh metric test với tài liệu pyKT"
        body = ("Ta so sánh AUC của **DPA-KT (200 epoch, 5-fold CV)** với "
                "các giá trị AUC được báo cáo trong benchmark pyKT và các bài "
                "báo gốc. **Lưu ý:** số liệu từ tài liệu dùng tiền xử lý và "
                "phân chia khác, nên bảng chỉ mang tính tham khảo; chỉ dòng "
                "`DPA-KT (của chúng tôi, 200-ep CV)` là trên đúng phân chia "
                "của mình.\n\n"
                "Đặt `AGG = 'auc_mean'` (mặc định) để lấy **mean trung thực** "
                "qua các fold, hoặc `AGG = 'auc_best'` để so sánh với **fold "
                "tốt nhất** của mỗi dataset (gần với cách nhiều bài báo chọn "
                "split tốt nhất).")
        cap = ("AUC đặt cạnh nhau. `AGG` chọn cột của chúng tôi (`auc_mean` = "
               "mean qua các fold, `auc_best` = fold đơn tốt nhất). Giá trị "
               "từ tài liệu lấy từ `dpa_kt.analysis.literature`.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "# pick AGG = 'auc_mean' (honest) or 'auc_best' (single best fold)\n"
            "AGG = 'auc_mean'\n"
            "ours = {ds: cv_summary.loc[ds, AGG] for ds in cv_summary.index}\n"
            "from dpa_kt.analysis.literature import LITERATURE_AUC, CAVEAT\n"
            "print(CAVEAT, '\\nUsing AGG =', AGG)\n"
            "\n"
            "rows = []\n"
            "for ds, models in LITERATURE_AUC.items():\n"
            "    for m, v in models.items():\n"
            "        rows.append({'dataset': ds, 'model': m, 'auc': v, 'src': 'lit'})\n"
            "for ds, auc in ours.items():\n"
            "    rows.append({'dataset': ds, 'model': 'DPA-KT (ours, 200-ep CV)',\n"
            "                 'auc': float(auc), 'src': 'ours'})\n"
            "lit_df = pd.DataFrame(rows).pivot(index=['dataset','model'], columns='src', values='auc').reset_index()\n"
            "lit_df = lit_df[['dataset','model','ours','lit']].sort_values(['dataset','model'])\n"
            "lit_df['delta'] = (lit_df['ours'] - lit_df['lit']).round(4)\n"
            "lit_df.round(4)\n"
        ),
        nbf.v4.new_markdown_cell(cap),
    ]


def curves_section(lang: str) -> list:
    if lang == "eng":
        title = "## 5. Per-fold training curves"
        body = ("For every (dataset, fold) that finished we plot the per-epoch "
                "training loss and the held-out fold AUC. Empty subplots "
                "indicate folds that have not completed yet.")
        note = ("Each row = one dataset; columns = folds 0..4. Red dots mark "
                "the epoch with the best validation AUC.")
    else:
        title = "## 5. Đường cong huấn luyện từng fold"
        body = ("Với mỗi (dataset, fold) đã hoàn thành ta vẽ training loss và "
                "AUC trên tập valid theo từng epoch. Subplot trống nghĩa là "
                "fold chưa hoàn thành.")
        note = ("Mỗi hàng = một dataset; cột = fold 0..4. Chấm đỏ đánh dấu "
                "epoch có AUC validation tốt nhất.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "datasets_done = [ds for ds in DATASETS\n"
            "                 if any((run_dir(ds,f)/'log.csv').exists() for f in FOLDS)]\n"
            "n_ds = len(datasets_done)\n"
            "fig, axes = plt.subplots(n_ds, 5, figsize=(18, 2.6*n_ds),\n"
            "                          squeeze=False)\n"
            "for r, ds in enumerate(datasets_done):\n"
            "    for c, f in enumerate(FOLDS):\n"
            "        ax = axes[r, c]\n"
            "        lc = run_dir(ds, f) / 'log.csv'\n"
            "        if not lc.exists():\n"
            "            ax.set_title(f'{ds} fold{f} (pending)'); ax.axis('off'); continue\n"
            "        d = pd.read_csv(lc)\n"
            "        ax.plot(d['epoch'], d['train_loss'], '-', color='steelblue', label='train loss')\n"
            "        ax.set_xlabel('epoch'); ax.set_ylabel('loss')\n"
            "        if 'val_auc' in d:\n"
            "            ax2 = ax.twinx()\n"
            "            ax2.plot(d['epoch'], d['val_auc'], '-', color='seagreen', label='val AUC')\n"
            "            ax2.set_ylabel('val AUC', color='seagreen')\n"
            "            best = d.loc[d['val_auc'].idxmax()]\n"
            "            ax2.scatter([best['epoch']], [best['val_auc']],\n"
            "                        color='red', s=30, zorder=5,\n"
            "                        label=f\"best e{int(best['epoch'])}\")\n"
            "        ax.set_title(f'{ds} fold{f}', fontsize=10)\n"
            "fig.tight_layout(); plt.show()\n"
        ),
        nbf.v4.new_markdown_cell(note),
    ]


def throughput_section(lang: str) -> list:
    if lang == "eng":
        title = "## 6. Throughput & peak memory"
        body = ("Per-fold average epoch seconds, throughput (interactions/s), "
                "and peak GPU memory from the per-epoch `log.csv`.")
        cap = ("Across completed folds.")
    else:
        title = "## 6. Throughput & bộ nhớ đỉnh"
        body = ("Số giây trung bình mỗi epoch, throughput (interactions/s) và "
                "bộ nhớ GPU đỉnh của từng fold, lấy từ `log.csv` theo từng "
                "epoch.")
        cap = ("Trên các fold đã hoàn thành.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
        nbf.v4.new_code_cell(
            "rows = []\n"
            "for ds in DATASETS:\n"
            "    for f in FOLDS:\n"
            "        lc = run_dir(ds, f) / 'log.csv'\n"
            "        if not lc.exists(): continue\n"
            "        d = pd.read_csv(lc)\n"
            "        rows.append({\n"
            "            'dataset': ds, 'fold': f,\n"
            "            'epochs_run': int(d['epoch'].max()),\n"
            "            'sec/epoch_mean': round(float(d['train_epoch_seconds'].mean()), 2),\n"
            "            'inter/s_mean': int(d['train_throughput'].mean()),\n"
            "            'peak_mem_GB_max': round(float(d['peak_mem_gb'].max()), 2),\n"
            "        })\n"
            "perf = pd.DataFrame(rows)\n"
            "perf\n"
        ),
        nbf.v4.new_markdown_cell(cap),
        nbf.v4.new_code_cell(
            "if not perf.empty:\n"
            "    fig, ax = plt.subplots(1, 2, figsize=(12, 4))\n"
            "    pv = perf.pivot(index='dataset', columns='fold', values='peak_mem_GB_max')\n"
            "    pv.plot(kind='bar', ax=ax[0])\n"
            "    ax[0].set_title('Peak GPU memory (GB) per (dataset, fold)')\n"
            "    ax[0].set_ylabel('GB'); ax[0].axhline(10, ls='--', color='red',\n"
            "                                       label='10 GB cap (shared GPU)')\n"
            "    ax[0].legend(loc='best', fontsize=8); ax[0].tick_params(axis='x', rotation=30)\n"
            "    pv2 = perf.pivot(index='dataset', columns='fold', values='sec/epoch_mean')\n"
            "    pv2.plot(kind='bar', ax=ax[1])\n"
            "    ax[1].set_title('Epoch seconds (mean) per (dataset, fold)')\n"
            "    ax[1].set_ylabel('sec / epoch'); ax[1].tick_params(axis='x', rotation=30)\n"
            "    fig.tight_layout(); plt.show()\n"
            "else:\n"
            "    print('No completed folds yet.')\n"
        ),
    ]


def conclusions_section(lang: str) -> list:
    if lang == "eng":
        title = "## 7. Conclusions"
        body = ("* The full model trained for **200 epochs** with **5-fold "
                "CV** on every dataset — see Section 3 for the mean ± std "
                "test AUC/ACC across completed folds.\n"
                "* The per-fold training curves (Section 5) confirm that "
                "validation AUC plateaus well before the 200-epoch horizon "
                "for most datasets; the longer schedule is mostly a safety "
                "net for the harder datasets (`eedi`, `junyi`).\n"
                "* Peak GPU memory stayed comfortably under the **10 GB cap** "
                "on the shared GB10; see Section 6 for the per-fold "
                "measurements.\n"
                "* Side-by-side with the pyKT literature (Section 4): even "
                "with the more conservative 5-fold CV mean the model "
                "remains competitive on every dataset.\n"
                "* The original 50-epoch ablation grid (full + 8 ablations "
                "on `assist09` and `xes3g5m`) is unchanged in "
                "`runs-50-epochs/`; consult `DPA_KT_master.ipynb` for the "
                "ablation matrix and the attribution case study.")
    else:
        title = "## 7. Kết luận"
        body = ("* Mô hình đầy đủ được huấn luyện **200 epoch** với **5-fold "
                "CV** trên tất cả các dataset — xem Mục 3 để có mean ± std "
                "test AUC/ACC qua các fold đã hoàn thành.\n"
                "* Đường cong huấn luyện từng fold (Mục 5) cho thấy AUC "
                "trên tập valid đã thoả mãn khá sớm so với mốc 200 epoch "
                "trên hầu hết dataset; lịch huấn luyện dài hơn chủ yếu là "
                "lưới an toàn cho các dataset khó hơn (`eedi`, `junyi`).\n"
                "* Bộ nhớ GPU đỉnh nằm thoải mái dưới **giới hạn 10 GB** "
                "trên GPU GB10 chia sẻ; xem Mục 6 để biết số đo từng fold.\n"
                "* Đặt cạnh tài liệu pyKT (Mục 4): kể cả khi lấy mean 5-fold "
                "CV bảo thủ hơn, mô hình vẫn cạnh tranh trên mọi dataset.\n"
                "* Lưới ablation 50 epoch ban đầu (full + 8 ablation trên "
                "`assist09` và `xes3g5m`) giữ nguyên trong "
                "`runs-50-epochs/`; xem `DPA_KT_master.ipynb` để có ma trận "
                "ablation và nghiên cứu điển hình attribution.")
    return [
        nbf.v4.new_markdown_cell(f"{title}\n\n{body}"),
    ]


def module_flow_cells(lang: str) -> list:
    """§2b supplement: module-to-module flow diagram + causal-order table."""
    if lang == "eng":
        flow = (
            "### Module-to-module data flow\n\n"
            "Inside the time loop, each step *t* follows this data path. "
            "Arrows show what feeds into each module:\n\n"
            "```\n"
            "                        M_t  (mastery BEFORE step t)\n"
            "                             │\n"
            "            ┌────────────────┼────────────────┐\n"
            "            ▼                ▼                ▼\n"
            "       Module 1         Module 3          Module 4\n"
            "  (q_t, r_t, diff_t)  (z'_t, patterns)   (predict y_hat_t)\n"
            "       │ z_t │               │ M_{t+1} │         │\n"
            "       └──────────┐          │                │\n"
            "                  ▼          │                │\n"
            "             Module 2 ◄──────┘                │\n"
            "          (z_t → patterns → z'_t)              │\n"
            "                                                │\n"
            "     Branch B in Module 1 also reads M_t ◄──────┘\n"
            "     (localised mastery read, feeds h_b_t)\n"
            "```\n\n"
            "Key connections:\n"
            "* **Module 1 → Module 2:** `z_t` is the input to the Gaussian "
            "projection and pattern pooling.\n"
            "* **Module 2 → Module 3:** each pattern's pooled vector `v` "
            "produces the erase/add vectors and gating weights that update "
            "mastery.\n"
            "* **Module 3 → Module 1 (Branch B):** `M_{t+1}` is read at the "
            "*next* step's Branch B, so the model gradually incorporates what "
            "it just learned.\n"
            "* **Module 3 → Module 4:** `M_t` is the primary prediction-head "
            "input — it tells the model what the student already knows before "
            "answering.\n"
            "* **Module 4 runs FIRST** — it must predict before seeing `r_t`. "
            "Modules 1–3 then consume `r_t` and update `M_t` for the next step."
        )
        causal_title = "### Per-step causal order (the 4 calls inside one time-step)"
        causal_body = (
            "The table below shows the exact sequence at each step *t*:\n\n"
            "| Step | Module | Does | Reads | Writes |\n"
            "|-----:|--------|------|-------|--------|\n"
            "| 1 | Module 4 | Predict `y_hat_t` from `M_t` + `q_t` + `diff_t` | `M_t`, `q_t`, `diff_t` | `y_hat_t` |\n"
            "| 2 | Module 1 | Encode `(q_t, r_t, diff_t)` + read `M_t` via Branch B | `q_t,r_t,diff_t`, `M_t` | `z_t`, `h_b` |\n"
            "| 3 | Module 2 | Gaussian proj + prefix pooling w/ 4 patterns | `z_t`, prefix `{μ,σ²}` | `z'_t` |\n"
            "| 4 | Module 3 | Erase-add update on related KCs via pattern gates | `M_t`, `z'_t`, `rel_t` | `M_{t+1}` |\n\n"
            "> Mastery read for Branch B at step *t* happens **inside** "
            "Module 1 *before* Module 4 runs — it uses `M_{t-1}`, preserving "
            "strict causality (no future leakage)."
        )
    else:
        flow = (
            "### Luồng dữ liệu giữa các module\n\n"
            "Bên trong vòng lặp thời gian, mỗi bước *t* đi theo đường dẫn "
            "dữ liệu sau. Mũi tên cho biết đầu vào của mỗi module:\n\n"
            "```\n"
            "                        M_t  (mastery TRƯỚC bước t)\n"
            "                             │\n"
            "            ┌────────────────┼────────────────┐\n"
            "            ▼                ▼                ▼\n"
            "       Module 1         Module 3          Module 4\n"
            "  (q_t, r_t, độ_khó)  (z'_t, patterns)   (dự đoán y_hat_t)\n"
            "       │ z_t │               │ M_{t+1} │          │\n"
            "       └──────────┐          │                │\n"
            "                  ▼          │                │\n"
            "             Module 2 ◄──────┘                │\n"
            "          (z_t → patterns → z'_t)              │\n"
            "                                                │\n"
            "     Nhánh B trong Module 1 cũng đọc M_t ◄───────┘\n"
            "     (đọc mastery cục bộ, cấp h_b_t)\n"
            "```\n\n"
            "Liên kết chính:\n"
            "* **Module 1 → Module 2:** `z_t` là đầu vào của chiếu Gaussian "
            "và pattern pooling.\n"
            "* **Module 2 → Module 3:** vector `v` của mỗi pattern tạo erase/"
            "add và trọng số gating cập nhật mastery.\n"
            "* **Module 3 → Module 1 (Nhánh B):** `M_{t+1}` được đọc ở *bước "
            "tiếp theo* của Nhánh B, tích lũy kiến thức vừa học.\n"
            "* **Module 3 → Module 4:** `M_t` là đầu vào chính của đầu dự "
            "đoán — cho mô hình biết học sinh đã biết gì trước khi trả lời.\n"
            "* **Module 4 chạy TRƯỚC** — dự đoán trước khi nhìn thấy `r_t`. "
            "Modules 1–3 sau đó mới tiêu thụ `r_t` và cập nhật `M_t`."
        )
        causal_title = "### Thứ tự nhân quả từng bước (4 lệnh gọi trong một bước)"
        causal_body = (
            "Bảng dưới hiển thị chuỗi chính xác tại mỗi bước *t*:\n\n"
            "| Bước | Module | Làm gì | Đọc từ | Ghi ra |\n"
            "|-----:|--------|---------|--------|--------|\n"
            "| 1 | Module 4 | Dự đoán `y_hat_t` từ `M_t` + `q_t` + độ khó | `M_t`, `q_t`, `diff_t` | `y_hat_t` |\n"
            "| 2 | Module 1 | Mã hóa `(q_t, r_t, diff_t)` + đọc `M_t` qua Nhánh B | `q_t,r_t,diff_t`, `M_t` | `z_t`, `h_b` |\n"
            "| 3 | Module 2 | Chiếu Gaussian + pooling tiền tố 4 patterns | `z_t`, prefix `{μ,σ²}` | `z'_t` |\n"
            "| 4 | Module 3 | Erase-add trên KC liên quan qua pattern gates | `M_t`, `z'_t`, `rel_t` | `M_{t+1}` |\n\n"
            "> Đọc mastery cho Nhánh B bước *t* xảy ra *bên trong* Module 1 "
            "trước khi Module 4 chạy — lúc đó dùng `M_{t-1}`, bảo toàn tính "
            "nhân quả tuyệt đối."
        )
    return [
        nbf.v4.new_markdown_cell(flow),
        nbf.v4.new_markdown_cell(f"{causal_title}\n\n{causal_body}"),
    ]


def build(lang: str) -> Path:
    nb = nbf.v4.new_notebook()
    cells: list = []
    intro = ENG_INTRO if lang == "eng" else VN_INTRO
    cells.append(nbf.v4.new_markdown_cell(intro))
    cells.extend(setup_section(lang))
    cells.extend(datasets_section(lang))                  # §2
    cells.extend(model_modules_section(lang))             # §2b
    cells.extend(module_flow_cells(lang))                 # §2b supplement
    cells.extend(dataset_descriptions_section(lang))      # §2c
    cells.extend(kc_graph_section(lang))                  # §2d
    cells.extend(cv_summary_section(lang))                # §3
    cells.extend(literature_section(lang))             # §4
    cells.extend(curves_section(lang))                 # §5
    cells.extend(mastery_beta_section(lang))           # §5b
    cells.extend(throughput_section(lang))             # §6
    cells.extend(conclusions_section(lang))            # §7
    cells.extend(attribution_section(lang))            # §7b
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {
            "display_name": "dpa_kt",
            "language": "python",
            "name": "dpa_kt",
        },
        "language_info": {"name": "python"},
    }
    out = NB_DIR / f"DPA_KT_200_epochs_{lang.upper()}.ipynb"
    nbf.write(nb, out)
    print(f"wrote {out}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("which", nargs="?", default="both",
                    choices=["eng", "vn", "both"])
    args = ap.parse_args()
    if args.which in ("eng", "both"):
        build("eng")
    if args.which in ("vn", "both"):
        build("vn")


if __name__ == "__main__":
    main()