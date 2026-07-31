"""Matplotlib visualizations for every module + training curves.

All functions take an Axes (or create a Figure) and return the Figure, so the
notebook can compose them freely. No seaborn / external style deps.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Training curves (from runs-50-epochs/<run>/log.csv or runs-200-epochs/<run>/log.csv)
# ----------------------------------------------------------------------
def plot_learning_curves(csv_path, title: str = ""):
    import matplotlib.pyplot as plt
    import pandas as pd

    df = pd.read_csv(csv_path)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(df["epoch"], df["train_loss"], "-o", ms=3, label="train loss")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("loss"); ax[0].legend()
    ax[0].set_title(f"{title} loss")
    if "val_auc" in df:
        ax[1].plot(df["epoch"], df["val_auc"], "-o", ms=3, label="val AUC")
    if "val_acc" in df:
        ax[1].plot(df["epoch"], df["val_acc"], "-s", ms=3, label="val ACC")
    ax[1].set_xlabel("epoch"); ax[1].legend(); ax[1].set_title(f"{title} val metrics")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# KC graph
# ----------------------------------------------------------------------
def plot_kc_graph_degree(P_rel, N_rel, title: str = "KC graph degrees"):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    ax[0].hist(np.asarray(P_rel).sum(1), bins=20, color="#4C72B0")
    ax[0].set_title("prerequisite out-degree"); ax[0].set_xlabel("# edges")
    ax[1].hist(np.asarray(N_rel).sum(1), bins=20, color="#55A868")
    ax[1].set_title("neighbor degree"); ax[1].set_xlabel("# edges")
    fig.suptitle(title); fig.tight_layout()
    return fig


def _text_from_top(ax, x_frac, top_pad_pt, text, **kwargs):
    """Place text at a FIXED point-distance below a panel's top edge.

    Anchoring with a y-fraction (e.g. ``y=0.9``) makes the visual gap to
    the border depend on the panel's height -- retune a row height
    elsewhere (as happened here) and previously-fine text can end up
    colliding with the border above it, or with the figure's suptitle.
    Anchoring via ``xycoords="axes fraction"`` + a points-based
    ``xytext`` offset keeps the gap constant in physical size regardless
    of how tall/short the panel ends up being.
    """
    return ax.annotate(text, xy=(x_frac, 1.0), xycoords="axes fraction",
                       xytext=(0, -top_pad_pt), textcoords="offset points",
                       ha="left", va="top", **kwargs)


def _panel_frame(ax, facecolor="#fbfbfd", edgecolor="#a0a0a0", ticks_off=True):
    """Give a panel a visible light-grey border so it reads as a distinct
    "card" in a multi-panel figure, instead of floating with no boundary."""
    if ticks_off:
        ax.set_xticks([])
        ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(edgecolor)
        spine.set_linewidth(1.1)
    ax.set_facecolor(facecolor)


def plot_kc_graph_with_parents(
    P_rel,
    N_rel,
    prereq_ratio,
    kc_names: dict,
    title: str,
    max_nodes: int = 60,
    max_edges: int = 200,
    seed: int = 0,
    top_k_parents: int = 3,
    ax=None,
    legend_ax=None,
    key_ax=None,
    legend_header: str = None,
    key_items: list = None,
    explain_lines: list = None,
    explain_ax=None,
    legend_wrap_width: int = 100,
    explain_wrap_width: int = 60,
):
    """Node graph annotated with the strongest parents of each hub KC.

    Two edge styles are overlaid:
      * **Blue (directed arrows)** = prerequisite edges (estimated from
        first-encounter ordering, see kc_graph.py:115-125).
      * **Orange (undirected)** = neighbor edges (top-5 PMI per KC,
        kc_graph.py:102-113).

    The 5 highest-degree hub KCs are highlighted as larger red-outlined
    nodes and tagged with a circled digit (①..⑤, ordered by descending
    degree) placed radially outward from the graph centroid and
    pairwise-repelled so the tags never overlap each other, regardless of
    how tightly the hubs themselves are clustered. Full names are kept out
    of the graph and listed instead in the legend panel, keyed by the same
    digit. If ``legend_ax`` is provided, a legend listing each hub's
    top-``top_k_parents`` parents is drawn on it; otherwise a new figure
    with a legend strip below the graph is returned.

    Parameters
    ----------
    P_rel, N_rel : (C, C) {0,1} arrays as saved in the graph .npz
    prereq_ratio : (C, C) float in [0, 1] (same shape as kc_graph.py)
    kc_names : dict {str(kc_id): human_name} (or empty)
    title : figure title
    max_nodes, max_edges, seed : subsample controls for large graphs
    top_k_parents : number of parents shown per hub in the legend
    ax : optional matplotlib axes to draw the graph on
    legend_ax : optional axes to draw the parent legend on; if None a
        new figure is created with a legend strip below the graph
    key_ax : optional axes to draw the edge/node key on top of the
        graph. If None a new figure with the key panel is created.
    legend_header : optional custom header text for the legend (e.g. for
        localization). Defaults to English.
    key_items : optional list of Line2D handles for the top key legend.
        If None, default English labels are used.
    explain_lines : optional list of plain-language strings to render as
        a "How to read this figure" callout panel (rendered inside the
        figure itself, so the reader does not need to scroll back to the
        markdown). Each entry is one bullet line.
    explain_ax : optional matplotlib axes to draw the explanation panel
        on. If None and ``explain_lines`` is given, the function builds
        its own 4-row layout (key / graph / hub legend / explainer).
    legend_wrap_width : character width to wrap each hub/parent legend
        line at. The legend renders in two columns, so this should be
        roughly half of what a single full-width column would need.
    explain_wrap_width : character width to wrap each "how to read" bullet
        at. Also renders in two columns; same sizing logic as above.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    import networkx as nx

    P_rel = np.asarray(P_rel, dtype=np.uint8)
    N_rel = np.asarray(N_rel, dtype=np.uint8)
    prereq_ratio = np.asarray(prereq_ratio, dtype=float)
    P_rel = P_rel * (prereq_ratio > 0)

    n = P_rel.shape[0]
    rng = np.random.default_rng(seed)
    nodes = np.arange(n) if n <= max_nodes else rng.choice(
        n, size=max_nodes, replace=False)
    nodes = np.sort(nodes)

    G = nx.DiGraph()
    G.add_nodes_from(nodes.tolist())

    src, dst = np.where(P_rel[nodes][:, nodes] > 0)
    pairs = list(zip(src.tolist(), dst.tolist()))
    if len(pairs) > max_edges:
        idx = rng.choice(len(pairs), size=max_edges, replace=False)
        pairs = [pairs[i] for i in idx]
    for s, d in pairs:
        G.add_edge(int(nodes[s]), int(nodes[d]), kind="P")

    src, dst = np.where(N_rel[nodes][:, nodes] > 0)
    npairs = list(zip(src.tolist(), dst.tolist()))
    if len(npairs) > max_edges:
        idx = rng.choice(len(npairs), size=max_edges, replace=False)
        npairs = [npairs[i] for i in idx]
    for s, d in npairs:
        u, v = int(nodes[s]), int(nodes[d])
        if not G.has_edge(u, v):
            G.add_edge(u, v, kind="N")

    deg = G.degree()
    top_hubs = sorted(deg, key=lambda x: -x[1])[:5]
    hub_set = {n_ for n_, _ in top_hubs}

    own_fig = ax is None
    has_explain = bool(explain_lines)
    if own_fig:
        fig = plt.figure(figsize=(11, 13.0) if has_explain else (11, 10.5))
        if has_explain:
            gs = fig.add_gridspec(
                4, 1, height_ratios=[0.55, 4.0, 2.6, 2.4], hspace=0.32)
            key_ax = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            legend_ax = fig.add_subplot(gs[2])
            explain_ax = fig.add_subplot(gs[3])
        else:
            gs = fig.add_gridspec(
                3, 1, height_ratios=[0.55, 4.0, 2.8], hspace=0.30)
            key_ax = fig.add_subplot(gs[0])
            ax = fig.add_subplot(gs[1])
            legend_ax = fig.add_subplot(gs[2])
    elif key_ax is None:
        # caller provided ax + legend_ax but not key_ax -> build a separate
        # small figure for the key panel and return it as a 2nd Figure.
        # In practice the notebook passes a key_ax, so this branch is rare.
        key_fig = plt.figure(figsize=(11, 1.4))
        key_ax = key_fig.add_subplot(111)
    pos = nx.spring_layout(G, seed=seed, k=1.4 / np.sqrt(max(len(G), 1)),
                            iterations=80)

    node_colors = np.array([G.degree(n_) for n_ in G.nodes()])
    nodes_list = list(G.nodes())
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=[180 if n in hub_set else 40 for n in nodes_list],
        node_color=node_colors, cmap="viridis",
        edgecolors=["red" if n in hub_set else "black" for n in nodes_list],
        linewidths=[1.6 if n in hub_set else 0.4 for n in nodes_list],
    )

    p_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "P"]
    n_edges = [(u, v) for u, v, d in G.edges(data=True) if d["kind"] == "N"]
    nx.draw_networkx_edges(G, pos, edgelist=p_edges, ax=ax,
                           edge_color="#1f77b4", alpha=0.45,
                           arrows=True, arrowsize=5, width=0.6,
                           connectionstyle="arc3,rad=0.05")
    nx.draw_networkx_edges(G, pos, edgelist=n_edges, ax=ax,
                           edge_color="#ff7f0e", alpha=0.30,
                           arrows=False, width=0.5)

    # Hub labels are a single circled digit (\u2460..\u2468 = \u2460..\u2468) placed
    # radially outward from the graph's centroid, then pairwise-repelled so
    # the (few) markers never overlap regardless of how tightly the hubs
    # themselves are clustered. Full names + parents are listed in the
    # legend panel below, keyed by the same digit -- this keeps the graph
    # itself uncluttered even when hub names are long.
    import math
    from matplotlib import patheffects

    all_xy = np.array([pos[n_] for n_ in G.nodes()])
    extent = max(np.ptp(all_xy[:, 0]), np.ptp(all_xy[:, 1]), 1e-6)
    centroid = all_xy.mean(axis=0)

    # top_hubs is already sorted by descending degree; that order defines
    # the \u2460..\u2464 numbering used both on the graph and in the legend below.
    sorted_hubs = [n_ for n_, _ in top_hubs]
    marker_glyphs = [chr(0x2460 + i) for i in range(len(sorted_hubs))]

    label_pos = {}
    for i, n_ in enumerate(sorted_hubs):
        p = np.asarray(pos[n_], dtype=float)
        direction = p - centroid
        norm = np.linalg.norm(direction)
        if norm < 1e-6:
            ang = 2 * math.pi * i / max(len(sorted_hubs), 1)
            direction = np.array([math.cos(ang), math.sin(ang)])
        else:
            direction = direction / norm
        label_pos[n_] = p + direction * extent * 0.22

    # cheap pairwise repulsion (only a handful of points) so markers that
    # still land close together (e.g. all hubs clustered on one side of
    # the graph) get pushed apart to a minimum separation.
    min_sep = extent * 0.13
    keys = list(label_pos.keys())
    for _ in range(60):
        moved = False
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                pa, pb = label_pos[keys[a]], label_pos[keys[b]]
                d = pb - pa
                dist = np.linalg.norm(d)
                if dist < min_sep:
                    moved = True
                    push = (min_sep - dist) / 2
                    dirv = d / dist if dist > 1e-9 else np.array([1.0, 0.0])
                    label_pos[keys[a]] = pa - dirv * push
                    label_pos[keys[b]] = pb + dirv * push
        if not moved:
            break

    # Hard safety clamp: text glyphs have a real on-screen footprint that
    # isn't accounted for by data-coordinate autoscaling/margins, so a
    # label whose *point* is technically inside the view can still have
    # its rendered box poke past the panel border for edge-case layouts
    # (e.g. all hubs pushed to one side by the repulsion pass above).
    # Clamp well inside the ax.margins(0.16) view extent set below so
    # this can never happen, regardless of dataset/layout.
    x_lo, x_hi = all_xy[:, 0].min(), all_xy[:, 0].max()
    y_lo, y_hi = all_xy[:, 1].min(), all_xy[:, 1].max()
    clamp_pad = extent * 0.09  # stays inside the 0.16 margin added to ax below
    for n_ in label_pos:
        lx, ly = label_pos[n_]
        lx = min(max(lx, x_lo - clamp_pad), x_hi + clamp_pad)
        ly = min(max(ly, y_lo - clamp_pad), y_hi + clamp_pad)
        label_pos[n_] = np.array([lx, ly])

    for n_ in sorted_hubs:
        lx, ly = label_pos[n_]
        nx_, ny_ = pos[n_]
        ax.plot([nx_, lx], [ny_, ly], color="red", lw=0.7, alpha=0.6, zorder=1)
    for i, n_ in enumerate(sorted_hubs):
        lx, ly = label_pos[n_]
        txt = ax.annotate(
            marker_glyphs[i], xy=(lx, ly), xycoords="data",
            fontsize=19, fontweight="bold", ha="center", va="center",
            color="#d62728", zorder=4,
        )
        txt.set_path_effects([patheffects.withStroke(linewidth=3.0, foreground="white")])

    # Build legend lines: each hub (keyed by its \u2460..\u2464 marker) with its top-k parents.
    # Wrapped to a conservative character width (rather than left to matplotlib's
    # unreliable auto-wrap) so long parent names never run past the panel edge.
    import textwrap

    legend_lines = []
    for i, (hub, _) in enumerate(top_hubs):
        parents = np.argsort(-prereq_ratio[:, hub])[:top_k_parents + 1]
        parents = [int(p) for p in parents
                   if p != hub and prereq_ratio[p, hub] > 0][:top_k_parents]
        if parents:
            p_str = ", ".join(f"{_kc_label(p, kc_names, max_len=16)}"
                              f" (r={prereq_ratio[p, hub]:.2f})"
                              for p in parents)
        else:
            p_str = "(no parents passed threshold)"
        line = f"{marker_glyphs[i]} {_kc_label(hub, kc_names, max_len=16)} \u2190 {p_str}"
        legend_lines.append(textwrap.fill(line, width=legend_wrap_width, subsequent_indent="    "))

    _panel_frame(legend_ax)
    if legend_header is None:
        legend_header = ("\u2460..\u2465 = hub KC; names after \u2190 are their "
                         "top-3 estimated parents (prereq_ratio from "
                         "first-encounter ordering):")
    _text_from_top(legend_ax, 0.015, 14, legend_header,
                    fontsize=16, fontweight="bold", wrap=True)
    # Two columns so the (short) list of hubs uses the panel's full width
    # instead of a narrow strip down the left, and can run at a much
    # bigger font without needing extra panel height. Body is anchored a
    # fixed distance below the top too (not below the header specifically)
    # -- the header is one line at this font/width, so a slightly larger
    # fixed offset than the header's own leaves a constant gap between them.
    half = (len(legend_lines) + 1) // 2
    col_l, col_r = legend_lines[:half], legend_lines[half:]
    _text_from_top(legend_ax, 0.015, 46, "\n".join(col_l),
                    fontsize=16.5, linespacing=2.1)
    if col_r:
        _text_from_top(legend_ax, 0.52, 46, "\n".join(col_r),
                        fontsize=16.5, linespacing=2.1)

    # Edge / node key panel at the top
    if key_ax is not None:
        from matplotlib.lines import Line2D
        _panel_frame(key_ax)
        if key_items is None:
            key_items = [
                Line2D([0], [0], color="#1f77b4", lw=2.0, marker=">",
                       markersize=12, markevery=[-1],
                       label="Prereq edge (directed): KC i is a prerequisite of KC j \u2014 "
                             "students encounter i before j in \u2265 65% of histories (kc_graph.py:115-125)"),
                Line2D([0], [0], color="#ff7f0e", lw=2.0,
                       label="Neighbor edge (undirected): KCs that co-occur in the same problem "
                             "or in a 3-step window (top-5 PMI, kc_graph.py:102-113)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#9bbb59",
                       markeredgecolor="black", markersize=12, lw=0,
                       label="Regular KC node (size = degree, color = degree on viridis)"),
                Line2D([0], [0], marker="o", color="w", markerfacecolor="#9bbb59",
                       markeredgecolor="red", markersize=14, markeredgewidth=1.6, lw=0,
                       label="Hub KC \u2014 one of the 5 highest-degree KCs in the plotted subgraph; "
                             "its estimated parents are listed below"),
            ]
        key_ax.legend(handles=key_items, loc="center", ncol=2, frameon=False,
                      fontsize=13.5, handlelength=2.5, handleheight=1.4,
                      labelspacing=0.9, columnspacing=2.0)

    # extra margin so hub markers/leader-lines near the border never clip
    # against the panel frame added below (kept > clamp_pad above)
    ax.margins(0.16)
    ax.set_title(f"{title}  ({len(G)} nodes, {G.number_of_edges()} edges)",
                 fontsize=16)
    _panel_frame(ax, facecolor="white")
    _use_cjk_font(legend_ax)

    # ------------------------------------------------------------------
    # Optional "How to read this figure" callout panel
    # ------------------------------------------------------------------
    if has_explain and explain_ax is not None:
        _panel_frame(explain_ax)
        _text_from_top(explain_ax, 0.015, 16, "Cách đọc hình này (How to read):",
                       fontsize=17, fontweight="bold")
        # Two columns, same reasoning as the hub legend above: a fixed,
        # short list of bullets should use the panel's full width rather
        # than leaving the right half empty. matplotlib's wrap=True wraps
        # against the whole panel, not a half-width column, so each line
        # is pre-wrapped manually (same approach as the legend above).
        import textwrap as _tw
        wrapped = [_tw.fill(line, width=explain_wrap_width, subsequent_indent="    ")
                   for line in explain_lines]
        half = (len(wrapped) + 1) // 2
        col_l, col_r = wrapped[:half], wrapped[half:]
        # single blank line (not a full empty-line gap) between bullets --
        # enough visual separation without ballooning the column height
        _text_from_top(explain_ax, 0.015, 52, "\n".join(col_l),
                       fontsize=15, linespacing=1.75)
        if col_r:
            _text_from_top(explain_ax, 0.52, 52, "\n".join(col_r),
                           fontsize=15, linespacing=1.75)
    if own_fig:
        fig.suptitle(title, fontsize=11, y=0.995)
        fig.tight_layout()
        return fig
    return ax


def _kc_label(kc_id: int, kc_names: dict, max_len: int = 18) -> str:
    """Return a short human-readable label for a KC id.

    Truncates at the last whole word within ``max_len`` and appends an
    ellipsis, rather than hard-cutting mid-word (e.g. "Quadratic Formula"
    -> "Quadratic Formul", which reads as a typo rather than a
    deliberate truncation).
    """
    name = kc_names.get(str(int(kc_id)), "")
    if not name:
        return f"{int(kc_id)}"
    if len(name) <= max_len:
        return f"{int(kc_id)}:{name}"
    truncated = name[:max_len].rsplit(" ", 1)[0]
    if not truncated:  # single word longer than max_len: hard cut
        truncated = name[:max_len]
    return f"{int(kc_id)}:{truncated}…"


def _plot_kc_graph_raw(P_rel, N_rel, kc_diff_bin, title,
                       max_nodes=80, max_edges=200, seed=0, ax=None):
    """Plain (un-annotated) spring-layout node graph of a subsampled KC graph.

    This is the "raw" panel shown side-by-side with the hub-annotated one
    in :func:`plot_kc_graph_dashboard`. If ``ax`` is provided, draws into
    it and returns the Axes; otherwise creates its own Figure.
    """
    import matplotlib.pyplot as plt
    import networkx as nx

    G = nx.DiGraph()
    n = P_rel.shape[0]
    rng = np.random.default_rng(seed)
    nodes = np.arange(n) if n <= max_nodes else \
        rng.choice(n, size=max_nodes, replace=False)
    G.add_nodes_from(nodes.tolist())
    src_, dst = np.where(P_rel[nodes][:, nodes] > 0)
    pairs = list(zip(src_.tolist(), dst.tolist()))
    if len(pairs) > max_edges:
        pairs = [pairs[i] for i in rng.choice(len(pairs), size=max_edges, replace=False)]
    for s, d in pairs:
        G.add_edge(int(nodes[s]), int(nodes[d]), kind="P")

    own_fig = ax is None
    if own_fig:
        fig, ax = plt.subplots(figsize=(7, 6))
    pos = nx.spring_layout(G, seed=seed, k=1.2 / np.sqrt(max(len(G), 1)))
    colors = kc_diff_bin[list(G.nodes())]
    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=60,
                           node_color=colors, cmap="viridis",
                           edgecolors="black", linewidths=0.4)
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.25,
                           arrows=True, arrowsize=6,
                           edge_color="C0", width=0.6)
    ax.margins(0.16)
    ax.set_title(f"{title} ({len(G)} nodes, {G.number_of_edges()} edges shown)",
                 fontsize=16)
    _panel_frame(ax, facecolor="white")
    if own_fig:
        fig.tight_layout()
        return fig
    return ax


def _default_kc_key_items_vi():
    from matplotlib.lines import Line2D

    return [
        Line2D([0], [0], color="#1f77b4", lw=2.0, marker=">", markersize=12,
               markeredgecolor="#1f77b4", markeredgewidth=1.0,
               label="Cạnh tiên quyết (có hướng) - xem mục (2) bên dưới"),
        Line2D([0], [0], color="#ff7f0e", lw=2.0, marker="None",
               markeredgecolor="black", markeredgewidth=1.0,
               label="Cạnh láng giềng (vô hướng) - xem mục (3) bên dưới"),
        Line2D([0], [0], color="w", lw=0.0, marker="o", markersize=12,
               markeredgecolor="black", markeredgewidth=1.0,
               label="KC thường (kích thước & màu = bậc)"),
        Line2D([0], [0], color="w", lw=0.0, marker="o", markersize=14,
               markeredgecolor="red", markeredgewidth=1.6,
               label="Hub KC, đánh số ①–⑤ - xem mục (4) và bảng bên dưới"),
    ]


_DEFAULT_KC_EXPLAIN_LINES_VI = [
    "(1) Mỗi chấm tròn = 1 knowledge component (KC). Màu chấm = bậc trong đồ thị con (xanh = bậc thấp, vàng = bậc cao).",
    "(2) Mũi tên xanh dương ('tiên quyết', có hướng): học sinh gặp KC A trước KC B ở ≥65% lịch sử (và ≥10 học sinh gặp cả hai) ⇒ vẽ A → B.",
    "(3) Đường cam ('láng giềng', vô hướng): hai KC hay đi cùng nhau - cùng một câu hỏi, hoặc trong cửa sổ 3 bước liên tiếp của cùng học sinh - top-5 theo PMI.",
    "(4) Vòng viền đỏ, to hơn = 'hub' KC (1 trong 5 KC bậc cao nhất trong đồ thị con). Mỗi hub được đánh số ①–⑤ ngay trên hình; tên đầy đủ nằm trong bảng ngay bên dưới.",
    "(5) Bảng '①–⑤' bên dưới liệt kê top-3 'parent' ước tính của từng hub: KC đứng trước có prereq_ratio (tỉ lệ học sinh gặp trước) lớn nhất tới hub đó.",
    "(6) Không thấy cạnh nào nối tới một KC ⇒ KC đó bị cô lập trong mẫu vẽ (không đồng xuất hiện đủ với KC nào khác trong cửa sổ 3 bước).",
    "(7) Đây là ĐỒ THỊ CON được LẤY MẪU (tối đa 70 nút / 300 cạnh) để dễ nhìn - không phải toàn bộ KC/cạnh thật của dataset.",
    "(8) Không cạnh nào tới từ metadata có sẵn - tất cả suy ra từ chuỗi tương tác thô (uid, ts, kcs); xem ví dụ dữ liệu thô ở mục ngay phía trên.",
]


def plot_kc_graph_dashboard(
    ds: str,
    P_rel,
    N_rel,
    prereq_ratio,
    kc_names: dict,
    kc_diff_bin=None,
    max_nodes: int = 70,
    max_edges: int = 300,
    seed: int = 0,
    top_k_parents: int = 3,
    figsize=(14, 22),
    height_ratios=(1.3, 1.7, 1.7, 1.7, 7.5, 8.0),
    legend_wrap_width: int = 42,
    explain_wrap_width: int = 46,
    key_items: list = None,
    explain_lines: list = None,
):
    """Build the full raw-vs-annotated KC graph dashboard for one dataset.

    This is the single source of truth for the figure used in the "2e"
    section of the results notebook: a 6-row grid of
    key strip / raw graph + annotated graph (equal height) / hub-parent
    legend / "how to read" explainer, each panel bordered as a distinct
    card. Tune the layout here (in ``code/scripts/render_kc_graph_dashboard.py``
    for a fast look-at-the-PNG iteration loop) rather than editing the
    notebook cell directly, so both stay in sync.

    Parameters
    ----------
    ds : dataset name, used in titles.
    P_rel, N_rel, prereq_ratio : as saved in ``data_cache/graphs/<ds>.npz``.
    kc_names : dict {str(kc_id): human_name} (or empty).
    kc_diff_bin : optional per-KC difficulty bin array for the raw graph's
        node coloring; defaults to all-zeros (single color) if omitted.
    figsize, legend_wrap_width, explain_wrap_width : layout tuning knobs;
        see ``plot_kc_graph_with_parents`` for what the wrap widths do.
    key_items, explain_lines : override the Vietnamese defaults if needed.

    Returns the Figure.
    """
    import matplotlib.pyplot as plt

    if kc_diff_bin is None:
        kc_diff_bin = np.zeros(np.asarray(P_rel).shape[0], dtype=np.uint8)
    if key_items is None:
        key_items = _default_kc_key_items_vi()
    if explain_lines is None:
        explain_lines = _DEFAULT_KC_EXPLAIN_LINES_VI

    # Row heights are sized in "inches" (used as ratios, since figsize height
    # is fixed) from the actual wrapped-line counts at the wrap widths above,
    # not guessed -- see the derivation in
    # code/scripts/render_kc_graph_dashboard.py's module docstring. Key
    # insight: on-screen text size in a notebook is governed by
    # font_pt / figure_width_in (browsers shrink wide figures to fit the
    # output pane), so a narrower figsize is what actually makes text look
    # bigger -- bumping font size alone does not help once the browser
    # scales the whole image down to fit.
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 2, height_ratios=list(height_ratios),
                          width_ratios=[1, 1.08], hspace=0.25, wspace=0.10,
                          left=0.03, right=0.985, top=0.94, bottom=0.008)
    ax_key = fig.add_subplot(gs[0, :])
    ax_raw = fig.add_subplot(gs[1:4, 0])
    ax_right = fig.add_subplot(gs[1:4, 1])
    ax_leg = fig.add_subplot(gs[4, :])
    ax_explain = fig.add_subplot(gs[5, :])

    _plot_kc_graph_raw(P_rel, N_rel, kc_diff_bin,
                       title=f"{ds} - đồ thị thô (mẫu)",
                       max_nodes=max_nodes, max_edges=max_edges, seed=seed,
                       ax=ax_raw)
    plot_kc_graph_with_parents(
        P_rel, N_rel, prereq_ratio, kc_names,
        title=f"{ds} - đồ thị đã đánh số hub (①–⑤)",
        max_nodes=max_nodes, max_edges=max_edges, seed=seed,
        top_k_parents=top_k_parents,
        ax=ax_right, legend_ax=ax_leg, key_ax=ax_key,
        legend_header=("①–⑤ = 5 KC trung tâm (hub); tên sau dấu ← là top-3 "
                       "KC được ước tính là 'học trước' hub đó nhiều nhất:"),
        key_items=key_items, explain_lines=explain_lines, explain_ax=ax_explain,
        legend_wrap_width=legend_wrap_width, explain_wrap_width=explain_wrap_width,
    )
    fig.suptitle(f"{ds}: đồ thị KC - thô (trái) vs. đã đánh số hub + giải thích (phải)",
                fontsize=16, y=0.99)
    return fig


# ----------------------------------------------------------------------
# Module 1: embeddings
# ----------------------------------------------------------------------
def plot_embedding_scatter(emb: np.ndarray, title: str, color=None, max_pts=2000):
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    emb = np.asarray(emb)
    if len(emb) > max_pts:
        idx = np.random.default_rng(0).choice(len(emb), max_pts, replace=False)
        emb = emb[idx]
        color = None if color is None else np.asarray(color)[idx]
    xy = PCA(n_components=2).fit_transform(emb)
    fig, ax = plt.subplots(figsize=(5, 5))
    sc = ax.scatter(xy[:, 0], xy[:, 1], s=6, c=color, cmap="viridis", alpha=0.6)
    if color is not None:
        fig.colorbar(sc, ax=ax, label="difficulty bin")
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Module 2: distributional space with uncertainty
# ----------------------------------------------------------------------
def plot_distribution_space(mu: np.ndarray, var: np.ndarray, title: str, n=60):
    """2-D PCA of means with uncertainty ellipses (mean scalar std)."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.decomposition import PCA

    mu, var = np.asarray(mu), np.asarray(var)
    n = min(n, len(mu))
    idx = np.random.default_rng(0).choice(len(mu), n, replace=False)
    p = PCA(n_components=2).fit(mu)
    xy = p.transform(mu[idx])
    r = np.sqrt(var[idx].mean(1))  # scalar spread per point
    r = 0.5 * r / (r.mean() + 1e-9) * (xy.std() + 1e-9)
    fig, ax = plt.subplots(figsize=(6, 5))
    for (x, y), rr in zip(xy, r):
        ax.add_patch(Ellipse((x, y), 2 * rr, 2 * rr, fill=False,
                             edgecolor="#C44E52", alpha=0.5))
    ax.scatter(xy[:, 0], xy[:, 1], s=10, color="#4C72B0")
    ax.set_title(title); ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
    ax.autoscale_view()
    fig.tight_layout()
    return fig


def plot_pattern_weights(trace, b: int, step: int, names):
    """Heatmap-style bars of each pattern operator's pooling weights over the
    prefix, for one (batch item, step)."""
    import matplotlib.pyplot as plt

    w = trace["pattern_w"][step]  # (4, B, step+1)
    fig, axes = plt.subplots(len(names), 1, figsize=(9, 1.4 * len(names)), sharex=True)
    for i, (ax, nm) in enumerate(zip(axes, names)):
        wi = np.asarray(w[i, b])
        ax.bar(np.arange(len(wi)), wi, color="#4C72B0", width=1.0)
        ax.set_ylabel(nm, rotation=0, ha="right", va="center", fontsize=9)
        ax.set_yticks([])
    axes[-1].set_xlabel(f"interaction index (predicting step {step})")
    fig.suptitle(f"Pattern pooling weights - student {b}, step {step}")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Module 3: gating + mastery evolution
# ----------------------------------------------------------------------
def plot_gating_heatmap(trace, b: int, step: int, names, kc_names=None):
    """A_i[c] gating: pattern (rows) x related KC (cols) for one step."""
    import matplotlib.pyplot as plt

    gates = np.asarray(trace["gates"])[:, b, step, :]  # (4, K_rel)
    rel = np.asarray(trace["rel"])[b, step]            # (K_rel,)
    valid = rel >= 0
    gates = gates[:, valid]
    labels = [str(int(c)) for c in rel[valid]]
    if kc_names:
        labels = [kc_names.get(l, l)[:14] for l in labels]
    fig, ax = plt.subplots(figsize=(max(6, 0.5 * gates.shape[1]), 3))
    im = ax.imshow(gates, aspect="auto", cmap="magma")
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_title(f"Pattern→KC gating A_i - student {b}, step {step}")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    return fig


def plot_mastery_evolution(trace, b: int, kc_ids, q_seq, r_seq, kc_names=None,
                           max_steps=60):
    """Scalar mastery m_t[c] curves for selected KCs, with correct/incorrect
    markers at steps that involved each KC."""
    import matplotlib.pyplot as plt

    mastery = np.asarray(trace["mastery"])[b]  # (L, C)
    T = min(max_steps, mastery.shape[0])
    fig, ax = plt.subplots(figsize=(11, 4))
    for c in kc_ids:
        lbl = kc_names.get(str(c), str(c))[:18] if kc_names else str(c)
        ax.plot(range(T), mastery[:T, c], label=lbl, lw=1.5)
    ax.set_xlabel("interaction step"); ax.set_ylabel("scalar mastery")
    ax.set_title(f"Mastery evolution - student {b}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Module 4: prediction contributions
# ----------------------------------------------------------------------
def plot_beta_contributions(trace, b: int, step: int, kc_names=None):
    import matplotlib.pyplot as plt

    beta = np.asarray(trace["beta"])[b, step]  # (K_rel,)
    rel = np.asarray(trace["rel"])[b, step]
    valid = rel >= 0
    beta, rel = beta[valid], rel[valid]
    labels = [kc_names.get(str(int(c)), str(int(c)))[:16] if kc_names else str(int(c))
              for c in rel]
    order = np.argsort(beta)[::-1]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.bar(range(len(beta)), beta[order], color="#8172B3")
    ax.set_xticks(range(len(beta)))
    ax.set_xticklabels([labels[i] for i in order], rotation=90, fontsize=7)
    ax.set_ylabel("β (KC→prediction)")
    ax.set_title(f"Prediction KC contributions - student {b}, step {step}")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Mastery spider/radar chart
# ----------------------------------------------------------------------
def plot_mastery_spider(mastery_first: np.ndarray, mastery_last: np.ndarray,
                        kc_labels, title="Mastery spider"):
    """Radar chart comparing mastery at first vs last interaction.

    mastery_first / mastery_last: 1-D arrays of length n_kcs (or subset).
    kc_labels: list of strings for KC names/ids.
    """
    import matplotlib.pyplot as plt

    mastery_first = np.asarray(mastery_first)
    mastery_last = np.asarray(mastery_last)
    kc_labels = [str(l) for l in kc_labels]
    n = len(kc_labels)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    mf = np.concatenate([mastery_first, mastery_first[:1]])
    ml = np.concatenate([mastery_last, mastery_last[:1]])

    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(8, n * 0.6)), subplot_kw=dict(projection="polar"))
    ax.plot(angles, mf, "-o", ms=3, label="first interaction", color="#4C72B0")
    ax.fill(angles, mf, alpha=0.15, color="#4C72B0")
    ax.plot(angles, ml, "-o", ms=3, label="last interaction", color="#C44E52")
    ax.fill(angles, ml, alpha=0.15, color="#C44E52")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(kc_labels, fontsize=max(7, min(10, 90 // n)))
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))

    # Set CJK-capable font so non-ASCII labels (e.g. Chinese KC names in xes3g5m)
    # render correctly instead of as boxed glyphs.
    _use_cjk_font(ax)

    fig.tight_layout()
    return fig


def _use_cjk_font(ax) -> None:
    """Configure a CJK-capable font on the given Axes so non-Latin labels render.

    Falls back silently to whatever font is available if no CJK font is found.
    """
    import matplotlib.font_manager as fm
    from matplotlib import rcParams

    cjk_candidates = [
        "Noto Sans CJK SC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
        "WenQuanYi Micro Hei",
        "SimHei",
        "Microsoft YaHei",
        "PingFang SC",
        "Source Han Sans CN",
        "Source Han Sans SC",
        "AR PL UMing CN",
        "AR PL UKai CN",
        "Droid Sans Fallback",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = next((c for c in cjk_candidates if c in available), None)
    if chosen is None:
        return
    fp = fm.FontProperties(family=chosen)
    for lbl in ax.get_xticklabels():
        lbl.set_fontproperties(fp)
    for lbl in ax.get_yticklabels():
        lbl.set_fontproperties(fp)
    ax.set_title(ax.get_title(), fontproperties=fp)
    leg = ax.get_legend()
    if leg is not None:
        for txt in leg.get_texts():
            txt.set_fontproperties(fp)
    rcParams["font.sans-serif"] = [chosen, "DejaVu Sans"]
    rcParams["axes.unicode_minus"] = False


# ----------------------------------------------------------------------
# Ablation matrix
# ----------------------------------------------------------------------
def plot_ablation_matrix(results: dict, datasets, title="Ablation ΔAUC vs full"):
    """results: {dataset: {ablation_name: auc}}; renders ΔAUC heatmap."""
    import matplotlib.pyplot as plt

    ablations = sorted({a for d in results.values() for a in d})
    if "full" in ablations:
        ablations = ["full"] + [a for a in ablations if a != "full"]
    mat = np.full((len(ablations), len(datasets)), np.nan)
    for j, ds in enumerate(datasets):
        base = results.get(ds, {}).get("full", np.nan)
        for i, ab in enumerate(ablations):
            v = results.get(ds, {}).get(ab, np.nan)
            mat[i, j] = v - base if ab != "full" else v
    fig, ax = plt.subplots(figsize=(1.6 * len(datasets) + 2, 0.5 * len(ablations) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="RdBu", vmin=-0.05, vmax=0.05)
    ax.set_xticks(range(len(datasets))); ax.set_xticklabels(datasets, rotation=45, ha="right")
    ax.set_yticks(range(len(ablations))); ax.set_yticklabels(ablations)
    for i in range(len(ablations)):
        for j in range(len(datasets)):
            if not np.isnan(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:+.3f}" if ablations[i] != "full"
                        else f"{mat[i, j]:.3f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, label="ΔAUC (full=abs)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------
# Per-dataset composite (single big figure for one dataset)
# ----------------------------------------------------------------------
def plot_dataset_composite(
    ds: str,
    spider_first_ax: plt.Axes,
    spider_last_ax: plt.Axes,
    beta_first_ax: plt.Axes,
    beta_last_ax: plt.Axes,
    kc_graph_ax: plt.Axes,
    titles: dict,
    fig=None,
):
    """No-op aggregator; use ``plot_dataset_composite_grid`` to compose
    multiple composite figures into one large grid figure.
    """
    raise NotImplementedError(
        "Use plot_dataset_composite_grid to render multiple dataset composites."
    )


def plot_dataset_composite_grid(
    per_dataset_payloads: list[tuple[str, dict]],
    ncols: int = 1,
    suptitle: str = "Per-dataset mastery spider + beta contributions",
):
    """Compose one big figure with each dataset in its own row.

    Each row has 4 panels: KC-graph | mastery spider (first student) | beta
    bar (first step) | beta bar (last step). Saves a single PNG and returns
    the Figure so the notebook can embed it.

    per_dataset_payloads: list of (dataset_name, payload_dict).
        payload_dict keys:
          - "kc_graph": (P_rel, N_rel)
          - "spider_first": (m_first_vals, m_last_vals, kc_labels)
          - "beta_first": (beta_vals, kc_labels_sorted)
          - "beta_last":  (beta_vals, kc_labels_sorted)
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    n = len(per_dataset_payloads)
    rows = n
    fig = plt.figure(figsize=(24, 5 * rows))
    gs = GridSpec(rows, 4, figure=fig, hspace=0.55, wspace=0.35,
                  width_ratios=[1, 1.4, 1, 1])

    for r, (ds, payload) in enumerate(per_dataset_payloads):
        # Panel 1: KC graph degree distributions
        ax_kc = fig.add_subplot(gs[r, 0])
        if payload.get("kc_graph") is not None:
            P, N = payload["kc_graph"]
            ax_kc.hist(np.asarray(P).sum(1), bins=20, color="#4C72B0", label="prereq")
            ax_kc2 = ax_kc.twinx()
            ax_kc2.hist(np.asarray(N).sum(1), bins=20, color="#55A868",
                        alpha=0.6, label="neighbor")
            ax_kc.set_title(f"{ds}: KC degree distribution")
            ax_kc.set_xlabel("# edges")
            ax_kc.set_ylabel("prereq count", color="#4C72B0")
            ax_kc2.set_ylabel("neighbor count", color="#55A868")
            ax_kc.legend(loc="upper left", fontsize=8)
            ax_kc2.legend(loc="upper right", fontsize=8)
            _use_cjk_font(ax_kc); _use_cjk_font(ax_kc2)
        else:
            ax_kc.text(0.5, 0.5, f"{ds}\nKC graph N/A", ha="center", va="center")
            ax_kc.set_xticks([]); ax_kc.set_yticks([])

        # Panel 2: mastery spider (polar)
        ax_sp = fig.add_subplot(gs[r, 1], projection="polar")
        if payload.get("spider_first") is not None:
            mf, ml, labels = payload["spider_first"]
            mf = np.asarray(mf); ml = np.asarray(ml)
            k = len(labels)
            angles = np.linspace(0, 2 * np.pi, k, endpoint=False).tolist()
            angles += angles[:1]
            ax_sp.plot(angles, np.concatenate([mf, mf[:1]]), "-o", ms=3,
                       color="#4C72B0", label="first interaction")
            ax_sp.fill(angles, np.concatenate([mf, mf[:1]]), alpha=0.15, color="#4C72B0")
            ax_sp.plot(angles, np.concatenate([ml, ml[:1]]), "-o", ms=3,
                       color="#C44E52", label="last interaction")
            ax_sp.fill(angles, np.concatenate([ml, ml[:1]]), alpha=0.15, color="#C44E52")
            ax_sp.set_xticks(angles[:-1])
            ax_sp.set_xticklabels(labels, fontsize=max(7, min(10, 90 // k)))
            ax_sp.set_ylim(0, 1)
            ax_sp.set_title(f"{ds}: mastery spider (first student)")
            ax_sp.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
            _use_cjk_font(ax_sp)
        else:
            ax_sp.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax_sp.set_xticks([]); ax_sp.set_yticks([])

        # Panel 3: beta at first step
        ax_b1 = fig.add_subplot(gs[r, 2])
        if payload.get("beta_first") is not None:
            vals, kc_labels_sorted = payload["beta_first"]
            ax_b1.bar(range(len(vals)), vals, color="#8172B3")
            ax_b1.set_xticks(range(len(kc_labels_sorted)))
            ax_b1.set_xticklabels(kc_labels_sorted, rotation=90, fontsize=7)
            ax_b1.set_ylabel("β (KC→prediction)")
            ax_b1.set_title(f"{ds}: β at first step")
            _use_cjk_font(ax_b1)
        else:
            ax_b1.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax_b1.set_xticks([]); ax_b1.set_yticks([])

        # Panel 4: beta at last step
        ax_b2 = fig.add_subplot(gs[r, 3])
        if payload.get("beta_last") is not None:
            vals, kc_labels_sorted = payload["beta_last"]
            ax_b2.bar(range(len(vals)), vals, color="#C44E52")
            ax_b2.set_xticks(range(len(kc_labels_sorted)))
            ax_b2.set_xticklabels(kc_labels_sorted, rotation=90, fontsize=7)
            ax_b2.set_ylabel("β (KC→prediction)")
            ax_b2.set_title(f"{ds}: β at last step")
            _use_cjk_font(ax_b2)
        else:
            ax_b2.text(0.5, 0.5, "N/A", ha="center", va="center")
            ax_b2.set_xticks([]); ax_b2.set_yticks([])

    fig.suptitle(suptitle, fontsize=14)
    return fig
