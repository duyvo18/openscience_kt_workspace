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
    fig.suptitle(f"Pattern pooling weights — student {b}, step {step}")
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
    ax.set_title(f"Pattern→KC gating A_i — student {b}, step {step}")
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
    ax.set_title(f"Mastery evolution — student {b}")
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
    ax.set_title(f"Prediction KC contributions — student {b}, step {step}")
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
