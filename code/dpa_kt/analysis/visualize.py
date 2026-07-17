"""Matplotlib visualizations for every module + training curves.

All functions take an Axes (or create a Figure) and return the Figure, so the
notebook can compose them freely. No seaborn / external style deps.
"""
from __future__ import annotations

import numpy as np


# ----------------------------------------------------------------------
# Training curves (from runs/<run>/log.csv)
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
