"""Extract and render the intrinsic attribution trace for one student.

The trace threads: interactions -> pattern pooling weights -> pattern-to-KC
gating A_i -> mastery state -> KC-to-prediction contribution beta -> prediction.
This module pulls a single student's trace out of a model forward pass and
assembles the end-to-end case-study figure the paper describes.
"""
from __future__ import annotations

import numpy as np
import torch


def trace_one_student(model, batch, b: int = 0, device="cuda") -> dict:
    """Run a single-sequence forward with return_trace and slice item b."""
    model.eval()
    sub = {k: v[b : b + 1].to(device) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                          enabled=(device == "cuda")):
        out = model(sub, return_trace=True)
    return {
        "y": out["y"][0].float().cpu().numpy(),
        "q": sub["q"][0].cpu().numpy(),
        "r": sub["r"][0].cpu().numpy(),
        "kc": sub["kc"][0].cpu().numpy(),
        "trace": out["trace"],
    }


def most_active_kcs(trace, b: int, top: int = 6) -> list[int]:
    """KCs whose scalar mastery moves the most across the sequence."""
    mastery = np.asarray(trace["mastery"])[b]  # (L, C)
    span = mastery.max(0) - mastery.min(0)
    return np.argsort(span)[::-1][:top].tolist()


def render_panel(panel: str, model, batch, b: int, step: int,
                 kc_names=None, device: str = "cuda") -> "matplotlib.figure.Figure":
    """Render exactly one panel of the attribution case study.

    Used by the notebook to display each panel as its own figure, so an
    explanation cell can sit directly under it.

    panel in {"pattern_weights", "gating", "mastery", "beta", "prediction"}.
    Returns the matplotlib Figure.
    """
    import matplotlib.pyplot as plt
    from . import visualize as viz

    model.eval()
    sub = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                          enabled=(device == "cuda")):
        out = model(sub, return_trace=True)
    trace = out["trace"]
    names = trace["pattern_names"]

    if panel == "pattern_weights":
        return viz.plot_pattern_weights(trace, b, step, names)
    if panel == "gating":
        return viz.plot_gating_heatmap(trace, b, step, names, kc_names)
    if panel == "mastery":
        kc_ids = most_active_kcs(trace, b)
        return viz.plot_mastery_evolution(
            trace, b, kc_ids, sub["q"][b].cpu().numpy(),
            sub["r"][b].cpu().numpy(), kc_names)
    if panel == "beta":
        return viz.plot_beta_contributions(trace, b, step, kc_names)
    if panel == "prediction":
        y = out["y"][b].float().cpu().numpy()
        r = sub["r"][b].cpu().numpy()
        fig, ax = plt.subplots(figsize=(11, 3))
        T = min(60, len(y))
        ax.plot(range(T), y[:T], "-o", ms=3, label="predicted P(correct)")
        ax.scatter(range(T), r[:T], marker="x", color="k", label="actual", zorder=3)
        ax.axvline(step, color="red", ls="--", alpha=0.6, label=f"step {step}")
        ax.set_title(f"Predictions vs actual - student {b} "
                     f"(guess={trace['guess']:.2f}, slip={trace['slip']:.2f})")
        ax.set_xlabel("step"); ax.set_ylabel("P(correct)"); ax.legend(fontsize=8)
        fig.tight_layout()
        return fig
    raise ValueError(f"unknown panel: {panel}")


def attribution_case_study(model, batch, b: int = 0, step: int | None = None,
                           kc_names=None, device="cuda", auto_pick: bool = True):
    """Multi-panel figure: pattern weights, gating, mastery curves, β, ŷ.

    If ``auto_pick`` is True and ``step`` is None, the function searches the
    selected student ``b`` for a prediction step that (a) is inside the real
    (non-padded) prefix and (b) has at least one valid related KC, so the
    gating / β panels have something to draw. If no such step exists, the
    original ``step`` is used and panels will be empty.
    """
    import matplotlib.pyplot as plt

    from . import visualize as viz

    model.eval()
    sub = {k: v.to(device) for k, v in batch.items()}
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16,
                                          enabled=(device == "cuda")):
        out = model(sub, return_trace=True)
    trace = out["trace"]
    names = trace["pattern_names"]
    kc_ids = most_active_kcs(trace, b)

    if auto_pick and step is None:
        rel = np.asarray(trace["rel"])[b]               # (L, K_rel)
        sel = sub["selectmask"][b].cpu().numpy()        # (L,)
        valid = (rel >= 0).any(axis=1) & sel
        step = int(np.argmax(valid)) if valid.any() else (step or 0)

    figs = {
        "pattern_weights": viz.plot_pattern_weights(trace, b, step, names),
        "gating": viz.plot_gating_heatmap(trace, b, step, names, kc_names),
        "mastery": viz.plot_mastery_evolution(
            trace, b, kc_ids, sub["q"][b].cpu().numpy(),
            sub["r"][b].cpu().numpy(), kc_names),
        "beta": viz.plot_beta_contributions(trace, b, step, kc_names),
    }
    y = out["y"][b].float().cpu().numpy()
    r = sub["r"][b].cpu().numpy()
    fig, ax = plt.subplots(figsize=(11, 3))
    T = min(60, len(y))
    ax.plot(range(T), y[:T], "-o", ms=3, label="predicted P(correct)")
    ax.scatter(range(T), r[:T], marker="x", color="k", label="actual", zorder=3)
    ax.axvline(step, color="red", ls="--", alpha=0.6, label=f"step {step}")
    ax.set_title(f"Predictions vs actual - student {b} "
                 f"(guess={trace['guess']:.2f}, slip={trace['slip']:.2f})")
    ax.set_xlabel("step"); ax.set_ylabel("P(correct)"); ax.legend(fontsize=8)
    fig.tight_layout()
    figs["prediction"] = fig
    return figs


def pick_attribution_examples(batch, k: int = 4, strategy: str = "spread"):
    """Return up to ``k`` (student, step) pairs suitable for attribution plots.

    Each pair satisfies two conditions:
      * the prediction step is inside the real (non-padded) prefix, and
      * the step has at least one related KC (rel >= 0), so the β / gating
        panels have content to draw.

    Strategies:
      * ``"spread"`` (default): pick one pair per student at quartile points
        of that student's real prefix length, so the figures cover early,
        mid, and late interactions rather than always step 1.
      * ``"first"``: legacy behaviour - pick the first valid step per student.
    """
    import torch

    sel = batch["selectmask"].cpu().numpy()            # (B, L)
    kc = batch["kc"].cpu().numpy()                     # (B, L, K_max)
    B, L = sel.shape
    pairs: list[tuple[int, int]] = []
    student_ids: list[int] = list(range(min(B, k)))
    if strategy == "spread":
        step_fractions = [0.25, 0.5, 0.75, 0.9]
    else:
        step_fractions = [0.0]
    for i, b in enumerate(student_ids):
        prefix_len = int(sel[b].sum())
        if prefix_len < 2:
            continue
        target = step_fractions[i % len(step_fractions)]
        # Walk forward from the desired fraction to find a step with a KC.
        s = max(1, int(prefix_len * target))
        while s < prefix_len and not (kc[b, s] >= 0).any():
            s += 1
        if (kc[b, s] >= 0).any():
            pairs.append((b, s))
    return pairs
