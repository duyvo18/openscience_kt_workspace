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


def attribution_case_study(model, batch, b: int, step: int, kc_names=None,
                           device="cuda"):
    """Multi-panel figure: pattern weights, gating, mastery curves, β, ŷ."""
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
    ax.set_title(f"Predictions vs actual — student {b} "
                 f"(guess={trace['guess']:.2f}, slip={trace['slip']:.2f})")
    ax.set_xlabel("step"); ax.set_ylabel("P(correct)"); ax.legend(fontsize=8)
    fig.tight_layout()
    figs["prediction"] = fig
    return figs
