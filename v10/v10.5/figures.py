"""Publication-style vector and raster figures for exploratory follow-up analyses."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in (".pdf", ".png"):
        fig.savefig(stem.with_suffix(suffix), bbox_inches="tight", dpi=300)
    plt.close(fig)


def prompt_figure(per_seed: pd.DataFrame, subset: pd.DataFrame, stem: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), constrained_layout=True)
    arms = ["duration_original", "duration_raw", "duration_preprocessed",
            "wavlm_original", "wavlm_matched"]
    colors = ["#788896", "#126782", "#4a9b9b", "#b06a42", "#d19b65"]
    for i, arm in enumerate(arms):
        values = per_seed.loc[per_seed.arm == arm, "roc_auc"].to_numpy()
        axes[0].scatter(np.full(len(values), i), values, color=colors[i], alpha=.4, s=14)
        axes[0].plot([i - .25, i + .25], [values.mean()] * 2, color=colors[i], lw=2)
    axes[0].set_xticks(range(len(arms)), ["Original\nduration", "Matched\nraw", "Matched\nprocessed",
                                           "Original\nWavLM", "Matched\nWavLM"])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Speaker-level ROC-AUC")
    axes[0].set_title("A  Prompt-matched discrimination", loc="left", weight="bold")
    for label, group in subset.groupby("label"):
        axes[1].hist(group.raw_seconds, bins=np.linspace(0, min(12, subset.raw_seconds.max()), 21),
                     alpha=.45, density=True, label="Dysarthric" if label else "Control",
                     color="#b06a42" if label else "#126782")
    axes[1].set_xlabel("Raw recording duration (s)")
    axes[1].set_ylabel("Density")
    axes[1].set_title("B  Shared-prompt durations", loc="left", weight="bold")
    axes[1].legend(frameon=False)
    _save(fig, stem)


def background_figure(per_seed: pd.DataFrame, effects: pd.DataFrame, stem: Path) -> None:
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "pdf.fonttype": 42})
    order = ["original", "sham", "neutralized", "swapped", "intensity_half", "intensity_double"]
    labels = ["Original", "Sham", "Neutralized", "Swapped", "Half intensity", "Double intensity"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.3), constrained_layout=True)
    color = "#126782"
    for i, condition in enumerate(order):
        values = per_seed.loc[per_seed.condition == condition, "roc_auc"].to_numpy()
        axes[0].scatter(np.full(len(values), i), values, color=color, alpha=.4, s=13)
        axes[0].plot([i - .23, i + .23], [values.mean()] * 2, color=color, lw=2)
    axes[0].set_xticks(range(len(order)), labels, rotation=40, ha="right")
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Speaker-level ROC-AUC")
    axes[0].set_title("A  Frozen-detector ranking", loc="left", weight="bold")
    conditions = order[2:]
    data = [effects.loc[effects.condition == c, "delta_logit"].to_numpy() for c in conditions]
    axes[1].boxplot(data, positions=np.arange(len(data)), showfliers=False,
                    medianprops={"color": "#b06a42"})
    axes[1].axhline(0, color="#788896", lw=.8)
    axes[1].set_xticks(range(len(data)), labels[2:], rotation=40, ha="right")
    axes[1].set_ylabel("Mean speaker logit change vs sham")
    axes[1].set_title("B  Paired source intervention", loc="left", weight="bold")
    _save(fig, stem)
