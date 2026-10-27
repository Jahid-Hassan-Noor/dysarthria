"""Make supplementary V8 paper figures from frozen, saved results only.

Run from anywhere with the project Python environment:
    .venv/bin/python v8/v8_research_completion/make_paper_figures.py
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
SAVED = ROOT / "v8_publication_completion"
OUT = Path(__file__).resolve().parent / "figures"
SOURCES = OUT / "source_data"
CONTROL = "#287a89"
DYSARTHRIC = "#bf6330"
NEUTRAL = "#353d49"


def read(relative: str, required: tuple[str, ...]) -> pd.DataFrame:
    path = SAVED / relative
    frame = pd.read_csv(path)
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing {sorted(missing)}")
    return frame


def save(fig: plt.Figure, stem: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(OUT / f"{stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def duration_figure() -> dict:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharex=True, sharey=True)
    summary = {}
    for ax, corpus, pred_path, meta_path in (
        (axes[0], "TORGO", "results/primary_torgo/predictions.csv",
         "frozen_reference/run/features/usable_metadata.csv"),
        (axes[1], "UASpeech M3", "results/primary_uaspeech_m3/predictions.csv",
         "frozen_reference/run/external_uaspeech/M3/features/usable_metadata.csv"),
    ):
        pred = read(pred_path, ("physical_speaker_id", "seed", "label", "probability"))
        meta = read(meta_path, ("physical_speaker_id", "label", "source_seconds"))
        if pred.duplicated(["physical_speaker_id", "seed"]).any():
            raise ValueError(f"Duplicate {corpus} speaker/seed predictions")
        speaker = pred.groupby("physical_speaker_id").agg(
            label=("label", "first"), mean_probability=("probability", "mean"),
            min_probability=("probability", "min"), max_probability=("probability", "max"),
            n_seeds=("seed", "nunique"),
        )
        durations = meta.groupby("physical_speaker_id").agg(
            median_duration_s=("source_seconds", "median"),
            mean_duration_s=("source_seconds", "mean"),
            n_recordings=("source_seconds", "size"), label_from_audio=("label", "first"),
        )
        speaker = speaker.join(durations, validate="one_to_one")
        if speaker.isna().any().any() or (speaker.label != speaker.label_from_audio).any():
            raise ValueError(f"Speaker inventory mismatch in {corpus}")
        if speaker.n_seeds.nunique() != 1 or speaker.n_seeds.iloc[0] != 5:
            raise ValueError(f"Expected five predictions per {corpus} speaker")
        speaker = speaker.drop(columns="label_from_audio").reset_index()
        speaker.insert(0, "corpus", corpus)
        speaker.to_csv(SOURCES / f"duration_vs_score_{corpus.lower().replace(' ', '_')}.csv", index=False)

        for label, name, color, marker in ((0, "Control", CONTROL, "o"),
                                           (1, "Dysarthric", DYSARTHRIC, "s")):
            part = speaker[speaker.label.eq(label)]
            low = part.mean_probability - part.min_probability
            high = part.max_probability - part.mean_probability
            ax.errorbar(part.median_duration_s, part.mean_probability,
                        yerr=np.vstack([low, high]), fmt=marker, markersize=5.6,
                        color=color, ecolor=color, elinewidth=.9, capsize=2,
                        linestyle="none", alpha=.9, label=f"{name} (n={len(part)})")
        rho = float(spearmanr(speaker.median_duration_s, speaker.mean_probability).statistic)
        summary[corpus] = {"n_speakers": len(speaker), "rho_all_speakers": rho,
                           "n_recordings": int(speaker.n_recordings.sum())}
        ax.set_title(f"{corpus}  ·  Spearman ρ = {rho:+.2f}")
        ax.set_xlabel("Median recording duration per speaker (s)")
        ax.grid(alpha=.17)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
    axes[0].set_ylabel("Mean V8 dysarthria probability across five seeds")
    axes[0].set_ylim(-.04, 1.04)
    axes[0].set_xlim(1.1, 6.15)
    fig.suptitle("Recording duration and frozen V8 speaker scores")
    fig.text(.5, -.025, "Whiskers show the five-seed score range; duration is measured on the saved usable recordings.",
             ha="center", fontsize=9, color=NEUTRAL)
    fig.tight_layout()
    save(fig, "fig10_duration_vs_v8_score")
    return summary


def matched_content_figure() -> dict:
    full = read("results/primary_uaspeech_m3/predictions.csv",
                ("physical_speaker_id", "seed", "label", "probability", "threshold"))
    matched = read("results/matched_content/predictions.csv",
                   ("physical_speaker_id", "seed", "label", "probability", "threshold"))
    if full.duplicated(["physical_speaker_id", "seed"]).any() or matched.duplicated(["physical_speaker_id", "seed"]).any():
        raise ValueError("Duplicate UASpeech speaker/seed predictions")
    paired = matched.merge(full, on=["physical_speaker_id", "seed"],
                           suffixes=("_matched", "_full"), validate="one_to_one")
    if len(paired) != len(matched) or (paired.label_matched != paired.label_full).any():
        raise ValueError("Matched-content cohort does not align with the full M3 cohort")
    if not np.allclose(paired.threshold_matched, paired.threshold_full, atol=1e-12):
        raise ValueError("Matched-content predictions did not use the locked thresholds")
    paired["delta_matched_minus_full"] = paired.probability_matched - paired.probability_full
    paired["decision_changed"] = ((paired.probability_matched >= paired.threshold_matched) !=
                                   (paired.probability_full >= paired.threshold_full))
    paired.to_csv(SOURCES / "matched_words_paired_seed_scores.csv", index=False)
    speakers = paired.groupby("physical_speaker_id").agg(
        label=("label_full", "first"), full_score=("probability_full", "mean"),
        matched_score=("probability_matched", "mean"),
        mean_delta=("delta_matched_minus_full", "mean"),
        changed_decisions=("decision_changed", "sum"), n_seeds=("seed", "nunique"),
    ).reset_index()
    if speakers.n_seeds.nunique() != 1 or speakers.n_seeds.iloc[0] != 5:
        raise ValueError("Expected five seeds for every matched-content speaker")
    speakers.to_csv(SOURCES / "matched_words_paired_speaker_scores.csv", index=False)

    fig, ax = plt.subplots(figsize=(5.8, 5.5))
    ax.plot([0, 1], [0, 1], color="#9299a1", lw=1, ls="--", label="No score change")
    for label, name, color, marker in ((0, "Control", CONTROL, "o"),
                                       (1, "Dysarthric", DYSARTHRIC, "s")):
        part = speakers[speakers.label.eq(label)]
        ax.scatter(part.full_score, part.matched_score, s=42, marker=marker,
                   color=color, alpha=.9, label=f"{name} (n={len(part)})")
    changed = int(paired.decision_changed.sum())
    ax.text(.04, .97, f"{len(speakers)} speakers · {changed}/{len(paired)} locked decisions changed",
            transform=ax.transAxes, va="top", fontsize=9,
            bbox={"boxstyle": "round,pad=.35", "facecolor": "white", "edgecolor": "#dadde0"})
    ax.set(xlim=(-.04, 1.04), ylim=(-.04, 1.04),
           xlabel="Full M3: mean V8 probability", ylabel="Matched words: mean V8 probability",
           title="UASpeech M3 · same speakers, locked model")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=.15)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    save(fig, "fig11_matched_words_paired_scores")
    return {"n_speakers": len(speakers), "n_speaker_seed_pairs": len(paired),
            "changed_locked_decisions": changed,
            "mean_absolute_score_change": float(paired.delta_matched_minus_full.abs().mean())}


def ablation_delta_figure() -> dict:
    data = read("ablations/paired_differences.csv",
                ("model_arm", "reference", "metric", "delta", "lower_95", "upper_95", "n_resamples"))
    metrics = ("Balanced_Accuracy", "ROC_AUC", "PR_AUC")
    arms = (("A1", "Last layer + mean"), ("A2", "Last layer + attention"),
            ("B1", "Last-4 mix + mean"))
    part = data[data.metric.isin(metrics) & data.model_arm.isin([a for a, _ in arms])].copy()
    if len(part) != len(metrics) * len(arms) or part.duplicated(["model_arm", "metric"]).any():
        raise ValueError("Expected exactly one paired interval per ablation and metric")
    if (part.reference != "B2").any() or (part.lower_95 > part.delta).any() or (part.upper_95 < part.delta).any():
        raise ValueError("Invalid paired ablation comparison")
    part.to_csv(SOURCES / "ablation_paired_deltas.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), sharey=True)
    titles = {"Balanced_Accuracy": "Balanced accuracy", "ROC_AUC": "ROC AUC", "PR_AUC": "PR AUC"}
    for ax, metric in zip(axes, metrics):
        for y, (arm, _) in enumerate(arms):
            row = part[(part.model_arm == arm) & (part.metric == metric)].iloc[0]
            ax.errorbar(row.delta, y,
                        xerr=np.array([[row.delta - row.lower_95], [row.upper_95 - row.delta]]),
                        fmt="o", color=CONTROL, ecolor=CONTROL, capsize=3, markersize=5)
        ax.axvline(0, color=NEUTRAL, lw=1, ls="--")
        ax.set_title(titles[metric])
        ax.set_xlabel("Ablation − V8 reference")
        ax.grid(axis="x", alpha=.16)
        ax.set_ylim(2.5, -.5)
    axes[0].set_yticks(range(3), [name for _, name in arms])
    fig.suptitle("TORGO paired speaker-bootstrap effects · 95% confidence intervals")
    fig.tight_layout()
    save(fig, "fig12_paired_ablation_deltas")
    return {"n_comparisons": len(part), "n_resamples": sorted(part.n_resamples.unique().tolist()),
            "balanced_accuracy_intervals_include_zero": bool(
                ((part[part.metric.eq("Balanced_Accuracy")].lower_95 <= 0) &
                 (part[part.metric.eq("Balanced_Accuracy")].upper_95 >= 0)).all())}


def epoch_figure() -> dict:
    base = SAVED / "frozen_reference/run/folds/speech_only"
    rows = []
    fold_re = re.compile(r"seed(\d+)_outer(\d+)")
    for audit_path in sorted(base.glob("seed*_outer*/fold_audit.json")):
        match = fold_re.fullmatch(audit_path.parent.name)
        if not match:
            raise ValueError(f"Unexpected fold name: {audit_path.parent.name}")
        seed, outer = (int(x) for x in match.groups())
        audit = json.loads(audit_path.read_text())
        rows.append({"seed": seed, "outer_fold": outer, "stage": "Final fit",
                     "inner_fold": pd.NA, "epoch": int(audit["final_epochs"]),
                     "epochs_trained": int(audit["final_epochs"])})
        for inner_path in sorted(audit_path.parent.glob("inner*/complete.json")):
            inner = int(inner_path.parent.name.removeprefix("inner"))
            record = json.loads(inner_path.read_text())
            rows.append({"seed": seed, "outer_fold": outer, "stage": "Inner CV best",
                         "inner_fold": inner, "epoch": int(record["best_epoch"]),
                         "epochs_trained": len(record["history"])})
    epochs = pd.DataFrame(rows)
    inner = epochs[epochs.stage.eq("Inner CV best")]
    final = epochs[epochs.stage.eq("Final fit")]
    if len(inner) != 75 or len(final) != 25 or epochs.duplicated(["seed", "outer_fold", "stage", "inner_fold"]).any():
        raise ValueError("Expected 75 inner folds and 25 TORGO final fits")
    epochs.to_csv(SOURCES / "torgo_epoch_selection.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.4, 4.1))
    rng = np.random.default_rng(20261014)
    for x, (label, group, color) in enumerate((("Inner CV best", inner, CONTROL),
                                               ("Final fit", final, DYSARTHRIC))):
        vals = group.epoch.to_numpy(dtype=float)
        jitter = rng.uniform(-.16, .16, size=len(vals))
        ax.scatter(np.full(len(vals), x) + jitter, vals, s=18, alpha=.48, color=color)
        q1, median, q3 = np.quantile(vals, [.25, .5, .75])
        ax.plot([x-.23, x+.23], [median, median], color=NEUTRAL, lw=2.6)
        ax.plot([x, x], [q1, q3], color=NEUTRAL, lw=2.2, zorder=4)
        ax.text(x+.21, median+1.2, f"median {median:g}", fontsize=8, color=NEUTRAL)
    ax.set_xticks((0, 1), ("Inner fold optimum\n75 fits", "Selected final epoch\n25 outer folds"))
    ax.set_ylabel("Training epoch")
    ax.set_title("TORGO V8 epoch-selection variability")
    ax.set_xlim(-.45, 1.55)
    ax.set_ylim(0, max(epochs.epoch.max(), epochs.epochs_trained.max()) + 4)
    ax.grid(axis="y", alpha=.18)
    fig.tight_layout()
    save(fig, "fig13_epoch_selection")
    return {"n_inner_fits": len(inner), "n_outer_final_fits": len(final),
            "inner_best_median": float(inner.epoch.median()),
            "inner_best_range": [int(inner.epoch.min()), int(inner.epoch.max())],
            "final_epoch_median": float(final.epoch.median()),
            "final_epoch_range": [int(final.epoch.min()), int(final.epoch.max())],
            "inner_best_at_last_trained_epoch": int((inner.epoch == inner.epochs_trained).sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    SOURCES.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    summaries = {"duration": duration_figure(), "matched_content": matched_content_figure(),
                 "paired_ablation": ablation_delta_figure(), "epochs": epoch_figure()}
    source_paths = [
        "results/primary_torgo/predictions.csv",
        "results/primary_uaspeech_m3/predictions.csv",
        "results/matched_content/predictions.csv",
        "frozen_reference/run/features/usable_metadata.csv",
        "frozen_reference/run/external_uaspeech/M3/features/usable_metadata.csv",
        "ablations/paired_differences.csv",
    ]
    source_paths.extend(str(path.relative_to(SAVED)) for path in sorted(
        (SAVED / "frozen_reference/run/folds/speech_only").glob("seed*_outer*/fold_audit.json")))
    source_paths.extend(str(path.relative_to(SAVED)) for path in sorted(
        (SAVED / "frozen_reference/run/folds/speech_only").glob("seed*_outer*/inner*/complete.json")))
    manifest = {"description": "Derived figures from frozen V8 saved results; no model retraining",
                "summaries": summaries,
                "source_sha256": {name: hashlib.sha256((SAVED / name).read_bytes()).hexdigest()
                                  for name in source_paths}}
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Saved four PNG/PDF figure pairs to {OUT}")


if __name__ == "__main__":
    main()
