"""Generate the V10.4 results document, figures, and executable notebook."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import numpy as np
import pandas as pd

from common import HERE, OUT, read_json, sha256, write_json

FIGURES = OUT / "figures"


def number(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def interval(values) -> str:
    return f"[{number(values[0])}, {number(values[1])}]"


def save_figure(fig, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def overview_figure() -> None:
    threshold = pd.read_csv(OUT / "threshold/per_seed_metrics.csv")
    duration = pd.read_csv(OUT / "duration_increment/per_seed_auc.csv")
    background = pd.read_csv(OUT / "background/per_seed_metrics.csv")
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), layout="constrained")
    ax = axes[0]
    old = threshold[threshold.rule == "original"].sort_values("seed")
    new = threshold[threshold.rule == "youden"].sort_values("seed")
    for x, y in zip(old.balanced_accuracy, new.balanced_accuracy):
        ax.plot([0, 1], [x, y], color="0.7", linewidth=1)
    ax.scatter(np.zeros(len(old)), old.balanced_accuracy, color="#245782", label="Original")
    ax.scatter(np.ones(len(new)), new.balanced_accuracy, color="#aa603b", label="Youden J")
    ax.set_xticks([0, 1], ["Original", "Youden J"])
    ax.set_ylim(0.55, 1.01)
    ax.set_ylabel("Pooled balanced accuracy")
    ax.set_title("(a) Threshold rule")

    ax = axes[1]
    arms = ["probability_duration", "duration_stack", "duration_plus_v10", "probability_v10"]
    labels = ["Original\nduration", "Duration\nstack", "Duration +\nV10", "V10"]
    means = [duration.loc[duration.arm == arm, "roc_auc"].mean() for arm in arms]
    ax.bar(range(4), means, color=["#638caa", "#638caa", "#b67546", "#b67546"])
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean seed-pooled ROC-AUC")
    ax.set_title("(b) Duration increment")

    ax = axes[2]
    arms = ["background_trained", "locked_speech_same_recordings"]
    labels = ["Background\ntrained", "Speech on same\nrecordings"]
    means = [background.loc[background.arm == arm, "roc_auc"].mean() for arm in arms]
    ax.bar(range(2), means, color=["#8a6a96", "#638caa"])
    ax.set_xticks(range(2), labels, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean seed-pooled ROC-AUC")
    ax.set_title("(c) Background-edge control")
    save_figure(fig, "fig01_v10_4_saved_and_background")


def spread_figure() -> None:
    scores = pd.read_csv(OUT / "spread/speaker_scores_by_condition.csv")
    effects = pd.read_csv(OUT / "spread/speaker_effects.csv")
    conditions = ["original", "identity", "spread_0.88", "spread_1.12", "roundtrip"]
    labels = ["Original", "Identity", "SD ×0.88", "SD ×1.12", "Round-trip"]
    speaker_mean = scores.groupby(["physical_speaker_id", "label", "condition"], as_index=False).logit.mean()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), layout="constrained")
    rng = np.random.default_rng(20261021)
    ax = axes[0]
    for label, name, color in ((0, "Control", "#245782"), (1, "Dysarthric", "#aa603b")):
        group = speaker_mean[speaker_mean.label == label].pivot(index="physical_speaker_id", columns="condition", values="logit")
        values = group[conditions].to_numpy()
        bootstrap = np.array([
            values[rng.integers(0, len(values), size=len(values))].mean(axis=0)
            for _ in range(2000)
        ])
        mean = values.mean(axis=0)
        lo, hi = np.percentile(bootstrap, [2.5, 97.5], axis=0)
        x = np.arange(len(conditions))
        ax.plot(x, mean, "o-", color=color, label=name)
        ax.fill_between(x, lo, hi, color=color, alpha=0.15)
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Seed-mean speaker logit")
    ax.set_title("(a) Locked model response")
    ax.legend(fontsize=8)
    ax = axes[1]
    for label, color in ((0, "#245782"), (1, "#aa603b")):
        part = effects[effects.label == label]
        x = np.full(len(part), label) + np.linspace(-0.1, 0.1, len(part))
        ax.scatter(x, part.spread_delta, color=color, s=35, alpha=0.9)
    ax.axhline(0, color="0.4", linestyle=":")
    ax.set_xticks([0, 1], ["Control", "Dysarthric"])
    ax.set_ylabel("Logit(SD ×1.12) − Logit(SD ×0.88)")
    ax.set_title("(b) Speaker-level spread effect")
    save_figure(fig, "fig02_v10_4_spread_intervention")


def render_results() -> str:
    t = read_json(OUT / "threshold/summary.json")
    f = read_json(OUT / "fold_auc/summary.json")
    d = read_json(OUT / "duration_increment/summary.json")
    b = read_json(OUT / "background/summary.json")
    s = read_json(OUT / "spread/summary.json")
    checked = read_json(OUT / "spread/pipeline_checks.json")
    auc = d["mean_auc_by_arm"]
    spread_aucs = s["mean_auc_by_condition"]
    lines = [
        "# Dysarthria V10.4: five follow-up analyses",
        "",
        "V10.4 is a secondary, post hoc analysis of the frozen speech-only V10 detector. "
        "V8–V10.3 code and results were used only as read-only inputs. The original 15 TORGO "
        "physical speakers recur across five seeds; seeds are not independent cohorts.",
        "",
        "## Results at a glance",
        "",
        "| Question | V10.4 result | Interpretation |",
        "| --- | --- | --- |",
        f"| Youden J threshold | Mean balanced accuracy {number(t['youden_mean_balanced_accuracy'])} "
        f"versus {number(t['original_mean_balanced_accuracy'])} original; paired Δ "
        f"{number(t['youden_minus_original_balanced_accuracy'])}, 95% bootstrap "
        f"{interval(t['paired_bootstrap_delta_ba_95'])} | No improvement under this threshold rule. AUC remains {number(t['mean_roc_auc_both_rules'])}. |",
        f"| Per-fold AUC | {f['auc_value_counts'].get('1.0', 0)}/25 folds at 1.0; "
        f"{f['auc_value_counts'].get('0.5', 0)}/25 at 0.5; mean seed-pooled AUC "
        f"{number(f['mean_seed_pooled_auc_primary'])} | Each fold has two positive-negative pairs; fold results are descriptive. |",
        f"| Increment over duration | AUC {number(auc['duration_stack'])} duration stack, "
        f"{number(auc['duration_plus_v10'])} duration + V10; Δc "
        f"{number(d['incremental_delta_c']['point_delta_auc'])}, 95% bootstrap "
        f"[{number(d['incremental_delta_c']['lower_95'])}, {number(d['incremental_delta_c']['upper_95'])}] "
        f"| Incremental value is not established; original duration-only AUC {number(auc['probability_duration'])}. |",
        f"| Background-edge classifier | AUC {number(b['background_mean_auc'])} "
        f"[{number(b['background_minus_matched_speech_auc']['first_auc_lower_95'])}, "
        f"{number(b['background_minus_matched_speech_auc']['first_auc_upper_95'])}] versus "
        f"{number(b['matched_speech_mean_auc'])} "
        f"[{number(b['background_minus_matched_speech_auc']['second_auc_lower_95'])}, "
        f"{number(b['background_minus_matched_speech_auc']['second_auc_upper_95'])}] "
        f"for locked speech heads on the same source recordings; "
        f"paired Δ {number(b['background_minus_matched_speech_auc']['point_delta_auc'])}, "
        f"95% bootstrap [{number(b['background_minus_matched_speech_auc']['lower_95'])}, "
        f"{number(b['background_minus_matched_speech_auc']['upper_95'])}] "
        "| Background edges carry a discriminative signal, but its source is not identified. |",
        f"| Mean-preserving spread intervention | Mean logit(high SD − low SD) "
        f"{number(s['mean_spread_1_12_minus_0_88_logit'])}, 95% bootstrap "
        f"{interval(s['physical_speaker_bootstrap_95'])}; verdict: **{s['verdict']}** "
        "| Processing controls constrain interpretation. |",
        "",
        "![Overview](outputs/figures/fig01_v10_4_saved_and_background.png)",
        "",
        "![Spread intervention](outputs/figures/fig02_v10_4_spread_intervention.png)",
        "",
        "## Analysis details",
        "",
        "### Threshold and per-fold ranking",
        "",
        "Youden J was fitted separately to each original outer fold's 12 saved inner-OOF "
        "speaker predictions, then applied to its three untouched outer speakers. "
        "The frozen scores, epochs, and model weights were unchanged. The paired confidence "
        "interval resampled physical speakers within diagnosis and reused each draw across seeds. "
        "The 24 perfect within-fold rankings coexist with a lower pooled AUC because "
        "pooled AUC also compares scores produced by different fold models.",
        "",
        "### Incremental information beyond duration",
        "",
        "The duration-only and duration-plus-V10 stacks used the same fixed logistic recipe, "
        "the same original inner-OOF training speakers, and the same untouched outer test "
        "speakers. Both sets of inner predictions were already saved, and the duration "
        "scores came from the original four-statistic duration model. The direct "
        "original duration-only comparator was retained as "
        f"context (AUC {number(auc['probability_duration'])}), and V10 alone had AUC "
        f"{number(auc['probability_v10'])}. The small stacked Δc and its wide interval "
        "do not support a claim of disease-specific discrimination beyond duration.",
        "",
        "### Classifier trained on background edges",
        "",
        "A fresh WavLM-last-four speaker-attention head was fitted to the saved 750 "
        "low-energy TORGO edge embeddings under all 25 original outer folds and their "
        "speaker-held-out inner folds. Its epoch and threshold selection followed the "
        "frozen recipe. For the comparator, locked V10 speech heads scored the same 50 "
        "source recordings per speaker. These objective low-energy edges are not proven "
        "speech-free; diagnosis-correlated voice, microphone, room, or selection effects "
        "could contribute. The paired interval includes zero.",
        "",
        "### Duration-spread intervention",
        "",
        "Fifty recordings per speaker were chosen by a label-blind hash after fixed "
        "duration gates. Their preprocessed durations were transformed around each "
        "speaker's own mean: target = mean + α × (original − mean), with α=0.88 "
        "or 1.12. This preserves each speaker's mean and changes their within-speaker "
        "duration SD by the stated factors before sample rounding. The locked outer-fold "
        "heads scored newly extracted WavLM features for identity, low-spread, "
        "high-spread, and same-length round-trip controls. No altered audio trained a model.",
        "",
        f"Mean five-seed ROC-AUC: original {number(spread_aucs['original'])}, identity "
        f"{number(spread_aucs['identity'])}, low SD {number(spread_aucs['spread_0.88'])}, "
        f"high SD {number(spread_aucs['spread_1.12'])}, round-trip "
        f"{number(spread_aucs['roundtrip'])}. Median absolute high-versus-low logit "
        f"shift was {number(s['median_abs_spread_logit_shift'])}; median absolute "
        f"round-trip-versus-identity shift was {number(s['median_abs_roundtrip_artifact_logit_shift'])}. "
        f"The predeclared artifact gate {'passed' if s['artifact_gate_passed'] else 'failed'}. "
        f"Maximum achieved-versus-target duration difference was "
        f"{number(s['maximum_absolute_target_duration_error_seconds'], 5)} seconds. "
        f"No recordings were cropped in the primary manipulated conditions: "
        f"{s['n_cropped_by_condition'].get('spread_0.88', 0)} low-spread and "
        f"{s['n_cropped_by_condition'].get('spread_1.12', 0)} high-spread.",
        "",
        "## Reproducibility and limits",
        "",
        f"Unmodified audio reproduced frozen WavLM embeddings (relative maximum difference "
        f"{checked['reextraction_relative_max_difference']:.2e}); locked rescoring reproduced "
        f"all 75 published V10 probabilities (maximum difference "
        f"{checked['locked_scoring_max_probability_difference']:.2e}). "
        "The source manifest records SHA-256 hashes for the analysis plan and 158 "
        "frozen inputs. Every new checkpoint and result is under `v10.4/`. "
        "The five seeds repeat the same 15 speakers, so uncertainty is based on physical "
        "speakers. All V10.4 analyses are secondary; neither a high background-edge AUC "
        "nor an audio-perturbation response by itself identifies a clinical mechanism.",
        "",
        "Run `.venv/bin/python v10.4/run_v10_4.py all` from the project root to verify "
        "the frozen inputs and resume every stage. `analysis_plan.json` fixes the analyses; "
        "`source_manifest.json` records their inputs. The `outputs/` subfolders contain "
        "per-speaker predictions, per-seed metrics, confidence intervals, and figures.",
        "",
    ]
    return "\n".join(lines)


def make_notebook(results: str) -> None:
    cells = [
        nbformat.v4.new_markdown_cell(
            "# Dysarthria V10.4\n\nFive follow-up analyses of the frozen V10 speech-only detector. "
            "All new outputs and checkpoints are isolated in `v10.4/`. "
            "Run the code cell to verify the source hashes and reproduce or resume all stages."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\nimport subprocess, sys\n"
            "place = Path.cwd().resolve()\n"
            "version = place if place.name == 'v10.4' else place / 'v10.4'\n"
            "assert (version / 'run_v10_4.py').is_file(), 'Start Jupyter in the project root or v10.4'\n"
            "subprocess.run([sys.executable, str(version / 'run_v10_4.py'), 'all'], check=True)"
        ),
        nbformat.v4.new_markdown_cell(results),
    ]
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python 3 (dysarthria)", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    })
    nbformat.write(notebook, HERE / "dysarthria_v10.4.ipynb")


def main() -> None:
    required = [
        OUT / "threshold/summary.json", OUT / "fold_auc/summary.json",
        OUT / "duration_increment/summary.json", OUT / "background/summary.json",
        OUT / "spread/summary.json", OUT / "spread/pipeline_checks.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"V10.4 stages incomplete: {missing}")
    overview_figure()
    spread_figure()
    results = render_results()
    (HERE / "RESULTS.md").write_text(results)
    make_notebook(results)
    source = read_json(HERE / "source_manifest.json")
    code_paths = [HERE / name for name in (
        "common.py", "reanalyses.py", "background.py", "spread.py", "report.py", "run_v10_4.py"
    )]
    write_json(HERE / "provenance.json", {
        "version": "v10.4-1",
        "source_manifest_sha256": sha256(HERE / "source_manifest.json"),
        "analysis_plan_sha256": sha256(HERE / "analysis_plan.json"),
        "n_read_only_input_artifacts": source["n_inputs"],
        "code_sha256": {path.name: sha256(path) for path in code_paths},
        "result_sha256": {
            str(path.relative_to(HERE)): sha256(path) for path in required
        },
    })
    print("Wrote V10.4 results, two figures, executable notebook, and provenance", flush=True)


if __name__ == "__main__":
    main()
