"""Render compact Markdown views of the full-precision publication CSV tables."""
from __future__ import annotations

import csv
from pathlib import Path
from statistics import mean


TABLES = Path(__file__).resolve().parent / "tables"
METRICS = (
    ("ROC_AUC", "ROC-AUC"),
    ("Balanced_Accuracy", "Balanced accuracy"),
    ("Sensitivity", "Sensitivity"),
    ("Specificity", "Specificity"),
)


def rows(stem: str) -> list[dict[str, str]]:
    with (TABLES / f"{stem}.csv").open(newline="") as source:
        return list(csv.DictReader(source))


def display(value: str) -> str:
    return f"{float(value):.3f}" if value else "—"


def metric_table(stem: str, title: str, note: str) -> None:
    source = rows(stem)
    models = list(dict.fromkeys(row["model"] for row in source))
    lookup = {(row["model"], row["metric"]): row for row in source}
    lines = [
        f"# {title}", "",
        "| Model or condition | ROC-AUC [95% interval if reported] | Balanced accuracy | Sensitivity | Specificity |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for model in models:
        values = []
        for key, _ in METRICS:
            row = lookup.get((model, key))
            values.append(display(row["mean"]) if row else "—")
        auc = lookup.get((model, "ROC_AUC"))
        if auc:
            lower = auc.get("speaker_bootstrap_lower_95") or auc.get("lower_95")
            upper = auc.get("speaker_bootstrap_upper_95") or auc.get("upper_95")
            if lower and upper:
                values[0] += f" [{display(lower)}, {display(upper)}]"
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    lines += [
        "", note, "",
        f"Full-precision values, additional metrics, and seed variation: [{stem}.csv]({stem}.csv).",
        "",
    ]
    (TABLES / f"{stem}.md").write_text("\n".join(lines))


def layer_table() -> None:
    stem = "table17_layer_weights"
    source = rows(stem)
    layers = list(dict.fromkeys(row["layer"] for row in source))
    lines = [
        "# WavLM layer mixture weights", "",
        "| Layer | Mean normalized weight | Range across 25 outer folds |",
        "| --- | ---: | ---: |",
    ]
    for layer in layers:
        values = [float(row["normalized_weight"]) for row in source if row["layer"] == layer]
        assert len(values) == 25, (layer, len(values))
        lines.append(f"| {layer} | {mean(values):.3f} | {min(values):.3f}–{max(values):.3f} |")
    lines += [
        "", "Weights are descriptive; near-equal weights do not identify a speech mechanism.",
        f"All 100 fold-level values: [{stem}.csv]({stem}.csv).", "",
    ]
    (TABLES / f"{stem}.md").write_text("\n".join(lines))


def main() -> None:
    metric_table("table03_simple_baselines", "Simple baselines", "TORGO: 15 physical speakers across five repeated seeds. Intervals resample speakers; seeds are not independent cohorts.")
    metric_table("table04_controlled_ablations", "Controlled ablations", "TORGO: five repeated seeds. Intervals follow the original ablation analysis. These comparisons do not isolate a clinical mechanism.")
    metric_table("table08_channels", "UASpeech channel comparisons", "Speaker counts differ by channel. Intervals resample physical speakers within each channel.")
    metric_table("table10_confound_diagnostics", "Confound diagnostics", "TORGO: 15 physical speakers across five repeated seeds. A missing interval means the source table did not report one; cue-only discrimination does not establish cue use by the locked detector.")
    layer_table()


if __name__ == "__main__":
    main()
