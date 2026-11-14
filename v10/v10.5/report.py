"""Generate publication table source files from completed exploratory stages."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from support import HERE, OUT, ROOT, V10, metrics, read_json, reference_predictions, sha256, write_csv, write_json


def main() -> dict:
    tables = OUT / "tables"
    result = {"designation": "exploratory follow-up analyses"}
    prompt = OUT / "prompt_match"
    coverage = read_json(prompt / "coverage.json") if (prompt / "coverage.json").is_file() else {"status": "not_run"}
    result["prompt_match"] = coverage["status"]
    if (prompt / "per_seed_metrics.csv").is_file() and coverage["status"] == "complete":
        data = pd.read_csv(prompt / "per_seed_metrics.csv")
        summary = data.groupby("arm", as_index=False)[["roc_auc", "balanced_accuracy", "sensitivity",
                                                         "specificity", "brier", "ece10"]].mean()
        write_csv(tables / "table01_prompt_matched_duration.csv", summary)
    else:
        original_duration = pd.read_csv(ROOT / "v8/v8_publication_completion/results/duration/predictions.csv")
        original_duration = original_duration.query("corpus == 'TORGO'")
        original_wavlm = reference_predictions()
        context = []
        for name, frame in (("duration_original", original_duration), ("wavlm_original", original_wavlm)):
            values = pd.DataFrame([metrics(group) for _, group in frame.groupby("seed")]).mean()
            context.append({"arm": name, "status": "published_full_cohort_context", **values.to_dict()})
        for name in ("duration_raw_matched", "duration_preprocessed_matched", "wavlm_matched"):
            context.append({"arm": name, "status": coverage["status"]})
        write_csv(tables / "table01_prompt_matched_duration.csv", pd.DataFrame(context))
    arms = []
    comparisons = []
    effects = []
    for corpus in ("TORGO", "UASpeech"):
        folder = OUT / "background_intervention" / corpus
        if not (folder / "summary.json").is_file():
            result[corpus] = "not_run"
            continue
        result[corpus] = "complete"
        data = pd.read_csv(folder / "per_seed_metrics.csv")
        mean = data.groupby("condition", as_index=False)[["roc_auc", "balanced_accuracy",
                                                            "sensitivity", "specificity",
                                                            "brier", "ece10"]].mean()
        mean.insert(0, "corpus", corpus)
        arms.append(mean)
        paired = pd.read_csv(folder / "paired_bootstrap.csv")
        paired.insert(0, "corpus", corpus)
        comparisons.append(paired)
        effect = pd.read_csv(folder / "paired_effect_bootstrap.csv")
        effect.insert(0, "corpus", corpus)
        effects.append(effect)
    for filename, frames in (("table02_background_conditions.csv", arms),
                             ("table03_background_paired_metrics.csv", comparisons),
                             ("table04_background_paired_score_effects.csv", effects)):
        if frames:
            write_csv(tables / filename, pd.concat(frames, ignore_index=True))
    source_paths = [HERE / name for name in ("analysis_plan.json", "support.py", "prompt_match.py",
                                             "background_intervention.py", "figures.py", "report.py",
                                             "run_v10_5.py")]
    source_paths += [ROOT / "v10/v10.4/source_manifest.json",
                     V10 / "deployment/v8_torgo_model_lock.json"]
    for corpus in ("TORGO", "UASpeech"):
        manifest = OUT / "background_intervention" / corpus / "subset_manifest.csv"
        if manifest.is_file():
            source_paths.append(manifest)
            source_paths.extend(pd.read_csv(manifest).file_path.map(
                lambda x: ROOT / x if not Path(x).is_absolute() else Path(x)))
    prompt_map = HERE / "torgo_prompts.csv"
    if prompt_map.is_file():
        source_paths.append(prompt_map)
    unique_paths = sorted(set(source_paths))
    write_json(HERE / "provenance.json", {
        "version": "v10.5-1", "designation": result["designation"],
        "inputs_and_code": [{"path": str(path), "sha256": sha256(path), "bytes": path.stat().st_size}
                            for path in unique_paths],
    })
    write_json(OUT / "report_status.json", result)
    return result


if __name__ == "__main__":
    print(main())
