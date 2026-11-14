"""Exploratory TORGO prompt-matched duration and locked WavLM evaluation."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from support import (FROZEN, OUT, ROOT, checked_folds, load_backbone, metrics,
                     paired_bootstrap, read_json, reference_predictions, score_condition,
                     sha256, verify_outer_checkpoints, write_csv, write_json)
from v8.publication.baselines import DURATION_FEATURES, fit_classifier, predict

DEST = OUT / "prompt_match"
PROMPT_MAP = Path(__file__).resolve().parent / "torgo_prompts.csv"
SESSION = re.compile(r"(?:[FM]C?\d+)(S\d+)?", re.IGNORECASE)


def normalize_prompt(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9']+", str(value).casefold()))


def session_id(relative_path: str) -> str:
    folder = Path(relative_path).parent.name
    match = SESSION.search(folder)
    return match.group(1).upper() if match and match.group(1) else "unknown"


def align_prompts(meta: pd.DataFrame, prompt_map: pd.DataFrame) -> pd.DataFrame:
    """Join trusted prompt annotations; an utterance number alone is never a prompt."""
    if "source_id" not in prompt_map and "relative_path" not in prompt_map:
        raise ValueError("Prompt map needs source_id or relative_path")
    if not {"prompt_id", "transcript"} & set(prompt_map):
        raise ValueError("Prompt map needs prompt_id or transcript")
    key = "source_id" if "source_id" in prompt_map else "relative_path"
    if prompt_map[key].isna().any() or prompt_map[key].duplicated().any():
        raise ValueError("Prompt map contains missing or repeated recording keys")
    extra = set(prompt_map[key]) - set(meta[key])
    if extra:
        raise ValueError(f"Prompt map has {len(extra)} unknown recordings")
    columns = [key] + [c for c in ("prompt_id", "transcript", "session_id") if c in prompt_map]
    joined = meta.merge(prompt_map[columns], on=key, how="inner", validate="one_to_one")
    if "transcript" in joined:
        joined["normalized_transcript"] = joined.transcript.map(normalize_prompt)
        joined["prompt_type"] = joined.normalized_transcript.map(
            lambda text: "word" if len(text.split()) == 1 else "multiword")
    else:
        joined["prompt_type"] = "unknown"
    if "prompt_id" in joined:
        joined["prompt_key"] = joined.prompt_id.astype(str).str.strip()
        if joined.prompt_key.eq("").any() or joined.prompt_id.isna().any():
            raise ValueError("Blank prompt IDs")
        if "normalized_transcript" in joined:
            inconsistent = joined.groupby("prompt_key").normalized_transcript.nunique()
            if (inconsistent > 1).any():
                raise ValueError("A prompt ID maps to conflicting transcripts")
    else:
        joined["prompt_key"] = joined.normalized_transcript
        if joined.prompt_key.eq("").any():
            raise ValueError("Blank transcript")
    if "session_id" not in joined:
        joined["session_id"] = joined.relative_path.map(session_id)
    return joined


def select_common_prompts(aligned: pd.DataFrame, all_speakers: set[str]) -> pd.DataFrame:
    """One fixed microphone and recording per speaker/prompt; same prompts for all."""
    if aligned.source_id.duplicated().any():
        raise ValueError("Repeated source ID")
    if set(aligned.physical_speaker_id) != all_speakers:
        raise ValueError("Prompt annotation does not cover every TORGO speaker")
    candidates = aligned.copy()
    candidates["_mic_priority"] = candidates.mic_type.map({"head": 0, "array": 1})
    if candidates._mic_priority.isna().any():
        raise ValueError("Unknown TORGO microphone type")
    candidates = candidates.sort_values(["physical_speaker_id", "prompt_key", "_mic_priority",
                                         "session_id", "source_id"])
    candidates = candidates.drop_duplicates(["physical_speaker_id", "prompt_key"])
    coverage = candidates.groupby("prompt_key").physical_speaker_id.nunique()
    common = set(coverage[coverage == len(all_speakers)].index)
    subset = candidates[candidates.prompt_key.isin(common)].drop(columns="_mic_priority")
    if len(subset) != len(common) * len(all_speakers):
        raise AssertionError("Prompt/microphone deduplication failed")
    return subset.sort_values(["physical_speaker_id", "prompt_key"]).reset_index(drop=True)


def speaker_features(subset: pd.DataFrame, duration_column: str) -> pd.DataFrame:
    rows = []
    for speaker, group in subset.groupby("physical_speaker_id"):
        d = group[duration_column].to_numpy(dtype=float)
        rows.append({"physical_speaker_id": speaker, "label": int(group.label.iloc[0]),
                     "n_utterances": len(d), "duration_mean": float(d.mean()),
                     "duration_median": float(np.median(d)),
                     "duration_sd": float(d.std(ddof=0)),
                     "short_fraction": float((d < .5).mean())})
    return pd.DataFrame(rows)


def duration_oof(table: pd.DataFrame, backbone) -> tuple[pd.DataFrame, pd.DataFrame]:
    outer, inner = [], []
    for seed, fold, _, audit in checked_folds():
        train, test = set(audit["train_physical_speakers"]), set(audit["test_physical_speakers"])
        validation = []
        for split in audit["inner_folds"]:
            tr, va = set(split["train_physical_speakers"]), set(split["validation_physical_speakers"])
            if tr & va or tr & test or va & test:
                raise AssertionError("Nested duration speaker leakage")
            model = fit_classifier(table, tr, DURATION_FEATURES, seed)
            validation.append(predict(model, table, va, DURATION_FEATURES))
        valid = pd.concat(validation, ignore_index=True)
        if set(valid.physical_speaker_id) != train or valid.physical_speaker_id.duplicated().any():
            raise AssertionError("Incomplete inner duration OOF predictions")
        threshold = backbone.select_threshold(valid)
        valid["seed"], valid["outer_fold"] = seed, fold
        inner.append(valid)
        model = fit_classifier(table, train, DURATION_FEATURES, seed)
        test_scores = predict(model, table, test, DURATION_FEATURES)
        test_scores["seed"], test_scores["outer_fold"] = seed, fold
        test_scores["threshold"] = threshold
        outer.append(test_scores)
    return pd.concat(outer, ignore_index=True), pd.concat(inner, ignore_index=True)


def run(prompt_path: Path = PROMPT_MAP) -> dict:
    full = pd.read_csv(FROZEN / "features/usable_metadata.csv")
    coverage = {"status": "awaiting_prompt_annotations", "source_recordings": len(full),
                "n_speakers": int(full.physical_speaker_id.nunique()),
                "annotation_path": str(prompt_path),
                "reason": "The local TORGO audio release and frozen metadata contain no verified prompt IDs or transcripts. Recording sequence numbers cannot establish shared spoken content."}
    if not prompt_path.is_file():
        write_json(DEST / "coverage.json", coverage)
        return coverage
    prompt_map = pd.read_csv(prompt_path)
    aligned = align_prompts(full, prompt_map)
    per_speaker = full.groupby("physical_speaker_id").size().rename("all_recordings").to_frame()
    per_speaker["annotated_recordings"] = aligned.groupby("physical_speaker_id").size()
    per_speaker["annotated_recordings"] = per_speaker.annotated_recordings.fillna(0).astype(int)
    write_csv(DEST / "annotation_coverage_by_speaker.csv", per_speaker.reset_index())
    coverage.update(annotated_recordings=len(aligned),
                    annotated_speakers=int(aligned.physical_speaker_id.nunique()),
                    annotation_sha256=sha256(prompt_path))
    if set(aligned.physical_speaker_id) != set(full.physical_speaker_id):
        coverage["status"] = "incomplete_speaker_annotation"
        write_json(DEST / "coverage.json", coverage)
        return coverage
    subset = select_common_prompts(aligned, set(full.physical_speaker_id))
    coverage.update(status="aligned",
                    shared_prompts=int(subset.prompt_key.nunique()), matched_recordings=len(subset),
                    retained_fraction=float(len(subset) / len(full)),
                    original_recordings_per_speaker_mean=float(len(full) / 15),
                    matched_recordings_per_speaker=int(subset.prompt_key.nunique()),
                    microphone_policy="head preferred, then session/source ID; one recording per speaker/prompt")
    coverage["matched_prompt_types"] = subset.drop_duplicates("prompt_key").prompt_type.value_counts().to_dict()
    write_json(DEST / "coverage.json", coverage)
    write_csv(DEST / "aligned_recordings.csv", aligned)
    match_counts = subset.groupby("physical_speaker_id").size().rename("matched_recordings")
    by_speaker = per_speaker.join(match_counts).fillna({"matched_recordings": 0})
    write_csv(DEST / "matching_coverage_by_speaker.csv", by_speaker.reset_index())
    if subset.prompt_key.nunique() < 2:
        coverage["status"] = "insufficient_shared_prompts"
        write_json(DEST / "coverage.json", coverage)
        return coverage
    backbone, config = load_backbone()
    write_csv(DEST / "locked_outer_checkpoints.csv", verify_outer_checkpoints())
    durations = []
    for row in subset.itertuples(index=False):
        path = Path(row.file_path)
        if not path.is_file():
            raise FileNotFoundError(path)
        durations.append(len(backbone.preprocess_waveform(path, row.mic_type, config)) / config.sample_rate)
    subset["preprocessed_seconds"] = durations
    subset["raw_seconds"] = subset.source_seconds
    subset["frozen_index"] = full.set_index("source_id").index.get_indexer(subset.source_id)
    if (subset.frozen_index < 0).any():
        raise AssertionError("Matched source outside frozen WavLM cache")
    write_csv(DEST / "matched_recordings.csv", subset)
    tables = []
    for name in ("raw", "preprocessed"):
        table = speaker_features(subset, name + "_seconds")
        table["duration_basis"] = name
        tables.append(table)
    write_csv(DEST / "speaker_duration_features.csv", pd.concat(tables, ignore_index=True))
    scores = []
    for table in tables:
        name = table.duration_basis.iloc[0]
        outer, inner = duration_oof(table, backbone)
        outer["arm"] = "duration_" + name
        inner["arm"] = "duration_" + name
        scores.append(outer)
        write_csv(DEST / f"duration_{name}_inner_oof.csv", inner)
    speech = np.load(FROZEN / "features/speech_embeddings.npy", mmap_mode="r")
    acoustic = np.load(FROZEN / "features/acoustic_features.npy", mmap_mode="r")
    positions = subset.frozen_index.to_numpy(dtype=int)
    wavlm = score_condition(backbone, config, backbone.choose_device(), subset,
                            np.asarray(speech[positions]), np.asarray(acoustic[positions]))
    wavlm["arm"] = "wavlm_matched"
    scores.append(wavlm)
    original = reference_predictions()[["seed", "outer_fold", "physical_speaker_id", "label",
                                        "probability", "threshold"]].copy()
    original["arm"] = "wavlm_original"
    scores.append(original)
    duration_original = pd.read_csv(ROOT / "v8/v8_publication_completion/results/duration/predictions.csv")
    duration_original = duration_original.query("corpus == 'TORGO'")
    duration_original["arm"] = "duration_original"
    scores.append(duration_original)
    all_scores = pd.concat(scores, ignore_index=True)
    write_csv(DEST / "speaker_scores.csv", all_scores)
    rows = []
    for (arm, seed), group in all_scores.groupby(["arm", "seed"]):
        rows.append({"arm": arm, "seed": int(seed), **metrics(group)})
    per_seed = pd.DataFrame(rows)
    write_csv(DEST / "per_seed_metrics.csv", per_seed)
    comparisons = []
    for first, second in (("duration_raw", "duration_original"),
                          ("duration_preprocessed", "duration_raw"),
                          ("wavlm_matched", "duration_raw"),
                          ("wavlm_matched", "wavlm_original")):
        paired = paired_bootstrap(all_scores[all_scores.arm == first], all_scores[all_scores.arm == second])
        paired.insert(0, "first", first)
        paired.insert(1, "second", second)
        comparisons.append(paired)
    write_csv(DEST / "paired_bootstrap.csv", pd.concat(comparisons, ignore_index=True))
    coverage["status"] = "complete"
    coverage["stacking"] = "not_estimable: matched-subset inner-fold WavLM predictions are not available; original full-recording inner scores are not interchangeable"
    write_json(DEST / "coverage.json", coverage)
    from figures import prompt_figure
    prompt_figure(per_seed, subset, DEST / "fig01_prompt_matching")
    return coverage


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-map", type=Path, default=PROMPT_MAP)
    print(run(parser.parse_args().prompt_map))
