"""Mean-preserving duration-spread intervention on locked V10 outer-fold heads."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.special import logit
from sklearn.metrics import roc_auc_score

from common import (
    FROZEN, HERE, OUT, ROOT, SEEDS, V10, checked_folds, load_backbone,
    read_json, reference_predictions, sha256, write_csv, write_json,
)

SPREAD_OUT = OUT / "spread"
ALPHAS = {"spread_0.88": 0.88, "spread_1.12": 1.12}
CONDITIONS = ("original", "identity", "spread_0.88", "spread_1.12", "roundtrip")


def local_audio_path(row, backbone, config) -> Path:
    path = Path(row.file_path)
    if not path.is_file():
        path = backbone.resolve_dataset_root(config) / row.relative_path
    if not path.is_file():
        raise FileNotFoundError(f"TORGO audio missing: {row.relative_path}")
    return path


def select_subset(backbone, config, full_meta: pd.DataFrame) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """Hash-selected 50/speaker after label-blind source and processed-duration gates."""
    candidates = full_meta[full_meta.source_seconds.between(1.0, 6.0, inclusive="both")].copy()
    candidates["_rank"] = candidates.source_id.map(
        lambda source: hashlib.sha256(f"v10.4-spread:{source}".encode()).hexdigest()
    )
    chosen, waveforms = [], []
    for speaker, group in candidates.sort_values("_rank").groupby("physical_speaker_id", sort=True):
        accepted = 0
        for index, row in group.iterrows():
            audio = backbone.preprocess_waveform(local_audio_path(row, backbone, config), row.mic_type, config)
            duration = len(audio) / config.sample_rate
            if not 1.0 <= duration <= 6.0:
                continue
            record = row.drop(labels=["_rank"]).to_dict()
            record["frozen_index"] = int(index)
            record["preprocessed_seconds"] = float(duration)
            chosen.append(record)
            waveforms.append(audio)
            accepted += 1
            if accepted == 50:
                break
        if accepted != 50:
            raise AssertionError(f"Only {accepted} eligible recordings for speaker {speaker}")
    subset = pd.DataFrame(chosen).reset_index(drop=True)
    if len(subset) != 750 or subset.physical_speaker_id.nunique() != 15:
        raise AssertionError("Spread subset is incomplete")
    group_mean = subset.groupby("physical_speaker_id").preprocessed_seconds.transform("mean")
    subset["speaker_mean_seconds"] = group_mean
    for name, alpha in ALPHAS.items():
        target = group_mean + alpha * (subset.preprocessed_seconds - group_mean)
        subset[f"target_seconds_{name}"] = target
        subset[f"rate_{name}"] = subset.preprocessed_seconds / target
        if not np.all((subset[f"rate_{name}"] >= 0.8) & (subset[f"rate_{name}"] <= 1.25)):
            raise AssertionError("Intervention exceeds the predeclared 0.8-1.25 rate range")
        means = subset.groupby("physical_speaker_id")[f"target_seconds_{name}"].mean()
        originals = subset.groupby("physical_speaker_id").preprocessed_seconds.mean()
        if not np.allclose(means, originals, atol=1e-12, rtol=0):
            raise AssertionError("Spread target changed speaker mean duration")
    manifest = subset[[
        "source_id", "physical_speaker_id", "label", "mic_type", "frozen_index",
        "source_seconds", "preprocessed_seconds", "speaker_mean_seconds",
        "target_seconds_spread_0.88", "rate_spread_0.88",
        "target_seconds_spread_1.12", "rate_spread_1.12",
    ]]
    previous = SPREAD_OUT / "subset_manifest.csv"
    if previous.is_file():
        old = pd.read_csv(previous)
        if not old.source_id.equals(manifest.source_id):
            raise RuntimeError("Existing V10.4 spread subset differs; refusing to overwrite cached features")
    write_csv(previous, manifest)
    return subset, waveforms


def transformed_waveform(audio: np.ndarray, mic_type: str, rate: float, condition: str,
                         config) -> tuple[np.ndarray, bool]:
    if condition == "identity":
        transformed = librosa.effects.time_stretch(audio, rate=1.0)
    elif condition == "roundtrip":
        transformed = librosa.effects.time_stretch(audio, rate=float(rate))
        transformed = librosa.effects.time_stretch(transformed, rate=float(1.0 / rate))
        transformed = librosa.util.fix_length(transformed, size=len(audio))
    else:
        transformed = librosa.effects.time_stretch(audio, rate=float(rate))
    peak = float(np.max(np.abs(transformed)))
    if peak > 0:
        transformed = transformed / peak
    if mic_type == "array":
        transformed = np.where(np.abs(transformed) < .01, 0.0, transformed)
    maximum = int(config.max_seconds * config.sample_rate)
    cropped = len(transformed) > maximum
    if cropped:
        start = (len(transformed) - maximum) // 2
        transformed = transformed[start:start + maximum]
    if len(transformed) < 1024:
        transformed = np.pad(transformed, (0, 1024 - len(transformed)))
    transformed = np.asarray(transformed, dtype=np.float32)
    if not np.isfinite(transformed).all():
        raise ValueError("Non-finite transformed waveform")
    return transformed, cropped


def load_encoder(backbone, config, device):
    weight_source = ROOT / "v8/v8_publication_completion/frozen_reference/pretrained/wavlm_base.pth"
    folder = HERE / "weights/WAVLM_BASE"
    folder.mkdir(parents=True, exist_ok=True)
    link = folder / "wavlm_base.pth"
    if not link.exists():
        link.symlink_to(weight_source)
    if link.resolve() != weight_source.resolve():
        raise AssertionError("V10.4 weight link points to a different checkpoint")
    encoder, weight_hash, _ = backbone.load_pretrained(config, folder, device)
    lock = read_json(FROZEN / "deployment/v8_torgo_model_lock.json")
    if weight_hash != lock["pretrained_weight_sha256"]:
        raise AssertionError("WavLM weights differ from the frozen reference")
    return encoder


def extract_conditions(backbone, config, encoder, device, subset: pd.DataFrame,
                       waveforms: list[np.ndarray], frozen_speech, frozen_acoustic) -> tuple[dict, pd.DataFrame]:
    features = {
        "original": (
            np.asarray(frozen_speech[subset.frozen_index.to_numpy(dtype=int)]),
            np.asarray(frozen_acoustic[subset.frozen_index.to_numpy(dtype=int)]),
        )
    }
    records = []
    for condition in CONDITIONS[1:]:
        all_speech, all_acoustic = [], []
        for speaker, group in subset.groupby("physical_speaker_id", sort=True):
            positions = group.index.to_numpy(dtype=int)
            shard_dir = SPREAD_OUT / "features" / condition
            shard_dir.mkdir(parents=True, exist_ok=True)
            shard = shard_dir / f"{speaker}.npz"
            expected_ids = group.source_id.to_numpy(dtype=str)
            if shard.is_file():
                with np.load(shard, allow_pickle=False) as stored:
                    if not np.array_equal(stored["source_id"].astype(str), expected_ids):
                        raise RuntimeError(f"Spread cache source IDs changed: {shard}")
                    speech = stored["speech"]
                    acoustic = stored["acoustic"]
                    achieved = stored["achieved_seconds"]
                    cropped = stored["cropped"]
            else:
                speech_rows, acoustic_rows, achieved_rows, cropped_rows = [], [], [], []
                start = time.monotonic()
                for index in positions:
                    row = subset.loc[index]
                    rate = 1.0
                    if condition in ALPHAS:
                        rate = float(row[f"rate_{condition}"])
                    elif condition == "roundtrip":
                        rate = float(row["rate_spread_0.88"])
                    audio, was_cropped = transformed_waveform(
                        waveforms[index], row.mic_type, rate, condition, config
                    )
                    speech_rows.append(backbone._encode_batch(encoder, [audio], config, device)[0])
                    acoustic_rows.append(backbone.acoustic_features(audio, config.sample_rate))
                    achieved_rows.append(len(audio) / config.sample_rate)
                    cropped_rows.append(was_cropped)
                speech = np.stack(speech_rows)
                acoustic = np.stack(acoustic_rows)
                achieved = np.asarray(achieved_rows)
                cropped = np.asarray(cropped_rows)
                temporary = shard.with_name(shard.name + ".tmp")
                with temporary.open("wb") as handle:
                    np.savez(handle, source_id=expected_ids, speech=speech, acoustic=acoustic,
                             achieved_seconds=achieved, cropped=cropped)
                temporary.replace(shard)
                print(f"{condition} {speaker}: 50 recordings in {time.monotonic()-start:.1f}s", flush=True)
            if speech.shape != (50, 4, 1536) or acoustic.shape != (50, 39):
                raise AssertionError(f"Bad spread feature shard {shard}")
            for i, index in enumerate(positions):
                original = float(subset.loc[index, "preprocessed_seconds"])
                target = (float(subset.loc[index, f"target_seconds_{condition}"])
                          if condition in ALPHAS else original)
                records.append({
                    "condition": condition, "source_id": str(subset.loc[index, "source_id"]),
                    "physical_speaker_id": speaker, "original_seconds": original,
                    "target_seconds": target, "achieved_seconds": float(achieved[i]),
                    "cropped": bool(cropped[i]),
                })
            all_speech.append(speech)
            all_acoustic.append(acoustic)
        features[condition] = (np.concatenate(all_speech), np.concatenate(all_acoustic))
    duration = pd.DataFrame(records)
    write_csv(SPREAD_OUT / "achieved_durations.csv", duration)
    return features, duration


def score_condition(backbone, config, device, meta: pd.DataFrame,
                    speech: np.ndarray, acoustic: np.ndarray) -> pd.DataFrame:
    results = []
    for seed, outer, folder, audit in checked_folds():
        model = backbone._new_head(config, speech, acoustic, False, device)
        model.load_state_dict(torch.load(folder / "final/best.pt", map_location=device, weights_only=True))
        with np.load(folder / "final/acoustic_scaler.npz") as scale:
            scaler = (scale["mean"], scale["scale"])
        store = backbone.BagStore(meta, speech, acoustic, scaler, device)
        pred = backbone.predict_speakers(model, store, set(audit["test_physical_speakers"]),
                                         config.max_utterances_per_bag)
        pred["seed"] = seed
        pred["outer_fold"] = outer
        pred["threshold"] = float(audit["selected_threshold"])
        results.append(pred)
    frame = pd.concat(results, ignore_index=True)
    if len(frame) != 75:
        raise AssertionError("Spread scoring did not cover every speaker and seed")
    return frame


def summarize(scores: pd.DataFrame, durations: pd.DataFrame, subset: pd.DataFrame) -> dict:
    pivot = scores.pivot(index=["seed", "physical_speaker_id", "label"], columns="condition", values="logit")
    if pivot.shape != (75, 5):
        raise AssertionError("Spread scores are incomplete")
    pivot["spread_delta"] = pivot["spread_1.12"] - pivot["spread_0.88"]
    pivot["artifact_abs"] = np.abs(pivot["roundtrip"] - pivot["identity"])
    pivot["spread_abs"] = np.abs(pivot["spread_delta"])
    table = pivot.reset_index()
    write_csv(SPREAD_OUT / "paired_logit_scores.csv", table)
    by_speaker = table.groupby(["physical_speaker_id", "label"], as_index=False).agg(
        spread_delta=("spread_delta", "mean"), artifact_abs=("artifact_abs", "mean"),
        spread_abs=("spread_abs", "mean"),
    )
    write_csv(SPREAD_OUT / "speaker_effects.csv", by_speaker)
    rng = np.random.default_rng(20261021)
    classes = {c: by_speaker.loc[by_speaker.label == c, "spread_delta"].to_numpy() for c in (0, 1)}
    boots = np.array([
        np.concatenate([rng.choice(classes[c], len(classes[c]), replace=True) for c in (0, 1)]).mean()
        for _ in range(2000)
    ])
    ci = np.percentile(boots, [2.5, 97.5])
    auc_rows = []
    for condition, group in scores.groupby("condition"):
        for seed, seed_group in group.groupby("seed"):
            auc_rows.append({"condition": condition, "seed": int(seed),
                             "roc_auc": float(roc_auc_score(seed_group.label, seed_group.probability))})
    auc_table = pd.DataFrame(auc_rows)
    write_csv(SPREAD_OUT / "per_seed_auc.csv", auc_table)
    group_gaps = {}
    for condition in CONDITIONS:
        group_gaps[condition] = float(
            table.loc[table.label == 1, condition].mean() - table.loc[table.label == 0, condition].mean()
        )
    duration_stats = []
    for speaker, group in subset.groupby("physical_speaker_id"):
        original = group.preprocessed_seconds.to_numpy()
        for condition in ("original", *ALPHAS):
            target = original if condition == "original" else group[f"target_seconds_{condition}"].to_numpy()
            duration_stats.append({
                "physical_speaker_id": speaker, "condition": condition,
                "mean_seconds": float(target.mean()), "sd_seconds": float(target.std(ddof=0)),
                "sd_ratio_to_original": float(target.std(ddof=0) / original.std(ddof=0)),
            })
    write_csv(SPREAD_OUT / "duration_spread_by_speaker.csv", pd.DataFrame(duration_stats))
    artifact = float(table.artifact_abs.median())
    spread = float(table.spread_abs.median())
    gate_passes = bool(artifact < 0.5 * spread)
    if not gate_passes:
        verdict = "inconclusive_artifact_control"
    elif ci[0] > 0 or ci[1] < 0:
        verdict = "spread_sensitive_with_artifact_gate_passed"
    else:
        verdict = "no_clear_spread_effect"
    max_duration_error = float(np.max(np.abs(durations.achieved_seconds - durations.target_seconds)))
    summary = {
        "mean_spread_1_12_minus_0_88_logit": float(by_speaker.spread_delta.mean()),
        "physical_speaker_bootstrap_95": [float(ci[0]), float(ci[1])],
        "median_abs_spread_logit_shift": spread,
        "median_abs_roundtrip_artifact_logit_shift": artifact,
        "artifact_gate_passed": gate_passes,
        "verdict": verdict,
        "mean_auc_by_condition": {
            c: float(auc_table.loc[auc_table.condition == c, "roc_auc"].mean()) for c in CONDITIONS
        },
        "group_logit_gap_by_condition": group_gaps,
        "n_cropped_by_condition": {c: int(x.cropped.sum()) for c, x in durations.groupby("condition")},
        "maximum_absolute_target_duration_error_seconds": max_duration_error,
        "n_speakers": 15, "n_recordings_per_speaker": 50,
        "note": "Phase-vocoder processing may change acoustic cues besides duration spread; the artifact gate is necessary but not sufficient for a causal interpretation.",
    }
    write_json(SPREAD_OUT / "summary.json", summary)
    return summary


def main() -> None:
    backbone, config = load_backbone()
    device = backbone.choose_device()
    full_meta = pd.read_csv(FROZEN / "features/usable_metadata.csv")
    frozen_speech = np.load(FROZEN / "features/speech_embeddings.npy", mmap_mode="r")
    frozen_acoustic = np.load(FROZEN / "features/acoustic_features.npy", mmap_mode="r")
    subset, waveforms = select_subset(backbone, config, full_meta)
    encoder = load_encoder(backbone, config, device)
    check_indices = subset.groupby("physical_speaker_id", sort=True).head(2).index.to_numpy()
    extracted = np.stack([
        backbone._encode_batch(encoder, [waveforms[i]], config, device)[0] for i in check_indices
    ])
    frozen = np.asarray(frozen_speech[subset.iloc[check_indices].frozen_index.to_numpy(dtype=int)])
    relative_difference = float(np.max(np.abs(extracted - frozen)) / np.max(np.abs(frozen)))
    if relative_difference >= 1e-3:
        raise AssertionError("Unmodified audio does not reproduce frozen WavLM features")
    print("Frozen feature reproduction relative max difference:", relative_difference, flush=True)
    full_scores = score_condition(backbone, config, device, full_meta, frozen_speech, frozen_acoustic)
    expected = reference_predictions()
    paired = full_scores.merge(expected, on=["seed", "outer_fold", "physical_speaker_id"],
                               suffixes=("", "_expected"), validate="one_to_one")
    max_score_diff = float(np.max(np.abs(paired.probability - paired.probability_expected)))
    if len(paired) != 75 or max_score_diff >= 1e-6:
        raise AssertionError("Loaded fold heads do not reproduce saved V10 predictions")
    print("Locked model reproduction max probability difference:", max_score_diff, flush=True)
    write_json(SPREAD_OUT / "pipeline_checks.json", {
        "reextraction_relative_max_difference": relative_difference,
        "locked_scoring_max_probability_difference": max_score_diff,
        "pretrained_weights_sha256": sha256(ROOT / "v8/v8_publication_completion/frozen_reference/pretrained/wavlm_base.pth"),
    })
    features, achieved = extract_conditions(backbone, config, encoder, device, subset, waveforms,
                                            frozen_speech, frozen_acoustic)
    scores = []
    for condition in CONDITIONS:
        speech, acoustic = features[condition]
        part = score_condition(backbone, config, device, subset, speech, acoustic)
        part["condition"] = condition
        part["logit"] = logit(part.probability.clip(1e-6, 1 - 1e-6))
        scores.append(part)
    scores = pd.concat(scores, ignore_index=True)
    write_csv(SPREAD_OUT / "speaker_scores_by_condition.csv", scores)
    print("Spread:", summarize(scores, achieved, subset), flush=True)


if __name__ == "__main__":
    main()
