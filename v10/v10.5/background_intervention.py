"""Speech-preserving background edits scored with frozen WavLM heads and thresholds."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import torch
from scipy.special import logit

from support import (FROZEN, OUT, ROOT, V10, checked_folds, load_backbone, load_encoder,
                     metrics, paired_bootstrap, paired_effect_bootstrap, read_json, score_condition, sha256,
                     verify_outer_checkpoints,
                     write_csv, write_json)
from v8.publication.external_diagnostics import edge_background_segment

DEST = OUT / "background_intervention"
CONDITIONS = ("original", "sham", "neutralized", "swapped", "intensity_half", "intensity_double")
CACHE_VERSION = "smooth_mask_v2"
SELECTED = {
    "TORGO": ROOT / "v8/v8_publication_completion/diagnostics/silence/TORGO_selected_recordings.csv",
    "UASpeech": ROOT / "v8/v8_publication_completion/diagnostics/silence/UASpeech_selected_recordings.csv",
}


def raw_mono(path: Path, sample_rate: int) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    mono = audio.mean(axis=1)
    if sr != sample_rate:
        mono = librosa.resample(mono, orig_sr=sr, target_sr=sample_rate)
    return np.asarray(mono, dtype=np.float32)


def quiet_mask(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    """Only low-energy 20 ms frames; leave all other speech samples bit-identical."""
    frame = round(.02 * sample_rate)
    n = len(audio) // frame
    mask = np.zeros(len(audio), dtype=bool)
    if n:
        frames = audio[:n * frame].reshape(n, frame)
        rms = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        mask[:n * frame] = np.repeat(rms <= float(np.max(np.abs(audio))) * 10 ** (-35 / 20), frame)
    return mask


def smooth_mask(mask: np.ndarray, sample_rate: int) -> np.ndarray:
    """Five-ms zero-ended fades prevent click-like changes at quiet-frame edges."""
    weight = np.zeros(len(mask), dtype=np.float64)
    transitions = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(transitions == 1)
    ends = np.flatnonzero(transitions == -1)
    fade = max(1, round(.005 * sample_rate))
    for start, end in zip(starts, ends):
        offset = np.arange(end - start)
        weight[start:end] = np.minimum(1., np.minimum(offset / fade, (end - start - 1 - offset) / fade))
    return weight


def make_conditions(audio: np.ndarray, donor: np.ndarray, sample_rate: int) -> tuple[dict, list[dict]]:
    """Edit only quiet frames in the already-trimmed, WavLM-bound waveform."""
    mask = quiet_mask(audio, sample_rate)
    weight = smooth_mask(mask, sample_rate)
    if mask.sum() < .1 * sample_rate or len(donor) == 0:
        raise ValueError("Insufficient quiet samples or donor")
    native = audio[mask].astype(np.float64)
    donor_tiled = np.resize(donor, len(audio)).astype(np.float64)
    donor_tiled -= donor_tiled.mean()
    donor_rms = np.sqrt(np.mean(donor_tiled[mask] ** 2))
    native_rms = np.sqrt(np.mean(native ** 2))
    if donor_rms < 1e-7:
        raise ValueError("Donor background is effectively silent")
    # A fixed low floor makes array-microphone gated silence testable.
    target = max(native_rms, .003)
    replacement = donor_tiled * (target / donor_rms)
    outputs = {}
    for condition in CONDITIONS:
        result = audio.copy()
        if condition == "sham":
            target_audio = audio
        elif condition == "neutralized":
            target_audio = np.zeros_like(audio)
        elif condition == "swapped":
            target_audio = replacement
        elif condition == "intensity_half":
            target_audio = audio.astype(np.float64) * .5
        elif condition == "intensity_double":
            target_audio = audio.astype(np.float64) * 2
        else:
            target_audio = audio
        if condition != "original":
            result = np.asarray(audio.astype(np.float64) + weight * (target_audio - audio), dtype=np.float32)
        if len(result) != len(audio) or not np.array_equal(result[~mask], audio[~mask]):
            raise AssertionError("Foreground or duration changed")
        if not np.isfinite(result).all() or np.max(np.abs(result)) > 1.0 + 1e-6:
            raise AssertionError("Non-finite or clipped intervention")
        outputs[condition] = result
    speech_rms = float(np.sqrt(np.mean(audio[~mask].astype(np.float64) ** 2))) if (~mask).any() else 0
    total_energy = float(np.sum(audio.astype(np.float64) ** 2))
    quiet_energy_fraction = float(np.sum(audio[mask].astype(np.float64) ** 2) / max(total_energy, 1e-12))
    diagnostics = []
    for condition, result in outputs.items():
        background_rms = float(np.sqrt(np.mean(result[mask].astype(np.float64) ** 2)))
        diagnostics.append({"condition": condition, "duration_seconds": len(result) / sample_rate,
                            "peak": float(np.max(np.abs(result))),
                            "clipped_samples": int(np.sum(np.abs(result) > 1)),
                            "quiet_fraction": float(mask.mean()),
                            "quiet_energy_fraction": quiet_energy_fraction,
                            "edited_weight_fraction": float(np.mean(weight)),
                            "max_boundary_step_added": float(np.max(np.abs(np.diff(result - audio)))) if len(audio) > 1 else 0.,
                            "quiet_rms": background_rms,
                            "snr_proxy_db": float(20 * np.log10(speech_rms / max(background_rms, 1e-9))),
                            "foreground_max_abs_difference": float(np.max(np.abs(result[~mask] - audio[~mask]))) if (~mask).any() else 0,
                            "mean_abs_waveform_change": float(np.mean(np.abs(result - audio)))})
    if outputs["swapped"].tobytes() == outputs["sham"].tobytes():
        raise AssertionError("Swap did not reach the WavLM waveform")
    return outputs, diagnostics


def select_recordings(corpus: str, n_per_speaker: int = 3) -> pd.DataFrame:
    frame = pd.read_csv(SELECTED[corpus])
    if frame.source_id.duplicated().any():
        raise ValueError("Duplicate selected source IDs")
    frame["_rank"] = frame.source_id.map(lambda x: hashlib.sha256(
        f"v10.5-background:{x}".encode()).hexdigest())
    frame = frame.sort_values(["physical_speaker_id", "_rank"])
    backbone, config = load_backbone()
    rows = []
    for speaker, group in frame.groupby("physical_speaker_id"):
        accepted = []
        for _, row in group.iterrows():
            audio = backbone.preprocess_waveform(Path(row.file_path), row.mic_type, config)
            if quiet_mask(audio, config.sample_rate).sum() >= .1 * config.sample_rate:
                accepted.append(row)
            if len(accepted) == n_per_speaker:
                break
        if len(accepted) != n_per_speaker:
            raise ValueError(f"Only {len(accepted)} WavLM-bound quiet-frame recordings for {speaker}")
        rows.append(pd.DataFrame(accepted))
    selected = pd.concat(rows, ignore_index=True).drop(columns="_rank")
    # A deterministic source rotation, without diagnosis labels, supplies cross-source noise.
    donor_ids = []
    for row in selected.itertuples(index=False):
        options = selected[(selected.physical_speaker_id != row.physical_speaker_id)
                           & (selected.mic_type == row.mic_type)]
        if options.empty:
            raise ValueError("No different-speaker donor with the same microphone")
        donor_ids.append(options.iloc[int(hashlib.sha256(row.source_id.encode()).hexdigest(), 16) % len(options)].source_id)
    selected["donor_source_id"] = donor_ids
    return selected


def extract(corpus: str, selected: pd.DataFrame, backbone, config, encoder, device) -> tuple[dict, pd.DataFrame]:
    root = DEST / corpus
    donor_lookup = selected.set_index("source_id")
    features = {c: ([], []) for c in CONDITIONS}
    diagnostics = []
    for speaker, group in selected.groupby("physical_speaker_id", sort=True):
        paths = {c: root / "features_v2" / c / f"{speaker}.npz" for c in CONDITIONS}
        if all(p.is_file() for p in paths.values()):
            cached = {}
            for condition, path in paths.items():
                with np.load(path, allow_pickle=False) as data:
                    if not np.array_equal(data["source_id"].astype(str), group.source_id.to_numpy(dtype=str)):
                        raise RuntimeError("Feature cache source IDs changed")
                    cached[condition] = (data["speech"], data["acoustic"])
            d = pd.read_csv(root / "diagnostics" / f"{speaker}.csv")
            if "quiet_energy_fraction" not in d:
                fractions = {}
                for row in group.itertuples(index=False):
                    audio = backbone.preprocess_waveform(Path(row.file_path), row.mic_type, config)
                    mask = quiet_mask(audio, config.sample_rate)
                    energy = np.square(audio.astype(np.float64))
                    fractions[row.source_id] = float(energy[mask].sum() / max(energy.sum(), 1e-12))
                d["quiet_energy_fraction"] = d.source_id.map(fractions)
                write_csv(root / "diagnostics" / f"{speaker}.csv", d)
        else:
            blocks = {c: ([], []) for c in CONDITIONS}
            records = []
            for row in group.itertuples(index=False):
                path = Path(row.file_path)
                original = backbone.preprocess_waveform(path, row.mic_type, config)
                donor_row = donor_lookup.loc[row.donor_source_id]
                donor_raw = raw_mono(Path(donor_row.file_path), config.sample_rate)
                edge = edge_background_segment(donor_raw, config.sample_rate)
                if edge is None:
                    raise ValueError("Saved donor has no valid background edge")
                waveforms, checks = make_conditions(original, donor_raw[edge[0]:edge[1]], config.sample_rate)
                for check in checks:
                    check.update(source_id=row.source_id, physical_speaker_id=speaker,
                                 donor_source_id=row.donor_source_id, corpus=corpus)
                records.extend(checks)
                for condition, audio in waveforms.items():
                    blocks[condition][0].append(backbone._encode_batch(encoder, [audio], config, device)[0])
                    blocks[condition][1].append(backbone.acoustic_features(audio, config.sample_rate))
            cached = {c: (np.stack(v[0]), np.stack(v[1])) for c, v in blocks.items()}
            for condition, path in paths.items():
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_suffix(".npz.tmp")
                with temp.open("wb") as handle:
                    np.savez(handle, source_id=group.source_id.to_numpy(dtype=str),
                             speech=cached[condition][0], acoustic=cached[condition][1])
                temp.replace(path)
            d = pd.DataFrame(records)
            check_path = root / "diagnostics" / f"{speaker}.csv"
            write_csv(check_path, d)
            print(f"{corpus} {speaker}: {len(group)} recordings x {len(CONDITIONS)} conditions", flush=True)
        diagnostics.append(d)
        for condition in CONDITIONS:
            features[condition][0].append(cached[condition][0])
            features[condition][1].append(cached[condition][1])
    return {c: (np.concatenate(features[c][0]), np.concatenate(features[c][1]))
            for c in CONDITIONS}, pd.concat(diagnostics, ignore_index=True)


def score_external(backbone, config, device, meta, speech, acoustic) -> pd.DataFrame:
    lock = read_json(V10 / "deployment/v8_torgo_model_lock.json")
    rows = []
    for record in lock["models"]:
        folder = V10 / record["checkpoint"]
        scaler_path = V10 / record["acoustic_scaler"]
        if sha256(folder) != record["checkpoint_sha256"] or sha256(scaler_path) != record["acoustic_scaler_sha256"]:
            raise RuntimeError("Frozen external checkpoint changed")
        model = backbone._new_head(config, speech, acoustic, False, device)
        model.load_state_dict(torch.load(folder, map_location=device, weights_only=True))
        with np.load(scaler_path) as scale:
            scaler = (scale["mean"], scale["scale"])
        store = backbone.BagStore(meta, speech, acoustic, scaler, device)
        pred = backbone.predict_speakers(model, store, set(meta.physical_speaker_id),
                                         config.max_utterances_per_bag)
        pred["seed"] = int(record["seed"])
        pred["threshold"] = float(record["threshold"])
        rows.append(pred)
    return pd.concat(rows, ignore_index=True)


def run_corpus(corpus: str) -> dict:
    write_csv(DEST / "locked_outer_checkpoints.csv", verify_outer_checkpoints())
    selected = select_recordings(corpus)
    destination = DEST / corpus
    manifest_path = destination / "subset_manifest.csv"
    if manifest_path.is_file():
        prior = pd.read_csv(manifest_path)
        if not prior.source_id.equals(selected.source_id) or not prior.donor_source_id.equals(selected.donor_source_id):
            raise RuntimeError("Background subset or donors changed")
    write_csv(manifest_path, selected)
    backbone, config = load_backbone()
    device = backbone.choose_device()
    encoder = load_encoder(backbone, config, device)
    features, diagnostics = extract(corpus, selected, backbone, config, encoder, device)
    write_csv(destination / "waveform_diagnostics.csv", diagnostics)
    if diagnostics.foreground_max_abs_difference.max() != 0 or diagnostics.clipped_samples.sum() != 0:
        raise AssertionError("Signal integrity gate failed")
    if not np.allclose(diagnostics.groupby("source_id").duration_seconds.max(),
                       diagnostics.groupby("source_id").duration_seconds.min()):
        raise AssertionError("Intervention duration changed")
    scored = []
    for condition, (speech, acoustic) in features.items():
        score = (score_condition(backbone, config, device, selected, speech, acoustic)
                 if corpus == "TORGO" else score_external(backbone, config, device, selected, speech, acoustic))
        score["condition"] = condition
        score["predicted_label"] = (score.probability >= score.threshold).astype(int)
        score["logit"] = logit(score.probability.clip(1e-6, 1 - 1e-6))
        scored.append(score)
    all_scores = pd.concat(scored, ignore_index=True)
    write_csv(destination / "speaker_scores_by_condition.csv", all_scores)
    per_seed = pd.DataFrame([{"condition": condition, "seed": int(seed), **metrics(group)}
                             for (condition, seed), group in all_scores.groupby(["condition", "seed"])])
    write_csv(destination / "per_seed_metrics.csv", per_seed)
    sham = all_scores.query("condition == 'sham'")
    comparisons = []
    shifts = []
    for condition in CONDITIONS:
        if condition == "sham":
            continue
        part = all_scores[all_scores.condition == condition]
        paired = paired_bootstrap(part, sham)
        paired.insert(0, "condition", condition)
        comparisons.append(paired)
        merged = part.merge(sham, on=["seed", "physical_speaker_id", "label"],
                            suffixes=("", "_sham"), validate="one_to_one")
        merged["condition"] = condition
        merged["delta_logit"] = merged.logit - merged.logit_sham
        merged["delta_probability"] = merged.probability - merged.probability_sham
        merged["decision_changed"] = merged.predicted_label != merged.predicted_label_sham
        shifts.append(merged[["condition", "seed", "physical_speaker_id", "label",
                              "delta_logit", "delta_probability", "decision_changed"]])
    write_csv(destination / "paired_bootstrap.csv", pd.concat(comparisons, ignore_index=True))
    shifts = pd.concat(shifts, ignore_index=True)
    write_csv(destination / "paired_score_changes.csv", shifts)
    effect_intervals = []
    for condition, part in shifts.groupby("condition"):
        interval = paired_effect_bootstrap(part)
        interval.insert(0, "condition", condition)
        effect_intervals.append(interval)
    write_csv(destination / "paired_effect_bootstrap.csv", pd.concat(effect_intervals, ignore_index=True))
    speaker_effects = shifts.groupby(["condition", "physical_speaker_id", "label"], as_index=False).agg(
        delta_logit=("delta_logit", "mean"), delta_probability=("delta_probability", "mean"),
        locked_decision_change_rate=("decision_changed", "mean"))
    write_csv(destination / "speaker_effects.csv", speaker_effects)
    summary = {"corpus": corpus, "n_speakers": int(selected.physical_speaker_id.nunique()),
               "recordings_per_speaker": 3, "conditions": CONDITIONS,
               "mean_metrics": per_seed.groupby("condition")[list(metrics(sham.query('seed == 42')).keys())].mean().to_dict("index"),
               "mean_abs_logit_shift_vs_sham": shifts.groupby("condition").delta_logit.apply(lambda x: float(np.mean(np.abs(x)))).to_dict(),
               "locked_decision_changes_vs_sham": shifts.groupby("condition").decision_changed.sum().astype(int).to_dict(),
               "caveat": "Quiet-edge segments and low-energy frames may contain speech. SNR is a speech/quiet-frame proxy, not a ground-truth separation estimate. Three recordings per speaker limit power.",
               "interpretation_rule": "Compare each condition with sham; claim background sensitivity only when score shifts exceed sham and integrity gates pass."
               }
    write_json(destination / "summary.json", summary)
    from figures import background_figure
    background_figure(per_seed, speaker_effects, destination / "fig01_background_interventions")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", choices=("TORGO", "UASpeech", "all"), nargs="?", default="all")
    arg = parser.parse_args()
    for corpus in (SELECTED if arg.corpus == "all" else (arg.corpus,)):
        print(run_corpus(corpus), flush=True)
