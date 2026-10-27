"""Nested speaker-held-out head trained only on cached background-edge WavLM features."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from common import (
    FROZEN, HERE, OUT, ROOT, SEEDS, V10, checked_folds, load_backbone,
    mean_seed_auc, paired_auc_bootstrap, plan_hash, read_json, sha256, write_csv,
    write_json,
)

BG_FEATURES = ROOT / "v8/v8_publication_completion/external_cache/silence_TORGO/features"
BG_OUT = OUT / "background"


def background_arrays() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    manifest = read_json(BG_FEATURES / "manifest.json")
    if not manifest.get("complete") or manifest.get("extraction_errors"):
        raise RuntimeError("Background-edge feature cache is incomplete")
    meta = pd.read_csv(BG_FEATURES / "usable_metadata.csv")
    speech = np.load(BG_FEATURES / "speech_embeddings.npy", mmap_mode="r")
    acoustic = np.load(BG_FEATURES / "acoustic_features.npy", mmap_mode="r")
    if (len(meta), speech.shape, acoustic.shape) != (750, (750, 4, 1536), (750, 39)):
        raise AssertionError("Unexpected saved background feature dimensions")
    if not meta.groupby("physical_speaker_id").size().eq(50).all():
        raise AssertionError("Expected exactly 50 selected clips per TORGO speaker")
    if meta.source_id.duplicated().any() or meta.physical_speaker_id.nunique() != 15:
        raise AssertionError("Background source IDs or speakers are inconsistent")
    return meta, speech, acoustic


def check_splits(backbone, meta: pd.DataFrame) -> None:
    by_key = {(s, o): a for s, o, _, a in checked_folds()}
    for seed in SEEDS:
        for outer, train, test in backbone.grouped_speaker_splits(meta, seed, 5):
            original = by_key[(seed, outer)]
            if train != set(original["train_physical_speakers"]) or test != set(original["test_physical_speakers"]):
                raise AssertionError("Background outer folds differ from V10")
            subset = meta[meta.physical_speaker_id.isin(train)]
            for inner, inner_train, validation in backbone.grouped_speaker_splits(subset, seed + 731, 3):
                stored = original["inner_folds"][inner]
                if (inner_train != set(stored["train_physical_speakers"])
                        or validation != set(stored["validation_physical_speakers"])):
                    raise AssertionError("Background inner folds differ from V10")


def train_background() -> pd.DataFrame:
    backbone, config = load_backbone()
    meta, speech, acoustic = background_arrays()
    check_splits(backbone, meta)
    device = backbone.choose_device()
    run_signature = hashlib.sha256(json.dumps({
        "version": "v10.4-background-1",
        "plan_sha256": plan_hash(),
        "feature_manifest_sha256": sha256(BG_FEATURES / "manifest.json"),
        "source_ids_sha256": hashlib.sha256("\n".join(meta.source_id).encode()).hexdigest(),
    }, sort_keys=True).encode()).hexdigest()
    model_dir = BG_OUT / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for seed, outer, _, audit in checked_folds():
        prediction = backbone._run_outer_fold(
            config, meta, speech, acoustic, seed=seed, outer=outer,
            train_speakers=set(audit["train_physical_speakers"]),
            test_speakers=set(audit["test_physical_speakers"]),
            use_fusion=False, run_dir=model_dir, run_signature=run_signature,
            device=device,
        )
        results.append(prediction)
        if outer == 4:
            partial = pd.concat(results, ignore_index=True)
            write_csv(BG_OUT / "background_outer_scores_partial.csv", partial)
            print(f"Background classifier completed seed {seed}; {len(partial)}/75 predictions", flush=True)
    frame = pd.concat(results, ignore_index=True)
    if len(frame) != 75 or frame.groupby("seed").physical_speaker_id.nunique().ne(15).any():
        raise AssertionError("Incomplete background classifier predictions")
    write_csv(BG_OUT / "background_outer_scores.csv", frame)
    return frame


def matched_speech_scores() -> pd.DataFrame:
    backbone, config = load_backbone()
    bg_meta, _, _ = background_arrays()
    full_meta = pd.read_csv(FROZEN / "features/usable_metadata.csv")
    lookup = pd.Series(np.arange(len(full_meta)), index=full_meta.source_id)
    if lookup.index.duplicated().any() or not set(bg_meta.source_id).issubset(set(lookup.index)):
        raise AssertionError("Selected background recordings do not match frozen speech features")
    positions = lookup.loc[bg_meta.source_id].to_numpy(dtype=int)
    speech = np.asarray(np.load(FROZEN / "features/speech_embeddings.npy", mmap_mode="r")[positions])
    acoustic = np.asarray(np.load(FROZEN / "features/acoustic_features.npy", mmap_mode="r")[positions])
    if not np.array_equal(full_meta.iloc[positions].physical_speaker_id.to_numpy(), bg_meta.physical_speaker_id.to_numpy()):
        raise AssertionError("Source ID speaker mismatch")
    device = backbone.choose_device()
    results = []
    for seed, outer, folder, audit in checked_folds():
        model = backbone._new_head(config, speech, acoustic, False, device)
        model.load_state_dict(torch.load(folder / "final/best.pt", map_location=device, weights_only=True))
        with np.load(folder / "final/acoustic_scaler.npz") as scale:
            scaler = (scale["mean"], scale["scale"])
        store = backbone.BagStore(bg_meta, speech, acoustic, scaler, device)
        pred = backbone.predict_speakers(
            model, store, set(audit["test_physical_speakers"]), config.max_utterances_per_bag
        )
        pred["seed"] = seed
        pred["outer_fold"] = outer
        pred["threshold"] = float(audit["selected_threshold"])
        pred["predicted_label"] = (pred.probability >= pred.threshold).astype(int)
        results.append(pred)
    frame = pd.concat(results, ignore_index=True)
    write_csv(BG_OUT / "matched_speech_outer_scores.csv", frame)
    return frame


def summarize(background: pd.DataFrame, matched: pd.DataFrame) -> dict:
    comparison = paired_auc_bootstrap(background, matched)
    rows = []
    for arm, frame in (("background_trained", background), ("locked_speech_same_recordings", matched)):
        for seed, group in frame.groupby("seed"):
            rows.append({
                "arm": arm, "seed": int(seed),
                "roc_auc": float(backbone_auc(group)),
                "balanced_accuracy": float(backbone_ba(group)),
            })
    write_csv(BG_OUT / "per_seed_metrics.csv", pd.DataFrame(rows))
    summary = {
        "background_mean_auc": mean_seed_auc(background),
        "matched_speech_mean_auc": mean_seed_auc(matched),
        "background_minus_matched_speech_auc": comparison,
        "n_torgo_speakers": 15, "segments_per_speaker": 50,
        "caveat": "The selected low-energy edges can contain speech. A background-only score is evidence about the selected segments, not proof of a recording-artifact mechanism.",
    }
    write_json(BG_OUT / "summary.json", summary)
    return summary


def backbone_auc(frame: pd.DataFrame) -> float:
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(frame.label, frame.probability))


def backbone_ba(frame: pd.DataFrame) -> float:
    from sklearn.metrics import balanced_accuracy_score
    return float(balanced_accuracy_score(frame.label, frame.predicted_label))


def main() -> None:
    background = train_background()
    matched = matched_speech_scores()
    print("Background:", summarize(background, matched), flush=True)


if __name__ == "__main__":
    main()
