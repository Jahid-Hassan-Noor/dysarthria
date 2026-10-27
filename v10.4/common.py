"""Read-only source adapters and shared V10.4 reporting helpers."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "outputs"
FROZEN = ROOT / "v8/v8_publication_completion/frozen_reference/run"
V10 = ROOT / "v10/outputs/reference_training"
V8_DURATION = ROOT / "v8/v8_publication_completion/baselines/duration"
SEEDS = (42, 43, 44, 45, 46)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def inside_v104(path: Path) -> Path:
    path = Path(path).resolve()
    if path != HERE and HERE not in path.parents:
        raise PermissionError(f"V10.4 refuses to write outside {HERE}: {path}")
    return path


def write_json(path: Path, data: object) -> None:
    path = inside_v104(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path = inside_v104(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def plan_hash() -> str:
    return sha256(HERE / "analysis_plan.json")


def load_backbone():
    # Definitions are executed in memory. The frozen notebook and its caches
    # remain untouched; every training run supplies an explicit V10.4 path.
    from v8.publication.common import load_pipeline

    notebook = ROOT / "v8/v8_publication_completion/frozen_reference/source/dysarthria_v8.ipynb"
    backbone = load_pipeline(notebook)
    config = backbone.Config()
    if (config.encoder_name, config.use_fusion, tuple(config.seeds)) != (
        "WAVLM_BASE", False, SEEDS
    ):
        raise RuntimeError("The frozen speech-only WavLM configuration is required")
    return backbone, config


def reference_predictions() -> pd.DataFrame:
    frame = pd.read_csv(V10 / "v10_reference_speaker_oof_predictions.csv")
    needed = {"seed", "outer_fold", "physical_speaker_id", "label", "probability", "threshold"}
    if not needed.issubset(frame) or len(frame) != 75:
        raise AssertionError("Incomplete V10 reference predictions")
    if frame.duplicated(["seed", "physical_speaker_id"]).any():
        raise AssertionError("Repeated physical speaker within seed")
    return frame


def checked_folds() -> list[tuple[int, int, Path, dict]]:
    records = []
    reference = reference_predictions()
    for seed in SEEDS:
        for outer in range(5):
            folder = V10 / f"folds/speech_only/seed{seed}_outer{outer}"
            audit = read_json(folder / "fold_audit.json")
            train = set(audit["train_physical_speakers"])
            test = set(audit["test_physical_speakers"])
            inner = pd.read_csv(folder / "inner_oof_speaker_predictions.csv")
            outer_frame = pd.read_csv(folder / "outer_speaker_predictions.csv")
            if not audit["complete"] or train & test or len(train) != 12 or len(test) != 3:
                raise AssertionError(f"Bad split: {folder}")
            if set(inner.physical_speaker_id) != train or inner.physical_speaker_id.duplicated().any():
                raise AssertionError(f"Incomplete inner OOF: {folder}")
            if set(outer_frame.physical_speaker_id) != test:
                raise AssertionError(f"Incomplete outer predictions: {folder}")
            original = reference[(reference.seed == seed) & (reference.outer_fold == outer)]
            if set(original.physical_speaker_id) != test:
                raise AssertionError(f"Published fold mismatch: {folder}")
            records.append((seed, outer, folder, audit))
    return records


def mean_seed_auc(frame: pd.DataFrame, column: str = "probability") -> float:
    return float(np.mean([
        roc_auc_score(group.label, group[column]) for _, group in frame.groupby("seed")
    ]))


def mean_seed_ba(frame: pd.DataFrame, column: str = "predicted_label") -> float:
    return float(np.mean([
        balanced_accuracy_score(group.label, group[column]) for _, group in frame.groupby("seed")
    ]))


def paired_auc_bootstrap(
    first: pd.DataFrame, second: pd.DataFrame, *,
    first_probability: str = "probability", second_probability: str = "probability",
    draws: int = 2000, seed: int = 20261021,
) -> dict:
    """One diagnosis-stratified speaker draw shared by both arms and all seeds."""
    keys = ["seed", "physical_speaker_id", "label"]
    merged = first[keys + [first_probability]].merge(
        second[keys + [second_probability]], on=keys, suffixes=("_first", "_second"),
        validate="one_to_one",
    )
    if len(merged) != 75:
        raise AssertionError("Paired comparison needs 75 matched speaker/seed rows")
    first_col = first_probability + "_first" if first_probability == second_probability else first_probability
    second_col = second_probability + "_second" if first_probability == second_probability else second_probability
    ids = merged[["physical_speaker_id", "label"]].drop_duplicates()
    if ids.physical_speaker_id.duplicated().any() or len(ids) != 15:
        raise AssertionError("Expected 15 consistent physical speakers")
    classes = {
        int(label): ids.loc[ids.label == label, "physical_speaker_id"].to_numpy()
        for label in (0, 1)
    }
    by_seed = {int(s): x.set_index("physical_speaker_id") for s, x in merged.groupby("seed")}
    expected = set(ids.physical_speaker_id)
    if any(set(x.index) != expected for x in by_seed.values()):
        raise AssertionError("Each seed must cover the same 15 physical speakers")
    point = float(np.mean([
        roc_auc_score(x.label, x[first_col]) - roc_auc_score(x.label, x[second_col])
        for x in by_seed.values()
    ]))
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    first_values = np.empty(draws, dtype=float)
    second_values = np.empty(draws, dtype=float)
    for i in range(draws):
        sample = np.concatenate([
            rng.choice(classes[label], len(classes[label]), replace=True) for label in (0, 1)
        ])
        first_values[i] = np.mean([
            roc_auc_score(x.loc[sample].label, x.loc[sample, first_col])
            for x in by_seed.values()
        ])
        second_values[i] = np.mean([
            roc_auc_score(x.loc[sample].label, x.loc[sample, second_col])
            for x in by_seed.values()
        ])
        values[i] = first_values[i] - second_values[i]
    lo, hi = np.percentile(values, [2.5, 97.5])
    first_lo, first_hi = np.percentile(first_values, [2.5, 97.5])
    second_lo, second_hi = np.percentile(second_values, [2.5, 97.5])
    return {
        "point_delta_auc": point, "lower_95": float(lo), "upper_95": float(hi),
        "first_auc_lower_95": float(first_lo), "first_auc_upper_95": float(first_hi),
        "second_auc_lower_95": float(second_lo), "second_auc_upper_95": float(second_hi),
        "draws": draws, "resampling_unit": "physical_speaker",
        "seed_policy": "one draw shared across five repeated seeds",
    }


def source_audit() -> dict:
    paths = [
        HERE / "analysis_plan.json",
        ROOT / "v8/v8_publication_completion/frozen_reference/source/dysarthria_v8.ipynb",
        FROZEN / "features/manifest.json",
        V10 / "v10_reference_speaker_oof_predictions.csv",
        ROOT / "v8/v8_publication_completion/results/duration/predictions.csv",
        ROOT / "v8/v8_publication_completion/external_cache/silence_TORGO/features/manifest.json",
        ROOT / "v8/v8_publication_completion/external_cache/silence_TORGO/features/speech_embeddings.npy",
        ROOT / "v8/v8_publication_completion/external_cache/silence_TORGO/features/usable_metadata.csv",
        ROOT / "v8/v8_publication_completion/frozen_reference/pretrained/wavlm_base.pth",
    ]
    for _, _, folder, _ in checked_folds():
        paths.extend([
            folder / "fold_audit.json", folder / "inner_oof_speaker_predictions.csv",
            folder / "outer_speaker_predictions.csv", folder / "final/best.pt",
            V8_DURATION / folder.name / "inner_oof_predictions.csv",
            V8_DURATION / folder.name / "split_manifest.json",
        ])
    records = []
    for path in paths:
        records.append({
            "path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size,
            "sha256": sha256(path),
        })
    return {"version": "v10.4-1", "n_inputs": len(records), "inputs": records}
