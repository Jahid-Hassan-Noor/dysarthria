"""Shared, read-only V10.4 adapters for exploratory V10.5 analyses."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import brier_score_loss, roc_auc_score, balanced_accuracy_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "outputs"
V104 = ROOT / "v10/v10.4"
# Keep this version's report/figure modules ahead of the V10.4 adapters.
sys.path.append(str(V104))
from common import (FROZEN, V10, SEEDS, checked_folds, load_backbone, read_json,
                    reference_predictions, sha256)  # noqa: E402
from spread import load_encoder, score_condition  # noqa: E402


def write_json(path: Path, value: object) -> None:
    path = Path(path).resolve()
    if HERE not in path.parents:
        raise PermissionError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def write_csv(path: Path, value: pd.DataFrame) -> None:
    path = Path(path).resolve()
    if HERE not in path.parents:
        raise PermissionError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    value.to_csv(temporary, index=False)
    temporary.replace(path)


def metrics(frame: pd.DataFrame) -> dict[str, float]:
    """Speaker-level metrics, with the original fixed-bin ECE recipe."""
    y = frame.label.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    t = frame.threshold.to_numpy(dtype=float)
    if len(np.unique(y)) != 2 or not np.isfinite(p).all():
        raise ValueError("Both classes and finite probabilities are required")
    d = p >= t
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
              for b in range(10) if (bins == b).any())
    return {"roc_auc": float(roc_auc_score(y, p)),
            "balanced_accuracy": float(balanced_accuracy_score(y, d)),
            "sensitivity": float(d[y == 1].mean()),
            "specificity": float((~d[y == 0]).mean()),
            "brier": float(brier_score_loss(y, p)), "ece10": float(ece)}


def paired_bootstrap(first: pd.DataFrame, second: pd.DataFrame, *, draws: int = 2000,
                     seed: int = 20261021) -> pd.DataFrame:
    """Diagnosis-stratified physical-speaker draws shared across all repeated seeds."""
    keys = ["seed", "physical_speaker_id", "label"]
    a = first[keys + ["probability", "threshold"]]
    b = second[keys + ["probability", "threshold"]]
    joined = a.merge(b, on=keys, suffixes=("_first", "_second"), validate="one_to_one")
    if len(joined) != len(a) or len(joined) != len(b):
        raise ValueError("Paired conditions have different speaker coverage")
    ids = joined[["physical_speaker_id", "label"]].drop_duplicates()
    if ids.physical_speaker_id.duplicated().any():
        raise ValueError("Speaker labels disagree")
    all_ids = set(ids.physical_speaker_id)
    groups = {int(s): x.set_index("physical_speaker_id") for s, x in joined.groupby("seed")}
    if set(groups) != set(SEEDS) or any(set(g.index) != all_ids for g in groups.values()):
        raise ValueError("Each seed must cover the same physical speakers")
    classes = {c: ids.loc[ids.label == c, "physical_speaker_id"].to_numpy() for c in (0, 1)}
    if min(map(len, classes.values())) < 2:
        raise ValueError("At least two speakers per diagnosis are required")

    def fast_metrics(x, suffix):
        y = x.label.to_numpy(dtype=int)
        p = x["probability_" + suffix].to_numpy(dtype=float)
        t = x["threshold_" + suffix].to_numpy(dtype=float)
        positive = y == 1
        negative = ~positive
        n1, n0 = int(positive.sum()), int(negative.sum())
        ranks = rankdata(p, method="average")
        auc = (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
        decisions = p >= t
        sensitivity = float(decisions[positive].mean())
        specificity = float((~decisions[negative]).mean())
        bins = np.minimum((p * 10).astype(int), 9)
        ece = sum(float((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean()))
                  for b in range(10) if (bins == b).any())
        return {"roc_auc": float(auc), "balanced_accuracy": (sensitivity + specificity) / 2,
                "sensitivity": sensitivity, "specificity": specificity,
                "brier": float(np.mean((p - y) ** 2)), "ece10": ece}

    def one(sample):
        values = []
        for suffix in ("first", "second"):
            rows = []
            for g in groups.values():
                rows.append(fast_metrics(g.loc[sample], suffix))
            values.append({k: float(np.mean([r[k] for r in rows])) for k in rows[0]})
        return {k: values[0][k] - values[1][k] for k in values[0]}

    point = one(ids.physical_speaker_id.to_numpy())
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in point}
    for _ in range(draws):
        selected = np.concatenate([rng.choice(classes[c], len(classes[c]), replace=True)
                                   for c in (0, 1)])
        value = one(selected)
        for k in samples:
            samples[k].append(value[k])
    return pd.DataFrame([{"metric": k, "delta_first_minus_second": point[k],
                          "lower_95": float(np.percentile(v, 2.5)),
                          "upper_95": float(np.percentile(v, 97.5)),
                          "draws": draws, "unit": "physical_speaker"}
                         for k, v in samples.items()])


def paired_effect_bootstrap(shifts: pd.DataFrame, *, draws: int = 2000,
                            seed: int = 20261021) -> pd.DataFrame:
    """Bootstrap condition-minus-sham score and decision effects by person."""
    columns = ("delta_logit", "delta_probability", "decision_changed")
    ids = shifts[["physical_speaker_id", "label"]].drop_duplicates()
    if ids.physical_speaker_id.duplicated().any():
        raise ValueError("Inconsistent speaker labels")
    if shifts.duplicated(["seed", "physical_speaker_id"]).any():
        raise ValueError("Repeated speaker/seed effect")
    by_speaker = shifts.groupby(["physical_speaker_id", "label"])[list(columns)].mean().reset_index()
    if len(by_speaker) != len(ids):
        raise ValueError("Incomplete speaker effects")
    classes = {c: by_speaker.loc[by_speaker.label == c, list(columns)].to_numpy(dtype=float)
               for c in (0, 1)}
    if min(len(x) for x in classes.values()) < 2:
        raise ValueError("Two speakers per diagnosis required")
    rng = np.random.default_rng(seed)
    values = np.empty((draws, len(columns)))
    for i in range(draws):
        sample = np.concatenate([classes[c][rng.integers(len(classes[c]), size=len(classes[c]))]
                                 for c in (0, 1)])
        values[i] = sample.mean(axis=0)
    point = by_speaker[list(columns)].mean().to_numpy()
    return pd.DataFrame([{"metric": column, "mean_condition_minus_sham": float(point[j]),
                          "lower_95": float(np.percentile(values[:, j], 2.5)),
                          "upper_95": float(np.percentile(values[:, j], 97.5)),
                          "draws": draws, "unit": "physical_speaker"}
                         for j, column in enumerate(columns)])


def verify_outer_checkpoints() -> pd.DataFrame:
    """Require all locked TORGO heads to match the V10.4 frozen-input manifest."""
    manifest = read_json(V104 / "source_manifest.json")
    expected = {r["path"]: r["sha256"] for r in manifest["inputs"]}
    records = []
    for seed, outer, folder, _ in checked_folds():
        checkpoint = folder / "final/best.pt"
        relative = str(checkpoint.relative_to(ROOT))
        actual = sha256(checkpoint)
        if expected.get(relative) != actual:
            raise RuntimeError(f"Locked V10 checkpoint changed: {relative}")
        records.append({"seed": seed, "outer_fold": outer, "checkpoint": relative,
                        "sha256": actual})
    return pd.DataFrame(records)
