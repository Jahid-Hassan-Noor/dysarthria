"""Three analyses reconstructed from saved speaker-level predictions."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from common import (
    OUT, ROOT, SEEDS, V8_DURATION, checked_folds, mean_seed_auc, mean_seed_ba,
    paired_auc_bootstrap, read_json, reference_predictions, write_csv, write_json,
)


def youden_threshold(frame: pd.DataFrame) -> tuple[float, float]:
    y = frame.label.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    if set(y) != {0, 1} or len(frame) != 12:
        raise AssertionError("Threshold selection requires 12 inner-OOF speakers in both classes")
    candidates = np.r_[np.nextafter(p.min(), -np.inf), np.unique(p), np.nextafter(p.max(), np.inf)]
    choices = []
    for threshold in candidates:
        sensitivity = float((p[y == 1] >= threshold).mean())
        specificity = float((p[y == 0] < threshold).mean())
        choices.append((sensitivity + specificity - 1, specificity, float(threshold)))
    best = max(choices)
    return best[2], best[0]


def threshold_analysis() -> dict:
    all_rows, policies = [], []
    for seed, outer, folder, audit in checked_folds():
        inner = pd.read_csv(folder / "inner_oof_speaker_predictions.csv")
        outer_frame = pd.read_csv(folder / "outer_speaker_predictions.csv")
        threshold, inner_j = youden_threshold(inner)
        original = float(audit["selected_threshold"])
        outer_frame = outer_frame[["seed", "outer_fold", "physical_speaker_id", "label", "probability"]].copy()
        outer_frame["original_threshold"] = original
        outer_frame["youden_threshold"] = threshold
        outer_frame["original_pred"] = (outer_frame.probability >= original).astype(int)
        outer_frame["youden_pred"] = (outer_frame.probability >= threshold).astype(int)
        all_rows.append(outer_frame)
        policies.append({
            "seed": seed, "outer_fold": outer, "original_threshold": original,
            "youden_threshold": threshold, "inner_youden_j": inner_j,
            "n_inner_speakers": 12, "n_outer_speakers": 3,
        })
    predictions = pd.concat(all_rows, ignore_index=True)
    if len(predictions) != 75:
        raise AssertionError("Missing outer threshold predictions")
    write_csv(OUT / "threshold/outer_predictions.csv", predictions)
    write_csv(OUT / "threshold/fold_policies.csv", pd.DataFrame(policies))
    seed_rows = []
    for seed, part in predictions.groupby("seed"):
        for rule in ("original", "youden"):
            p = part[f"{rule}_pred"].to_numpy()
            y = part.label.to_numpy()
            seed_rows.append({
                "seed": int(seed), "rule": rule,
                "balanced_accuracy": float(balanced_accuracy_score(y, p)),
                "sensitivity": float((p[y == 1] == 1).mean()),
                "specificity": float((p[y == 0] == 0).mean()),
                "roc_auc": float(roc_auc_score(y, part.probability)),
            })
    metrics = pd.DataFrame(seed_rows)
    write_csv(OUT / "threshold/per_seed_metrics.csv", metrics)
    original_ba = float(metrics.loc[metrics.rule == "original", "balanced_accuracy"].mean())
    youden_ba = float(metrics.loc[metrics.rule == "youden", "balanced_accuracy"].mean())

    ids = predictions[["physical_speaker_id", "label"]].drop_duplicates()
    classes = {label: ids.loc[ids.label == label, "physical_speaker_id"].to_numpy() for label in (0, 1)}
    by_seed = {s: x.set_index("physical_speaker_id") for s, x in predictions.groupby("seed")}
    rng = np.random.default_rng(20261021)
    boot = []
    for _ in range(2000):
        selected = np.concatenate([rng.choice(classes[c], len(classes[c]), replace=True) for c in (0, 1)])
        boot.append(np.mean([
            balanced_accuracy_score(x.loc[selected].label, x.loc[selected].youden_pred)
            - balanced_accuracy_score(x.loc[selected].label, x.loc[selected].original_pred)
            for x in by_seed.values()
        ]))
    lo, hi = np.percentile(boot, [2.5, 97.5])
    summary = {
        "original_mean_balanced_accuracy": original_ba,
        "youden_mean_balanced_accuracy": youden_ba,
        "youden_minus_original_balanced_accuracy": youden_ba - original_ba,
        "paired_bootstrap_delta_ba_95": [float(lo), float(hi)],
        "mean_roc_auc_both_rules": float(metrics.roc_auc.mean()),
        "n_folds": 25, "n_speakers": 15, "n_repeated_seeds": 5,
    }
    write_json(OUT / "threshold/summary.json", summary)
    return summary


def per_fold_auc() -> dict:
    rows = []
    for seed, outer, folder, _ in checked_folds():
        frame = pd.read_csv(folder / "outer_speaker_predictions.csv")
        pos = frame.loc[frame.label == 1, "probability"].to_numpy()
        neg = frame.loc[frame.label == 0, "probability"].to_numpy()
        comparisons = pos[:, None] - neg[None, :]
        rows.append({
            "seed": seed, "outer_fold": outer, "n_positive": len(pos), "n_negative": len(neg),
            "positive_negative_pairs": len(pos) * len(neg),
            "concordant_pairs": int((comparisons > 0).sum()),
            "discordant_pairs": int((comparisons < 0).sum()),
            "tied_pairs": int((comparisons == 0).sum()),
            "roc_auc": float(roc_auc_score(frame.label, frame.probability)),
            "held_out_speakers": ",".join(sorted(frame.physical_speaker_id)),
        })
    table = pd.DataFrame(rows)
    write_csv(OUT / "fold_auc/per_fold_auc.csv", table)
    summary = {
        "n_folds": len(table),
        "auc_value_counts": {str(k): int(v) for k, v in table.roc_auc.value_counts().sort_index().items()},
        "n_tied_pairs": int(table.tied_pairs.sum()),
        "mean_of_fold_auc_descriptive": float(table.roc_auc.mean()),
        "mean_seed_pooled_auc_primary": mean_seed_auc(reference_predictions()),
        "note": "Three speakers per fold yield two class-discordant pairs. The 25 folds reuse 15 speakers across five seeds.",
    }
    write_json(OUT / "fold_auc/summary.json", summary)
    return summary


def duration_increment() -> dict:
    reference = reference_predictions()
    duration = pd.read_csv(ROOT / "v8/v8_publication_completion/results/duration/predictions.csv")
    duration = duration[duration.corpus == "TORGO"].copy()
    duration["outer_fold"] = duration.outer_fold.astype(int)
    if len(duration) != 75:
        raise AssertionError("Duration-only outer scores are incomplete")
    rows = []
    for seed, outer, folder, audit in checked_folds():
        duration_folder = V8_DURATION / folder.name
        duration_audit = read_json(duration_folder / "split_manifest.json")
        for role in ("train_physical_speakers", "test_physical_speakers"):
            if set(audit[role]) != set(duration_audit[role]):
                raise AssertionError(f"Duration split differs at {folder.name}")
        for a, b in zip(audit["inner_folds"], duration_audit["inner_folds"]):
            for role in ("train_physical_speakers", "validation_physical_speakers"):
                if set(a[role]) != set(b[role]):
                    raise AssertionError(f"Duration inner split differs at {folder.name}")
        inner_v10 = pd.read_csv(folder / "inner_oof_speaker_predictions.csv")
        inner_duration = pd.read_csv(duration_folder / "inner_oof_predictions.csv")
        train = inner_duration[["physical_speaker_id", "label", "probability"]].merge(
            inner_v10[["physical_speaker_id", "label", "probability"]],
            on=["physical_speaker_id", "label"], suffixes=("_duration", "_v10"),
            validate="one_to_one",
        )
        outer_duration = duration[(duration.seed == seed) & (duration.outer_fold == outer)]
        outer_v10 = reference[(reference.seed == seed) & (reference.outer_fold == outer)]
        test = outer_duration[["physical_speaker_id", "label", "probability"]].merge(
            outer_v10[["physical_speaker_id", "label", "probability"]],
            on=["physical_speaker_id", "label"], suffixes=("_duration", "_v10"),
            validate="one_to_one",
        )
        if (set(train.physical_speaker_id) != set(audit["train_physical_speakers"])
                or set(test.physical_speaker_id) != set(audit["test_physical_speakers"])
                or len(train) != 12 or len(test) != 3):
            raise AssertionError(f"Stacking membership error in {folder.name}")
        for frame in (train, test):
            for name in ("duration", "v10"):
                frame[f"logit_{name}"] = logit(frame[f"probability_{name}"].clip(1e-6, 1 - 1e-6))
        outputs = test[["physical_speaker_id", "label", "probability_duration", "probability_v10"]].copy()
        for arm, columns in (
            ("duration_stack", ["logit_duration"]),
            ("duration_plus_v10", ["logit_duration", "logit_v10"]),
        ):
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=1, class_weight="balanced", max_iter=2000, random_state=seed
            ))
            model.fit(train[columns], train.label)
            outputs[arm] = model.predict_proba(test[columns])[:, 1]
        outputs["seed"] = seed
        outputs["outer_fold"] = outer
        rows.append(outputs)
    scores = pd.concat(rows, ignore_index=True)
    write_csv(OUT / "duration_increment/paired_outer_scores.csv", scores)
    metrics = []
    arms = ("probability_duration", "duration_stack", "duration_plus_v10", "probability_v10")
    for seed, group in scores.groupby("seed"):
        for arm in arms:
            metrics.append({"seed": int(seed), "arm": arm, "roc_auc": float(roc_auc_score(group.label, group[arm]))})
    metrics = pd.DataFrame(metrics)
    write_csv(OUT / "duration_increment/per_seed_auc.csv", metrics)
    augment = scores.rename(columns={"duration_plus_v10": "probability"})
    baseline = scores.rename(columns={"duration_stack": "probability"})
    paired = paired_auc_bootstrap(augment, baseline)
    summary = {
        "mean_auc_by_arm": {arm: float(metrics.loc[metrics.arm == arm, "roc_auc"].mean()) for arm in arms},
        "incremental_delta_c": paired,
        "n_outer_folds": 25, "n_training_speakers_per_fold": 12,
        "n_test_speakers_per_fold": 3,
        "interpretation": "Post hoc secondary analysis; stacking weights fit on outer-train speaker OOF scores only.",
    }
    write_json(OUT / "duration_increment/summary.json", summary)
    return summary


def main() -> None:
    print("Threshold:", threshold_analysis(), flush=True)
    print("Per-fold AUC:", per_fold_auc(), flush=True)
    print("Duration increment:", duration_increment(), flush=True)


if __name__ == "__main__":
    main()
