"""Publication artifacts for V8, computed exclusively from registered saved results.

No fitting or operating-point selection is performed here. Missing experiments are
shown explicitly; existence of a PDF is never treated as completion of an experiment.
PR_AUC follows the frozen V8 definition (average precision, not trapezoidal PR area).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

METRICS = ("Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity",
           "F1_dysarthric", "ROC_AUC", "PR_AUC", "Brier", "ECE")
SEEDS = {42, 43, 44, 45, 46}
ARMS = {
    "A1": "WavLM Last Layer + Mean Pooling",
    "A2": "WavLM Last Layer + Attention",
    "B1": "WavLM Last-4 Mix + Mean Pooling",
    "B2": "WavLM Last-4 Mix + Attention (V8 Reference)",
}
TABLES = {
    "table01_torgo_primary": "TORGO physical-speaker-independent V8 primary result",
    "table02_component_ablation": "V8 component ablation",
    "table03_uaspeech_m3": "Official UASpeech M3 locked zero-shot result",
    "table04_channel_robustness": "UASpeech M2–M8 channel robustness",
    "table05_utterance_budget": "Utterance-budget sensitivity",
    "table06_sanity_controls": "Artifact/content sanity controls",
}
FIGURES = ("fig01_ablation", "fig02_torgo_roc_pr", "fig03_uaspeech_m3_roc_pr",
           "fig04_internal_external", "fig05_utterance_budget", "fig06_channels",
           "fig07_probabilities", "fig08_calibration", "fig09_sanity_controls")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def calculate_metrics(predictions: pd.DataFrame) -> dict[str, float]:
    """Reproduce the original notebook's nine metrics (ten equal-width ECE bins)."""
    y = predictions.label.to_numpy(dtype=int)
    p = predictions.probability.to_numpy(dtype=float)
    threshold = predictions.threshold.to_numpy(dtype=float)
    if not len(y) or not set(y) <= {0, 1}:
        raise ValueError("Nonempty binary-labelled speaker predictions are required")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite and in [0, 1]")
    if not np.isfinite(threshold).all():
        raise ValueError("Frozen thresholds must be finite")
    pred = p >= threshold
    pos, neg = y == 1, y == 0
    tp, fp, fn, tn = (np.sum(pred & pos), np.sum(pred & neg),
                      np.sum(~pred & pos), np.sum(~pred & neg))
    sensitivity = tp / pos.sum() if pos.any() else np.nan
    specificity = tn / neg.sum() if neg.any() else np.nan
    bins = np.minimum((p * 10).astype(int), 9)
    ece = sum((bins == b).mean() * abs(y[bins == b].mean() - p[bins == b].mean())
              for b in range(10) if (bins == b).any())
    both = pos.any() and neg.any()
    return dict(n_speakers=len(y), Accuracy=float((pred == y).mean()),
                Balanced_Accuracy=float((sensitivity + specificity) / 2),
                Sensitivity=float(sensitivity), Specificity=float(specificity),
                F1_dysarthric=float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.,
                ROC_AUC=float(roc_auc_score(y, p)) if both else np.nan,
                PR_AUC=float(average_precision_score(y, p)) if both else np.nan,
                Brier=float(np.mean((p-y)**2)), ECE=float(ece))


def _weighted_metrics(frame: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """All bootstrap draws at once; tied-score AP and AUC match sklearn."""
    y = frame.label.to_numpy(dtype=int)
    p = frame.probability.to_numpy(dtype=float)
    pred = p >= frame.threshold.to_numpy(dtype=float)
    pos, neg = y == 1, y == 0
    total = weights.sum(axis=1)
    positive, negative = weights[:, pos].sum(1), weights[:, neg].sum(1)
    tp, tn = weights @ (pred & pos), weights @ (~pred & neg)
    fp, fn = negative - tn, positive - tp
    sens, spec = tp / positive, tn / negative
    f1 = np.divide(2*tp, 2*tp + fp + fn, out=np.zeros_like(tp, dtype=float), where=2*tp + fp + fn != 0)
    cmp = (p[pos, None] > p[None, neg]).astype(float) + .5*(p[pos, None] == p[None, neg])
    auc = np.einsum("bi,ij,bj->b", weights[:, pos], cmp, weights[:, neg]) / (positive*negative)
    order = np.argsort(-p, kind="stable")
    sorted_p, sorted_y = p[order], y[order]
    end = np.r_[np.flatnonzero(np.diff(sorted_p)), len(p)-1]
    cum_tp = np.cumsum(weights[:, order] * sorted_y, axis=1)[:, end]
    cum_n = np.cumsum(weights[:, order], axis=1)[:, end]
    precision = np.divide(cum_tp, cum_n, out=np.zeros_like(cum_tp, dtype=float), where=cum_n != 0)
    ap = np.sum(precision * np.diff(np.c_[np.zeros(len(weights)), cum_tp], axis=1), axis=1) / positive
    bins = np.minimum((p*10).astype(int), 9)
    ece = sum(np.abs(weights[:, bins == b] @ (y[bins == b] - p[bins == b])) for b in range(10))/total
    return np.column_stack(((tp+tn)/total, (sens+spec)/2, sens, spec, f1, auc, ap,
                            (weights @ ((p-y)**2))/total, ece))


def _replicate_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in ("seed", "repetition") if column in frame]


def speaker_bootstrap(predictions: pd.DataFrame, *, resamples: int = 1000,
                      random_seed: int = 20261013) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Paired stratified speaker bootstrap, common draw across arms/seeds/repetitions.

    The estimand is the mean of per-seed (and, where present, per-repetition)
    metrics. Seed SD remains separate. Labels and cohorts must agree exactly.
    """
    if resamples < 2:
        raise ValueError("At least two bootstrap resamples are required")
    identity = predictions[["physical_speaker_id", "label"]].drop_duplicates().sort_values("physical_speaker_id")
    if identity.physical_speaker_id.duplicated().any() or set(identity.label) != {0, 1}:
        raise ValueError("Bootstrap requires a consistent two-class physical-speaker roster")
    ids, labels = identity.physical_speaker_id.tolist(), identity.label.to_numpy()
    rng = np.random.default_rng(random_seed)
    weights = np.zeros((resamples, len(ids)), dtype=float)
    for label in (0, 1):
        loc = np.flatnonzero(labels == label)
        weights[:, loc] = rng.multinomial(len(loc), np.full(len(loc), 1/len(loc)), size=resamples)
    output, draws = [], {}
    for arm, frame in predictions.groupby("model_arm", sort=True):
        replicate = _replicate_columns(frame)
        groups = list(frame.groupby(replicate, dropna=False))
        distribution = np.zeros((resamples, len(METRICS)))
        point = []
        for _, group in groups:
            if group.physical_speaker_id.duplicated().any() or set(group.physical_speaker_id) != set(ids):
                raise ValueError("Paired bootstrap requires identical physical-speaker cohorts in every arm/replicate")
            group = group.set_index("physical_speaker_id").loc[ids]
            distribution += _weighted_metrics(group, weights) / len(groups)
            point.append([calculate_metrics(group)[m] for m in METRICS])
        draws[str(arm)] = distribution
        point_mean = np.mean(point, axis=0)
        for index, metric in enumerate(METRICS):
            low, high = np.percentile(distribution[:, index], [2.5, 97.5])
            output.append(dict(model_arm=arm, metric=metric, mean=point_mean[index],
                               lower_95=low, upper_95=high, n_resamples=resamples,
                               n_speakers=len(ids), resampling_unit="physical_speaker",
                               stratified_by_diagnosis=True, same_draw_across_arms=True,
                               same_draw_across_seeds=True, same_draw_across_repetitions=True,
                               bootstrap_random_seed=random_seed))
    return pd.DataFrame(output), draws


def paired_ablation_statistics(predictions: pd.DataFrame, *, resamples: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    ci, draws = speaker_bootstrap(predictions, resamples=resamples)
    records = []
    if "B2" in draws:
        for arm in sorted(set(draws)-{"B2"}):
            for metric in ("Balanced_Accuracy", "ROC_AUC", "PR_AUC", "Brier"):
                index = METRICS.index(metric)
                values = draws[arm][:, index] - draws["B2"][:, index]
                point = ci.loc[ci.model_arm.eq(arm) & ci.metric.eq(metric), "mean"].iloc[0] - ci.loc[ci.model_arm.eq("B2") & ci.metric.eq(metric), "mean"].iloc[0]
                lo, hi = np.percentile(values, [2.5, 97.5])
                records.append(dict(comparison=f"{arm} minus B2", metric=metric, difference=point,
                                    lower_95=lo, upper_95=hi, n_resamples=resamples))
    if set(ARMS) <= set(draws):
        contrasts = {"Layer mixing main effect": {"B1": .5, "B2": .5, "A1": -.5, "A2": -.5},
                     "Attention main effect": {"A2": .5, "B2": .5, "A1": -.5, "B1": -.5},
                     "Layer mixing × attention interaction": {"B2": 1, "B1": -1, "A2": -1, "A1": 1}}
        for name, coefficients in contrasts.items():
            for index, metric in enumerate(METRICS):
                values = sum(coef*draws[arm][:, index] for arm, coef in coefficients.items())
                point = sum(coef*ci.loc[ci.model_arm.eq(arm) & ci.metric.eq(metric), "mean"].iloc[0] for arm, coef in coefficients.items())
                lo, hi = np.percentile(values, [2.5, 97.5])
                records.append(dict(comparison=name, metric=metric, difference=point,
                                    lower_95=lo, upper_95=hi, n_resamples=resamples))
    return ci, pd.DataFrame(records, columns=["comparison", "metric", "difference", "lower_95", "upper_95", "n_resamples"])


def _registry(root: Path) -> dict:
    return _json(root / "registry.json") if (root / "registry.json").exists() else {"datasets": {}}


def _resolve(root: Path, path: str) -> Path:
    return Path(path) if Path(path).is_absolute() else root / path


def _load_results(root: Path) -> dict[str, dict]:
    results = {}
    for name, entry in _registry(root).get("datasets", {}).items():
        results[name] = {"entry": entry,
                         "predictions": pd.read_csv(_resolve(root, entry["prediction_file"])),
                         "metrics": pd.read_csv(_resolve(root, entry["metric_file"]))}
    return results


def _assert_split(row: pd.Series, manifest: dict) -> None:
    """Accept original TORGO fold audits and corpus-namespaced external manifests."""
    speaker = str(row.physical_speaker_id)
    corpus = str(row.get("corpus", "TORGO"))
    train = set(map(str, manifest.get("train_physical_speakers", manifest.get("train_speakers", []))))
    test = set(map(str, manifest.get("test_physical_speakers", [])))
    validation = set(map(str, manifest.get("validation_physical_speakers", manifest.get("validation_speakers", []))))
    if not test:
        raise AssertionError("Split manifest is missing held-out/evaluation physical speakers")
    key = f"{corpus}:{speaker}" if any(":" in item for item in test) else speaker
    if key not in test:
        raise AssertionError(f"Prediction for {key} is not in the split's test speakers")
    if train & test or train & validation or test & validation:
        raise AssertionError("Physical-speaker overlap in split manifest")
    if not train:
        raise AssertionError("Split manifest does not identify training physical speakers")
    if corpus.lower() == "uaspeech":
        if manifest.get("training_corpus") != "TORGO" or manifest.get("evaluation_only") is not True:
            raise AssertionError("External split must explicitly identify TORGO training and evaluation-only use")
        if any(not s.startswith("TORGO:") for s in train):
            raise AssertionError("External manifests must namespace training physical speakers by corpus")
    for inner in manifest.get("inner_folds", []):
        itrain = set(map(str, inner.get("train_physical_speakers", inner.get("train_speakers", []))))
        ival = set(map(str, inner.get("validation_physical_speakers", inner.get("validation_speakers", []))))
        if not itrain or not ival or itrain & ival or itrain & test or ival & test:
            raise AssertionError("Nested physical-speaker independence failed")
        if itrain | ival != train:
            raise AssertionError("Inner partitions must cover exactly the outer training physical speakers")


def verify_registered_results(root: str | Path, *, strict: bool = True) -> dict:
    """Fail on metric corruption, stale hashes, foreign runs or invalid lineage."""
    root = Path(root)
    registry = _registry(root)
    reports, errors = {}, []
    hash_cache, split_cache = {}, {}
    for name, entry in registry.get("datasets", {}).items():
        try:
            paths = {key: _resolve(root, entry[key]) for key in ("prediction_file", "metric_file", "provenance_file")}
            p, m, provenance = pd.read_csv(paths["prediction_file"]), pd.read_csv(paths["metric_file"]), _json(paths["provenance_file"])
            for kind in ("prediction", "metric"):
                if file_sha256(paths[kind+"_file"]) != provenance.get(kind+"_sha256"):
                    raise AssertionError(f"{kind} SHA256 does not match provenance")
            required = {"physical_speaker_id", "label", "probability", "seed", "model_arm", "threshold",
                        "run_signature", "checkpoint_path", "checkpoint_sha256", "split_manifest"}
            if not required <= set(p):
                raise AssertionError(f"Missing lineage columns: {sorted(required-set(p))}")
            run = registry.get("run_signature")
            if not run or provenance.get("run_signature") != run or set(p.run_signature) != {run}:
                raise AssertionError("Run signatures disagree across registry, provenance and predictions")
            if provenance.get("external_model_selection") is not False:
                raise AssertionError("Provenance must forbid external model selection explicitly")
            groups = entry["group_columns"]
            if groups != provenance.get("group_columns"):
                raise AssertionError("Grouping disagrees with saved provenance")
            if p.duplicated(groups + ["physical_speaker_id"]).any():
                raise AssertionError("Duplicate physical speaker within a metric replicate")
            if "predicted_label" in p and not np.array_equal((p.probability >= p.threshold).astype(int), p.predicted_label):
                raise AssertionError("Saved classes disagree with frozen thresholds")
            computed = []
            for key, group in p.groupby(groups, dropna=False, sort=True):
                key = key if isinstance(key, tuple) else (key,)
                computed.append(dict(zip(groups, key), **calculate_metrics(group)))
            expected = pd.DataFrame(computed)
            if m.duplicated(groups).any() or len(m) != len(expected):
                raise AssertionError("Metric CSV has missing or duplicate groups")
            merged = m.merge(expected, on=groups, suffixes=("_saved", "_recomputed"), validate="one_to_one")
            if len(merged) != len(m):
                raise AssertionError("Metric groups do not match prediction groups")
            for metric in ("n_speakers", *METRICS):
                if not np.allclose(merged[metric+"_saved"], merged[metric+"_recomputed"], atol=1e-10, rtol=1e-10, equal_nan=True):
                    raise AssertionError(f"{metric} cannot be reproduced from saved predictions")
            if name in ("primary_torgo", "primary_uaspeech_m3"):
                frozen_run = root / "frozen_reference/run"
                if name == "primary_torgo":
                    frozen_pred = frozen_run / "v8_speaker_oof_predictions.csv"
                    frozen_metrics = frozen_run / "v8_per_seed_speaker_metrics.csv"
                else:
                    frozen_pred = frozen_run / "external_uaspeech/M3/v8_uaspeech_speaker_predictions.csv"
                    frozen_metrics = frozen_run / "external_uaspeech/M3/v8_uaspeech_per_seed_metrics.csv"
                expected_predictions = pd.read_csv(frozen_pred)
                identity = ["seed", "physical_speaker_id"]
                compared = p.sort_values(identity).reset_index(drop=True)
                original = expected_predictions.sort_values(identity).reset_index(drop=True)
                if not compared[identity].equals(original[identity]):
                    raise AssertionError("Registered primary speaker cohort differs from frozen V8")
                for column in ("label", "probability", "threshold", "predicted_label"):
                    if not np.allclose(compared[column], original[column], atol=1e-12, rtol=0):
                        raise AssertionError(f"Registered primary {column} differs from frozen V8")
                original_metrics = pd.read_csv(frozen_metrics).set_index("seed").sort_index()
                current_metrics = m.set_index("seed").sort_index()
                for metric in ("n_speakers", *METRICS):
                    if not np.allclose(current_metrics[metric], original_metrics[metric], atol=1e-12, rtol=0):
                        raise AssertionError(f"Registered primary {metric} differs from frozen V8")
            for _, row in p.iterrows():
                path = _resolve(root, str(row.checkpoint_path))
                if path not in hash_cache:
                    hash_cache[path] = file_sha256(path)
                if hash_cache[path] != row.checkpoint_sha256:
                    raise AssertionError(f"Checkpoint SHA256 mismatch: {path}")
                path = _resolve(root, str(row.split_manifest))
                if path not in split_cache:
                    split_cache[path] = _json(path)
                _assert_split(row, split_cache[path])
                split = split_cache[path]
                if "selected_threshold" in split and not np.isclose(row.threshold, split["selected_threshold"], atol=1e-12, rtol=0):
                    raise AssertionError("Prediction threshold differs from split's locked threshold")
                if split.get("run_signature", run) != run:
                    raise AssertionError("Split belongs to a foreign run")
                if "seed" in split and int(row.seed) != int(split["seed"]):
                    raise AssertionError("Split seed disagrees with prediction seed")
            if "feature_manifest" in p:
                for value in p.feature_manifest.dropna().unique():
                    feature_path = _resolve(root, str(value))
                    if feature_path not in hash_cache:
                        hash_cache[feature_path] = file_sha256(feature_path)
                    if "feature_manifest_sha256" in p:
                        expected_hashes = set(p.loc[p.feature_manifest.eq(value), "feature_manifest_sha256"].dropna().astype(str))
                        if expected_hashes and expected_hashes != {hash_cache[feature_path]}:
                            raise AssertionError(f"Feature/source manifest SHA256 mismatch: {feature_path}")
                    if name == "matched_content":
                        for transcript in _json(feature_path).get("transcript_files", []):
                            if file_sha256(transcript["path"]) != transcript["sha256"]:
                                raise AssertionError(f"Official word transcript changed: {transcript['path']}")
            if name.startswith("budgets_"):
                corpus = "TORGO" if name.endswith("torgo") else "UASpeech"
                budget_dir = root / "budgets" / corpus
                protocol = budget_dir / "protocol.json"
                protocol_sha = file_sha256(protocol)
                for seed in sorted(p.seed.unique()):
                    receipt = _json(budget_dir / f"seed{int(seed)}_receipt.json")
                    if receipt["protocol_sha256"] != protocol_sha:
                        raise AssertionError("Budget sampling protocol changed")
                    for field, filename in (("predictions_sha256", f"seed{int(seed)}_predictions.csv"),
                                            ("draws_sha256", f"seed{int(seed)}_draws.csv")):
                        if receipt[field] != file_sha256(budget_dir / filename):
                            raise AssertionError(f"Budget sampling artifact changed: {filename}")
            reports[name] = {"status": "PASS", "n_predictions": len(p), "n_metric_groups": len(m)}
        except (AssertionError, OSError, ValueError, KeyError, TypeError) as exc:
            reports[name] = {"status": "FAIL", "reason": str(exc)}
            errors.append(f"{name}: {exc}")
    result = {"status": "FAIL" if errors or not reports else "PASS", "datasets": reports, "errors": errors}
    if strict and result["status"] != "PASS":
        raise AssertionError("Publication traceability failed: " + "; ".join(errors or ["No registered results"]))
    return result


def _complete(frame: pd.DataFrame) -> bool:
    return not frame.empty and set(frame.seed) == SEEDS


def _summary_rows(metrics: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    """Average repetitions inside each seed before reporting between-seed SD."""
    if metrics.empty:
        return pd.DataFrame()
    records = []
    grouper = group_columns[0] if len(group_columns) == 1 else group_columns
    groups: Iterable = metrics.groupby(grouper, dropna=False, sort=True) if group_columns else [((), metrics)]
    for key, frame in groups:
        key = key if isinstance(key, tuple) else (key,)
        record = dict(zip(group_columns, key))
        seed_metrics = frame.groupby("seed", dropna=False)[list(METRICS)].mean()
        record.update(n_seeds=len(seed_metrics), n_speakers_min=int(frame.n_speakers.min()),
                      n_speakers_max=int(frame.n_speakers.max()), status="COMPLETE" if set(seed_metrics.index) == SEEDS else "INCOMPLETE")
        for metric in METRICS:
            values = seed_metrics[metric].dropna()
            mean, sd = values.mean(), values.std(ddof=1)
            record[metric+"_mean"], record[metric+"_sd"] = mean, sd
            record[metric] = f"{mean:.3f} ± {sd:.3f}" if len(values) > 1 else f"{mean:.3f} (SD not estimable)" if len(values) else "NOT ESTIMABLE"
        if "repetition" in frame:
            record["n_repetitions_min"] = int(frame.groupby("seed").repetition.nunique().min())
            for metric in METRICS:
                record[metric+"_sampling_sd"] = float(frame.groupby("seed")[metric].std(ddof=1).mean())
        records.append(record)
    return pd.DataFrame(records)


def _missing(name: str, reason: str = "Saved result not available") -> dict:
    return dict(model=name, status="NOT ESTIMABLE", reason=reason, **{metric: "NOT ESTIMABLE" for metric in METRICS})


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [c for c in ["model", "corpus", "analysis", "channel", "budget", "status", "n_seeds", "n_speakers_min", "n_recordings", "n_matched_prompts", *METRICS, "reason"] if c in frame]
    view = frame[columns].fillna("").astype(str).map(lambda value: value.replace("|", "/").replace("\n", " "))
    return "| " + " | ".join(columns) + " |\n| " + " | ".join(["---"]*len(columns)) + " |\n" + "\n".join("| "+" | ".join(row)+" |" for row in view.to_numpy()) + "\n"


def generate_publication_report(publication_root: str | Path, *, bootstrap_resamples: int = 1000) -> dict:
    root = Path(publication_root)
    verify_registered_results(root, strict=True)
    results = _load_results(root)
    out = root / "publication"
    tables_dir, sources_dir, figure_dir = out / "tables", out / "figure_sources", out / "figures"
    for directory in (tables_dir, sources_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)
    status_path = root / "status.json"
    stages = _json(status_path).get("stages", {}) if status_path.exists() else {}
    artifacts, sources = {}, {}
    registry = _registry(root)
    for name, result in results.items():
        for key in ("prediction_file", "metric_file", "provenance_file"):
            path = _resolve(root, result["entry"][key])
            sources[str(path.resolve())] = file_sha256(path)
        for key in ("split_manifest", "checkpoint_path", "feature_manifest", "sampling_manifest"):
            if key not in result["predictions"]:
                continue
            for raw in result["predictions"][key].dropna().unique():
                path = _resolve(root, str(raw))
                sources[str(path.resolve())] = file_sha256(path)
                if name == "matched_content" and key == "feature_manifest":
                    for transcript in _json(path).get("transcript_files", []):
                        source = Path(transcript["path"])
                        sources[str(source.resolve())] = file_sha256(source)
        if name.startswith("budgets_"):
            corpus = "TORGO" if name.endswith("torgo") else "UASpeech"
            budget_dir = root / "budgets" / corpus
            for path in [budget_dir / "protocol.json", *budget_dir.glob("seed*_receipt.json")]:
                sources[str(path.resolve())] = file_sha256(path)
    for path in (root / "registry.json", status_path):
        if path.exists():
            sources[str(path.resolve())] = file_sha256(path)

    def save_csv(path: Path, frame: pd.DataFrame, dependencies: list[str]) -> pd.DataFrame:
        frame.to_csv(path, index=False)
        artifacts[str(path.resolve())] = dict(sha256=file_sha256(path), source_datasets=dependencies, kind="csv")
        return pd.read_csv(path) if len(frame.columns) else frame

    def stage_reason(name: str) -> str:
        return stages.get(name, {}).get("reason", "Saved result not available; experiment has not completed")

    def summarize(name: str, columns: list[str] | None = None) -> pd.DataFrame:
        if name not in results:
            return pd.DataFrame([_missing(name, stage_reason(name))])
        frame = results[name]["metrics"]
        group = columns if columns is not None else [c for c in ("model_arm", "corpus", "analysis", "channel", "budget") if c in frame]
        summary = _summary_rows(frame, group)
        if "model_arm" in summary:
            summary["model"] = summary.model_arm.map(lambda arm: ARMS.get(arm, ARMS["B2"] if arm == "speech_only" else arm))
        p = results[name]["predictions"]
        for index, row in summary.iterrows():
            selected = p
            for col in group:
                selected = selected[selected[col].eq(row[col])]
            if "n_utterances" in selected and len(selected):
                counts = selected.groupby(_replicate_columns(selected)).n_utterances.sum()
                summary.loc[index, "n_recordings"] = int(counts.min()) if counts.nunique() == 1 else f"{int(counts.min())}–{int(counts.max())}"
            if "n_matched_prompts" in selected and len(selected):
                summary.loc[index, "n_matched_prompts"] = int(selected.n_matched_prompts.min())
        return summary

    primary_torgo, primary_external = summarize("primary_torgo"), summarize("primary_uaspeech_m3")
    ablation_p = results.get("ablations", {}).get("predictions", pd.DataFrame()).copy()
    ablation_m = results.get("ablations", {}).get("metrics", pd.DataFrame()).copy()
    if "primary_torgo" in results and (ablation_p.empty or "B2" not in set(ablation_p.model_arm)):
        p = results["primary_torgo"]["predictions"].copy(); p["model_arm"] = "B2"
        m = results["primary_torgo"]["metrics"].copy(); m["model_arm"] = "B2"
        ablation_p = pd.concat([ablation_p, p], ignore_index=True)
        ablation_m = pd.concat([ablation_m, m], ignore_index=True)
    ablation = _summary_rows(ablation_m, ["model_arm"]) if len(ablation_m) else pd.DataFrame()
    if len(ablation):
        ablation["model"] = ablation.model_arm.map(lambda a: ARMS.get(a, a))
    for arm, title in ARMS.items():
        if ablation.empty or arm not in set(ablation.model_arm.dropna()):
            ablation = pd.concat([ablation, pd.DataFrame([dict(_missing(title, stage_reason("ablations")), model_arm=arm)])], ignore_index=True)
    ablation = ablation.sort_values("model_arm")
    channel_parts = [summarize(name).assign(analysis=analysis)
                     for name, analysis in (("channels", "all_available"),
                                            ("channels_matched", "matched_across_channels"))
                     if name in results]
    channels = pd.concat(channel_parts, ignore_index=True) if channel_parts else pd.DataFrame()
    if "primary_uaspeech_m3" in results and (channels.empty or "M3" not in set(channels.get("channel", []))):
        channels = pd.concat([channels, primary_external.assign(channel="M3", analysis="all_available")], ignore_index=True)
    for channel in [f"M{i}" for i in range(2, 9)]:
        if channels.empty or channel not in set(channels.channel):
            channels = pd.concat([channels, pd.DataFrame([dict(_missing(ARMS["B2"], stage_reason("channels")), channel=channel, analysis="all_available")])], ignore_index=True)
    channels["prespecified_primary"] = channels.channel.eq("M3")
    budget_parts = []
    for corpus, name in (("TORGO", "budgets_torgo"), ("UASpeech", "budgets_uaspeech")):
        part = summarize(name) if name in results else pd.DataFrame()
        for budget in ["1", "5", "10", "20", "50", "all"]:
            existing = set(part.budget.astype(str)) if "budget" in part else set()
            if budget not in existing:
                part = pd.concat([part, pd.DataFrame([dict(_missing(ARMS["B2"], stage_reason(name)), corpus=corpus, budget=budget)])], ignore_index=True)
        budget_parts.append(part)
    budgets = pd.concat(budget_parts, ignore_index=True)
    sanity_parts = []
    for corpus, primary in (("TORGO", primary_torgo), ("UASpeech", primary_external)):
        if len(primary) and primary.status.eq("COMPLETE").any():
            sanity_parts.append(primary.assign(corpus=corpus, analysis="primary_reference"))
    sanity_parts.extend(summarize(name).assign(analysis=name)
                        for name in ("silence", "duration", "artifact", "matched_content"))
    sanity = pd.concat(sanity_parts, ignore_index=True)
    tables = {"table01_torgo_primary": (primary_torgo, ["primary_torgo"]),
              "table02_component_ablation": (ablation, [n for n in ("primary_torgo", "ablations") if n in results]),
              "table03_uaspeech_m3": (primary_external, ["primary_uaspeech_m3"]),
              "table04_channel_robustness": (channels, [n for n in ("primary_uaspeech_m3", "channels", "channels_matched") if n in results]),
              "table05_utterance_budget": (budgets, [n for n in ("budgets_torgo", "budgets_uaspeech") if n in results]),
              "table06_sanity_controls": (sanity, [n for n in ("primary_torgo", "primary_uaspeech_m3", "silence", "duration", "artifact", "matched_content") if n in results])}
    for name, (frame, dependencies) in tables.items():
        path = tables_dir / f"v8_pub_{name}.csv"
        saved = save_csv(path, frame, dependencies)
        md = path.with_suffix(".md")
        md.write_text(f"# {TABLES[name]}\n\n" + _markdown_table(saved) + "\nMetrics are mean ± sample SD across seeds; repetitions are averaged inside each seed. "
                      "PR-AUC denotes average precision. Missing analyses contain no estimated values. "
                      "External results are locked zero-shot cross-corpus evaluations; M3 is prespecified primary.\n")
        artifacts[str(md.resolve())] = dict(sha256=file_sha256(md), source_files=[str(path.resolve())], kind="table")

    # Supplemental full-metric CIs, and paired effects sharing physical-speaker draws.
    ci_tables = []
    if len(ablation_p):
        ci, paired = paired_ablation_statistics(ablation_p, resamples=bootstrap_resamples)
        ci["dataset"] = "ablations"
        ci_tables.append(ci)
        save_csv(tables_dir / "v8_pub_paired_ablation_differences.csv", paired, tables["table02_component_ablation"][1])
    for name, result in results.items():
        if name in ("ablations", "primary_torgo"):
            continue
        p = result["predictions"]
        dimensions = [c for c in ("corpus", "analysis", "channel", "budget") if c in p]
        grouper = dimensions[0] if len(dimensions) == 1 else dimensions
        for key, frame in (p.groupby(grouper, dropna=False) if dimensions else [((), p)]):
            key = key if isinstance(key, tuple) else (key,)
            if set(frame.label) != {0, 1}:
                continue
            ci, _ = speaker_bootstrap(frame, resamples=bootstrap_resamples)
            ci["dataset"] = name
            for column, value in zip(dimensions, key):
                ci[column] = value
            ci_tables.append(ci)
    all_ci = pd.concat(ci_tables, ignore_index=True) if ci_tables else pd.DataFrame(columns=["dataset", "metric", "lower_95", "upper_95"])
    save_csv(tables_dir / "v8_pub_speaker_bootstrap_ci.csv", all_ci, list(results))

    # All plotting starts from files saved above or registered prediction CSVs.
    figure_sources = {}
    for name, (frame, dependencies) in tables.items():
        figure_sources[name] = save_csv(sources_dir / f"{name}.csv", frame, dependencies)
    probability_parts = []
    for corpus, name in (("TORGO", "primary_torgo"), ("UASpeech M3", "primary_uaspeech_m3")):
        if name in results:
            p = pd.read_csv(_resolve(root, results[name]["entry"]["prediction_file"]))
            p["display_corpus"] = corpus
            probability_parts.append(p)
    probabilities = pd.concat(probability_parts, ignore_index=True) if probability_parts else pd.DataFrame(columns=["display_corpus", "seed", "label", "probability", "physical_speaker_id", "threshold"])
    save_csv(sources_dir / "speaker_probabilities.csv", probabilities, [n for n in ("primary_torgo", "primary_uaspeech_m3") if n in results])
    _render_figures(sources_dir, figure_dir, tables_dir, artifacts)
    manifest = dict(schema_version=1, run_signature=registry.get("run_signature"), source_hashes=sources,
                    artifacts=artifacts, bootstrap_resamples=bootstrap_resamples,
                    interpretation="Locked zero-shot cross-corpus evaluation on official UASpeech; the corpus is not historically unseen to this project.")
    _write_json(out / "traceability_manifest.json", manifest)
    audit = audit_run(root, strict=True)
    return {"output_dir": str(out), "tables": len(TABLES), "figures": len(FIGURES), "audit": audit}


def _render_figures(source_dir: Path, figure_dir: Path, table_dir: Path, artifacts: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300})
    colors = {"Balanced_Accuracy": "#237a91", "ROC_AUC": "#c46b31", "PR_AUC": "#75559b"}
    source_files = list(source_dir.glob("*.csv")) + [table_dir / "v8_pub_speaker_bootstrap_ci.csv"]

    def save(fig, name):
        fig.tight_layout()
        for extension in ("png", "pdf"):
            path = figure_dir / f"v8_pub_{name}.{extension}"
            fig.savefig(path, dpi=300, bbox_inches="tight")
            artifacts[str(path.resolve())] = dict(sha256=file_sha256(path), kind="figure",
                source_files=[str(p.resolve()) for p in source_files], png_dpi=300 if extension == "png" else None)
        plt.close(fig)

    def missing(ax, message="NOT ESTIMABLE\nSaved results are not available"):
        ax.text(.5, .5, message, ha="center", va="center", transform=ax.transAxes, color="#656565")
        ax.set_xticks([]); ax.set_yticks([])

    def metric_plot(ax, frame, labels, metrics=("Balanced_Accuracy", "ROC_AUC")):
        x = np.arange(len(frame), dtype=float)
        present = False
        for index, metric in enumerate(metrics):
            if metric+"_mean" not in frame:
                continue
            y = pd.to_numeric(frame[metric+"_mean"], errors="coerce").to_numpy()
            sd = pd.to_numeric(frame.get(metric+"_sd", pd.Series(np.nan, index=frame.index)), errors="coerce").fillna(0).to_numpy()
            offset = (index-(len(metrics)-1)/2)*.16
            good = np.isfinite(y)
            if good.any():
                ax.errorbar(x[good]+offset, y[good], yerr=sd[good], fmt="o", capsize=3,
                            label=metric.replace("_", " "), color=colors.get(metric))
                present = True
        ax.set_xticks(x, labels, rotation=20, ha="right")
        ax.set_ylim(-.03, 1.07)
        ax.set_ylabel("Five-seed mean ± SD")
        ax.grid(axis="y", alpha=.15)
        if present:
            ax.legend(loc="lower left", fontsize=8)
        else:
            missing(ax)
        for i, (_, row) in enumerate(frame.iterrows()):
            if row.get("status") != "COMPLETE":
                ax.text(i, .04, "Incomplete", rotation=90, color="#888", ha="center", va="bottom", fontsize=8)

    ablation = pd.read_csv(source_dir / "table02_component_ablation.csv")
    fig, ax = plt.subplots(figsize=(8, 4.3))
    metric_plot(ax, ablation, ablation.model_arm)
    ax.set_title("V8 component ablation · TORGO physical speakers")
    ax.set_xlabel("A: last layer; B: last-4 mix · 1: mean; 2: attention (B2 reference)")
    save(fig, "fig01_ablation")

    probabilities = pd.read_csv(source_dir / "speaker_probabilities.csv")
    curve_rows, calibration_rows = [], []
    for (corpus, seed), group in probabilities.groupby(["display_corpus", "seed"]):
        y, p = group.label.to_numpy(), group.probability.to_numpy()
        if len(set(y)) == 2:
            fpr, tpr, _ = roc_curve(y, p)
            precision, recall, _ = precision_recall_curve(y, p)
            for x, yy in zip(fpr, tpr):
                curve_rows.append(dict(corpus=corpus, seed=seed, curve="ROC", x=x, y=yy))
            for x, yy in zip(recall, precision):
                curve_rows.append(dict(corpus=corpus, seed=seed, curve="PR", x=x, y=yy))
        bins = np.minimum((p*10).astype(int), 9)
        for index in range(10):
            use = bins == index
            if use.any():
                calibration_rows.append(dict(corpus=corpus, seed=seed, bin=index,
                    mean_probability=float(p[use].mean()), observed_fraction=float(y[use].mean()), n_speakers=int(use.sum())))
    for filename, rows, columns in (("roc_pr_curves.csv", curve_rows, ["corpus", "seed", "curve", "x", "y"]),
                                   ("calibration_bins.csv", calibration_rows, ["corpus", "seed", "bin", "mean_probability", "observed_fraction", "n_speakers"])):
        path = source_dir / filename
        pd.DataFrame(rows, columns=columns).to_csv(path, index=False)
        artifacts[str(path.resolve())] = dict(sha256=file_sha256(path), kind="csv", source_files=[str((source_dir / "speaker_probabilities.csv").resolve())])
        source_files.append(path)
    curves = pd.read_csv(source_dir / "roc_pr_curves.csv")
    for corpus, number in (("TORGO", "fig02_torgo_roc_pr"), ("UASpeech M3", "fig03_uaspeech_m3_roc_pr")):
        fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
        for ax, kind in zip(axes, ("ROC", "PR")):
            for seed, frame in curves[curves.corpus.eq(corpus) & curves.curve.eq(kind)].groupby("seed"):
                ax.plot(frame.x, frame.y, linewidth=1.4, label=f"Seed {seed}")
            if curves[curves.corpus.eq(corpus)].empty:
                missing(ax)
            else:
                ax.legend(loc="lower right" if kind == "ROC" else "lower left", fontsize=8)
            if kind == "ROC":
                ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=1)
            ax.set(xlim=(-.02, 1.02), ylim=(-.02, 1.02), xlabel="False-positive rate" if kind == "ROC" else "Recall",
                   ylabel="True-positive rate" if kind == "ROC" else "Precision", title=kind)
        fig.suptitle(f"{corpus} · speaker-level discrimination")
        save(fig, number)

    internal, external = (pd.read_csv(source_dir / filename) for filename in ("table01_torgo_primary.csv", "table03_uaspeech_m3.csv"))
    fig, ax = plt.subplots(figsize=(9, 4.2))
    compare_metrics = ["Balanced_Accuracy", "Sensitivity", "Specificity", "ROC_AUC", "PR_AUC", "Brier"]
    for offset, (name, frame) in enumerate((("TORGO OOF", internal), ("UASpeech M3 locked", external))):
        if frame.empty or "Balanced_Accuracy_mean" not in frame:
            continue
        row = frame.iloc[0]
        ax.errorbar(np.arange(len(compare_metrics))+(offset-.5)*.15,
                    [row.get(m+"_mean", np.nan) for m in compare_metrics],
                    yerr=[row.get(m+"_sd", 0) for m in compare_metrics], fmt="o", capsize=3, label=name)
    ax.set_xticks(np.arange(len(compare_metrics)), [m.replace("_", " ") for m in compare_metrics], rotation=20, ha="right")
    ax.set(ylabel="Five-seed mean ± SD", ylim=(-.03, 1.07), title="Internal versus locked external performance")
    if ax.has_data(): ax.legend(fontsize=8)
    save(fig, "fig04_internal_external")

    budgets = pd.read_csv(source_dir / "table05_utterance_budget.csv")
    cis = pd.read_csv(table_dir / "v8_pub_speaker_bootstrap_ci.csv")
    budget_order = {"1": 0, "5": 1, "10": 2, "20": 3, "50": 4, "all": 5}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, corpus in zip(axes, ("TORGO", "UASpeech")):
        part = budgets[budgets.corpus.eq(corpus)].copy() if "corpus" in budgets else pd.DataFrame()
        if len(part):
            part["x"] = part.budget.astype(str).map(budget_order)
            part = part.sort_values("x")
        for metric in ("Balanced_Accuracy", "ROC_AUC"):
            if metric+"_mean" not in part:
                continue
            y = pd.to_numeric(part[metric+"_mean"], errors="coerce")
            if not y.notna().any():
                continue
            ax.plot(part.x, y, "o-", label=metric.replace("_", " "), color=colors[metric])
            subset = cis[cis.dataset.eq("budgets_torgo" if corpus == "TORGO" else "budgets_uaspeech") & cis.metric.eq(metric)]
            low, high = [], []
            for budget in part.budget.astype(str):
                match = subset[subset.budget.astype(str).eq(budget)] if "budget" in subset else pd.DataFrame()
                low.append(match.lower_95.iloc[0] if len(match) else np.nan)
                high.append(match.upper_95.iloc[0] if len(match) else np.nan)
            ax.fill_between(part.x.to_numpy(dtype=float), np.asarray(low, float), np.asarray(high, float), alpha=.13, color=colors[metric])
        if not ax.has_data(): missing(ax)
        else: ax.legend(fontsize=8, loc="lower right")
        ax.set_xticks(range(6), list(budget_order))
        ax.set(title=corpus, xlabel="Utterances per speaker", ylabel="Mean; speaker-bootstrap 95% CI", ylim=(-.03, 1.03))
    fig.suptitle("Frozen-model utterance-budget sensitivity")
    save(fig, "fig05_utterance_budget")

    channel = pd.read_csv(source_dir / "table04_channel_robustness.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for ax, matched in zip(axes, (False, True)):
        mask = channel.get("analysis", pd.Series("", index=channel.index)).fillna("").astype(str).str.contains("matched")
        part = channel[mask if matched else ~mask].sort_values("channel")
        if len(part):
            labels = [c+"*" if c == "M3" else c for c in part.channel]
            metric_plot(ax, part, labels)
        else:
            missing(ax)
        ax.set_title("Matched content across channels" if matched else "All available recordings per channel")
    fig.suptitle("UASpeech channel robustness · M3* prespecified primary")
    save(fig, "fig06_channels")

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row_index, corpus in enumerate(("TORGO", "UASpeech M3")):
        part = probabilities[probabilities.display_corpus.eq(corpus)]
        ax, heat = axes[row_index]
        if part.empty:
            missing(ax); missing(heat)
            continue
        ordered_seeds = sorted(part.seed.unique())
        for index, seed in enumerate(ordered_seeds):
            for label, color in ((0, "#237a91"), (1, "#bb6633")):
                group = part[part.seed.eq(seed) & part.label.eq(label)].sort_values("physical_speaker_id")
                x = index+(label-.5)*.28
                jitter = np.linspace(-.07, .07, len(group))
                ax.scatter(x+jitter, group.probability, s=15, alpha=.8, color=color,
                           label=("Control" if label == 0 else "Dysarthric") if index == 0 else None)
                if len(group):
                    ax.plot([x-.09, x+.09], [group.probability.median()]*2, color=color, linewidth=2)
            thresholds = part[part.seed.eq(seed)].threshold
            ax.plot([index-.33, index+.33], [thresholds.median()]*2, "k--", linewidth=.8)
            if thresholds.nunique() > 1:
                ax.fill_between([index-.33, index+.33], thresholds.min(), thresholds.max(), alpha=.06, color="black")
        ax.set_xticks(range(len(ordered_seeds)), ordered_seeds)
        ax.set(title=corpus+" · probabilities by seed", xlabel="Seed", ylabel="P(dysarthric)", ylim=(-.03, 1.03))
        ax.legend(fontsize=8, loc="upper left")
        order = part[["physical_speaker_id", "label"]].drop_duplicates().sort_values(["label", "physical_speaker_id"]).physical_speaker_id
        matrix = part.pivot(index="physical_speaker_id", columns="seed", values="probability").loc[order]
        thresholds = part.pivot(index="physical_speaker_id", columns="seed", values="threshold").loc[order]
        labels = part.drop_duplicates("physical_speaker_id").set_index("physical_speaker_id").loc[order, "label"].to_numpy()
        picture = heat.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        bad = (matrix.to_numpy() >= thresholds.to_numpy()) != labels[:, None]
        for r, c in zip(*np.where(bad)):
            heat.text(c, r, "×", color="white", ha="center", va="center", fontsize=9)
        heat.set_xticks(range(len(matrix.columns)), matrix.columns)
        heat.set_yticks(range(len(matrix)), matrix.index, fontsize=7)
        heat.set(title="Speaker × seed · × indicates locked-threshold error", xlabel="Seed", ylabel="Corpus speaker ID")
        fig.colorbar(picture, ax=heat, label="P(dysarthric)", fraction=.03)
    fig.suptitle("Speaker probability diagnostics · bars: median probability; dashed: median locked threshold\nTORGO shading spans fold-specific thresholds")
    save(fig, "fig07_probabilities")

    calibration = pd.read_csv(source_dir / "calibration_bins.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.3))
    for ax, corpus in zip(axes, ("TORGO", "UASpeech M3")):
        for seed, part in calibration[calibration.corpus.eq(corpus)].groupby("seed"):
            ax.scatter(part.mean_probability, part.observed_fraction,
                       s=18 + 13 * part.n_speakers, alpha=.8, label=f"Seed {seed}")
        if not ax.has_data(): missing(ax)
        else: ax.legend(fontsize=8)
        ax.plot([0, 1], [0, 1], "--", color="#aaa", linewidth=1)
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel="Mean predicted probability", ylabel="Observed dysarthric fraction", title=corpus)
    fig.suptitle("Calibration · ten fixed equal-width bins; marker area reflects speaker count\nNo external recalibration; sparse bins are shown without interpolation")
    save(fig, "fig08_calibration")

    sanity = pd.read_csv(source_dir / "table06_sanity_controls.csv")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, corpus in zip(axes, ("TORGO", "UASpeech")):
        part = sanity[sanity.corpus.eq(corpus)] if "corpus" in sanity else sanity
        if len(part):
            label_map = {"primary_reference":"V8 full speech",
                         "silence":"V8 background",
                         "duration":"Duration only",
                         "artifact":"Recording features",
                         "matched_content":"V8 matched words"}
            labels = part.get("analysis", part.model).fillna(part.model).astype(str).map(lambda name:label_map.get(name,name.replace("_"," "))).tolist()
            metric_plot(ax, part, labels)
        else: missing(ax)
        ax.set_title(corpus)
    fig.suptitle("Sanity-control diagnostics · high discrimination can indicate recording/content confounds")
    save(fig, "fig09_sanity_controls")


def audit_run(publication_root: str | Path, *, strict: bool = True) -> dict:
    """Write a scientific completion audit; missing work is distinct from corrupt work."""
    root = Path(publication_root)
    registered = verify_registered_results(root, strict=False)
    results = _load_results(root)
    registry = _registry(root)
    out = root / "publication"
    stages = _json(root / "status.json").get("stages", {}) if (root / "status.json").exists() else {}
    limitations, failures = [], list(registered["errors"])
    snapshot = _resolve(root, registry.get("frozen_reference", "frozen_reference"))
    frozen, frozen_metadata = {"status": "FAIL"}, {}
    try:
        from .freeze import verify_reference
        frozen = verify_reference(snapshot, compare_originals=True)
        frozen_metadata = _json(snapshot / "snapshot_manifest.json")
        limitations.extend(frozen.get("limitations", []))
    except (OSError, ValueError, RuntimeError, AssertionError, KeyError) as exc:
        failures.append(f"Frozen reference: {exc}")
    trace_path = out / "traceability_manifest.json"
    figure_ok, tables_ok = False, False
    if trace_path.exists():
        manifest = _json(trace_path)
        for path, expected in manifest.get("source_hashes", {}).items():
            if not Path(path).is_file() or file_sha256(path) != expected:
                failures.append(f"Source hash mismatch: {path}")
        for path, item in manifest.get("artifacts", {}).items():
            if not Path(path).is_file() or file_sha256(path) != item["sha256"]:
                failures.append(f"Publication artifact hash mismatch: {path}")
            for source in item.get("source_files", []):
                if source not in manifest.get("artifacts", {}) and source not in manifest.get("source_hashes", {}):
                    failures.append(f"Unregistered figure/table source: {source}")
        figure_ok = all((out / "figures" / f"v8_pub_{name}.{ext}").exists() for name in FIGURES for ext in ("png", "pdf"))
        tables_ok = all((out / "tables" / f"v8_pub_{name}.csv").exists() for name in TABLES)
    else:
        failures.append("Publication traceability manifest is missing")

    def completed(name: str) -> bool:
        if name not in results or registered["datasets"].get(name, {}).get("status") != "PASS":
            return False
        frame = results[name]["predictions"]
        dimensions = [c for c in ("corpus", "model_arm", "channel", "budget") if c in frame]
        return all(set(part.seed) == SEEDS for _, part in frame.groupby(dimensions, dropna=False))

    def stage_answer(name: str, not_estimable: bool = False) -> str:
        if completed(name): return "YES"
        if not_estimable and stages.get(name, {}).get("state") == "not_estimable": return "NOT ESTIMABLE"
        return "NO"

    primary = completed("primary_torgo")
    ablations = completed("ablations") and set(results["ablations"]["predictions"].model_arm) == set(ARMS)
    nested = primary
    if primary:
        for path in results["primary_torgo"]["predictions"].split_manifest.unique():
            nested &= bool(_json(_resolve(root, path)).get("inner_folds"))
    budget_complete = True
    for name in ("budgets_torgo", "budgets_uaspeech"):
        if not completed(name):
            budget_complete = False
            continue
        p = results[name]["predictions"]
        budget_complete &= set(p.budget.astype(str)) == {"1", "5", "10", "20", "50", "all"}
        for budget, part in p.groupby("budget"):
            expected = 1 if str(budget) == "all" else 100
            budget_complete &= all(group.repetition.nunique() >= expected for _, group in part.groupby("seed"))
    channels = completed("channels") and set(results["channels"]["predictions"].channel) == {f"M{i}" for i in range(2, 9)}
    external_audit_paths = list(snapshot.rglob("v8_uaspeech_zero_shot_audit.json")) if snapshot.exists() else []
    external_audit = _json(external_audit_paths[0]) if external_audit_paths else {}
    used = {}
    for action in ("training", "fine_tuning", "threshold_selection", "calibration", "model_selection"):
        value = external_audit.get("external_dataset_used_for_"+action)
        used[action] = "NO" if value is False else "YES" if value is True else "UNKNOWN"
        if used[action] != "NO":
            failures.append(f"UASpeech {action} exclusion cannot be verified")
    ci_path = out / "tables/v8_pub_speaker_bootstrap_ci.csv"
    uncertainty = False
    if ci_path.exists():
        ci = pd.read_csv(ci_path)
        uncertainty = all(set(part.metric) == set(METRICS) for _, part in ci.groupby(["dataset", "model_arm"], dropna=False)) and len(ci) > 0
    checks = {
        "Reference V8 frozen": frozen["status"],
        "Physical-speaker leakage checks": "PASS" if registered["status"] == "PASS" else "FAIL",
        "Five TORGO seeds complete": "YES" if primary else "NO",
        "Nested inner speaker validation": "PASS" if nested else "FAIL",
        "V8 component ablation complete": "YES" if ablations else "NO",
        "Layer-mix contribution evaluated": "YES" if ablations else "NO",
        "Attention-pooling contribution evaluated": "YES" if ablations else "NO",
        "Speaker-bootstrap uncertainty complete": "YES" if uncertainty and ablations else "NO",
        "Utterance-budget analysis complete": "YES" if budget_complete else "NO",
        "Silence/background sanity check": stage_answer("silence", True),
        "Duration-only sanity baseline": stage_answer("duration"),
        "Artifact-only baseline": stage_answer("artifact"),
        "Matched-content UASpeech analysis": stage_answer("matched_content", True),
        "UASpeech M3 locked zero-shot complete": "YES" if completed("primary_uaspeech_m3") else "NO",
        "UASpeech M2-M8 diagnostic complete": "YES" if channels else "NO",
        "UASpeech used for training": used["training"],
        "UASpeech used for fine-tuning": used["fine_tuning"],
        "UASpeech used for threshold selection": used["threshold_selection"],
        "UASpeech used for calibration": used["calibration"],
        "UASpeech used for model selection": used["model_selection"],
        "Publication tables generated": "YES" if tables_ok else "NO",
        "Publication figures generated": "YES" if figure_ok else "NO",
        "Traceability audit": "PASS" if not failures else "FAIL",
    }
    required = ("ablations", "budgets_torgo", "budgets_uaspeech", "silence", "duration", "artifact", "matched_content", "channels", "channels_matched")
    for name in required:
        if not completed(name):
            detail = stages.get(name, {}).get("reason", "No complete registered five-seed result")
            limitations.append(f"{name}: {detail}")
    if not ablations:
        limitations.append("Layer-mixing and attention contributions cannot be concluded until all four arms finish the same five seeds and folds.")
    if not budget_complete:
        limitations.append("A complete budget experiment requires both corpora, all six budgets, and 100 finite-budget repetitions per seed.")
    if ablations:
        paired_path = root / "ablations/paired_differences.csv"
        if paired_path.exists():
            paired = pd.read_csv(paired_path)
            ba = paired[paired.metric.eq("Balanced_Accuracy")]
            if len(ba) == 3 and all((ba.lower_95 <= 0) & (ba.upper_95 >= 0)):
                limitations.append("All three ablation-minus-reference balanced-accuracy speaker-bootstrap intervals include zero. The observed five-seed results do not establish a clear advantage for the reference pooling and layer mix; they also do not prove equivalence.")
    if budget_complete:
        budget_metrics = results["budgets_torgo"]["metrics"]
        try:
            five = float(budget_metrics.loc[budget_metrics.budget.astype(str).eq("5"), "Balanced_Accuracy"].mean())
            all_speech = float(budget_metrics.loc[budget_metrics.budget.astype(str).eq("all"), "Balanced_Accuracy"].mean())
            if all_speech > five:
                limitations.append(f"TORGO balanced accuracy is lower with five utterances ({five:.3f}) than with all available utterances ({all_speech:.3f}); brief-speech deployment performance is weaker in this sample.")
        except (KeyError, ValueError):
            pass
    if "matched_content" in results:
        observed = int(results["matched_content"]["predictions"].physical_speaker_id.nunique())
        full = int(results["primary_uaspeech_m3"]["predictions"].physical_speaker_id.nunique()) if "primary_uaspeech_m3" in results else observed
        if observed < full:
            limitations.append(f"Verified matched-content analysis includes {observed} of {full} M3 physical speakers because official transcript metadata did not support the others; it is a descriptive subset.")
    if "channels" in results:
        channel_metrics = results["channels"]["metrics"]
        channel_auc = channel_metrics.groupby("channel").ROC_AUC.mean()
        if len(channel_auc):
            limitations.append(f"Diagnostic M2–M8 mean ROC-AUC spans {float(channel_auc.min()):.3f}–{float(channel_auc.max()):.3f}; M3 remains the prespecified primary channel regardless of channel ranking.")
            for channel, auc in channel_auc.items():
                if float(auc) < .7:
                    limitations.append(f"CHANNEL ROBUSTNESS CONCERN: {channel} mean ROC-AUC is {float(auc):.3f}, below 0.70. This channel's poorer discrimination is retained in the table and figure.")
    if "primary_uaspeech_m3" in results:
        m = results["primary_uaspeech_m3"]["metrics"]
        auc, sens, spec = (float(m[metric].mean()) for metric in ("ROC_AUC", "Sensitivity", "Specificity"))
        if auc >= .9 and spec > sens:
            limitations.append(f"UASpeech M3 threshold-free discrimination is strong (mean ROC-AUC {auc:.3f}), while the locked operating point is conservative (sensitivity {sens:.3f}, specificity {spec:.3f}); no UASpeech threshold adjustment was made.")
    for name in ("silence", "duration", "artifact"):
        if name in results:
            groupcols = [c for c in ("corpus", "channel") if c in results[name]["metrics"]]
            for key, m in (results[name]["metrics"].groupby(groupcols, dropna=False) if groupcols else [("all", results[name]["metrics"])]):
                auc = float(m.ROC_AUC.mean())
                if auc >= .8:
                    limitations.append(f"POTENTIAL CONFOUND: {name} control {key} achieves mean ROC-AUC {auc:.3f}. Recording artifacts or duration may explain part of discrimination; this is not evidence for disease specificity.")
    limitations.extend(["Speaker bootstrap intervals condition on the observed corpus, diagnosis strata and fixed trained seeds; they do not establish prospective clinical performance.",
                        "UASpeech was studied in earlier project versions. These are frozen-model external cross-corpus evaluations, not historically untouched prospective validation.",
                        "Calibration bins contain few independent speakers. Repeated seeds do not increase the number of independent participants."])
    limitations.extend(failures)
    labels = {"PRIMARY TORGO TABLE": ["primary_torgo"], "ABLATION TABLE": ["ablations"],
              "UASPEECH M3 TABLE": ["primary_uaspeech_m3"], "CHANNEL TABLE": ["channels", "channels_matched"],
              "BUDGET TABLE": ["budgets_torgo", "budgets_uaspeech"],
              "SANITY TABLE": ["silence", "duration", "artifact", "matched_content"]}
    table_trace = {name: "PASS" if all(n in results and registered["datasets"].get(n, {}).get("status") == "PASS" for n in names) else "INCOMPLETE" for name, names in labels.items()}
    table_trace["FIGURE SOURCES"] = "PASS" if figure_ok and not failures else "FAIL"
    audit = dict(checks=checks, table_traceability=table_trace, remaining_limitations=list(dict.fromkeys(limitations)),
                 registered_results=registered, errors=failures, complete=all(value in ("YES", "PASS", "NOT ESTIMABLE") for key, value in checks.items() if not key.startswith("UASpeech used")))
    _write_json(out / "publication_audit.json", audit)
    text = "V8 PUBLICATION COMPLETION AUDIT\n===============================\n\n" + "\n\n".join(f"{key}:\n{value}" for key, value in checks.items())
    text += "\n\nTABLE AND FIGURE TRACEABILITY\n" + "\n".join(f"{key}: {value}" for key, value in table_trace.items())
    text += "\n\nREMAINING LIMITATIONS\n" + "\n".join("- "+item for item in audit["remaining_limitations"]) + "\n"
    (out / "publication_audit.txt").write_text(text)
    if strict and failures:
        raise AssertionError("Publication traceability failed: " + "; ".join(failures))
    return audit
