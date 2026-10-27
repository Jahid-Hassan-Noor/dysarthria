"""Controlled V8 component ablations and paired physical-speaker inference.

All training, epoch selection, threshold selection, bag sampling and chunked
inference are delegated to the current V8 notebook. Only the two components
under study change. B2 is the previously completed, frozen reference, never a
new fit. This module accepts the validated publication Context from common.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch


ARMS = {
    "A1": (False, False, "WavLM Last Layer + Mean Pooling"),
    "A2": (False, True, "WavLM Last Layer + Attention"),
    "B1": (True, False, "WavLM Last-4 Mix + Mean Pooling"),
    "B2": (True, True, "WavLM Last-4 Mix + Attention (V8 Reference)"),
}
METRICS = ("Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity",
           "F1_dysarthric", "ROC_AUC", "PR_AUC", "Brier", "ECE")
DIFFERENCES = ("Balanced_Accuracy", "ROC_AUC", "PR_AUC", "Brier")


def make_ablation_head(base_class, arm: str):
    """Preserve dimensions and initialization; replace just mix and/or weights.

    The reference concatenates weighted pooling with mean context. In a mean
    arm both slots receive the mean: [context, context]. No smaller head is
    introduced. Unused parameters remain instantiated, to preserve the random
    initialization of every downstream weight, and are frozen.
    """
    use_mix, use_attention, _ = ARMS[arm]

    class ComponentAblation(base_class):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if not use_mix:
                self.layer_logits.requires_grad_(False)
            if not use_attention:
                for parameter in self.attention.parameters():
                    parameter.requires_grad_(False)

        def forward(self, speech, acoustic=None, valid_mask=None, return_attention=False):
            if speech.ndim != 3 or speech.shape[0] == 0:
                raise ValueError("Expected one nonempty speaker bag [utterance, layer, feature]")
            if speech.shape[1] != self.layer_logits.numel():
                raise ValueError("Embedding cache layer count does not match the reference")
            if valid_mask is None:
                valid_mask = torch.ones(speech.shape[0], dtype=torch.bool, device=speech.device)
            if valid_mask.sum() == 0:
                raise ValueError("Speaker bag has no valid utterances")
            if use_mix:
                mix = torch.softmax(self.layer_logits, dim=0)
                representation = (speech * mix[None, :, None]).sum(dim=1)
            else:
                representation = speech[:, -1, :]
            utterance = self.speech_proj(representation)
            mask_f = valid_mask.to(utterance.dtype)[:, None]
            context = (utterance * mask_f).sum(dim=0) / mask_f.sum()
            if use_attention:
                expanded = context[None, :].expand_as(utterance)
                score = self.attention(torch.cat((utterance, expanded, utterance * expanded), dim=-1)).squeeze(-1)
                score = score.masked_fill(~valid_mask, torch.finfo(score.dtype).min)
                weight = torch.softmax(score, dim=0)
            else:
                weight = valid_mask.to(utterance.dtype) / mask_f.sum()
            pooled = torch.cat(((utterance * weight[:, None]).sum(dim=0), context), dim=0)
            if self.use_fusion:
                if acoustic is None or acoustic.shape[0] != speech.shape[0]:
                    raise ValueError("Fusion requires aligned utterance acoustic features")
                acoustic_mean = (acoustic * mask_f).sum(dim=0) / mask_f.sum()
                acoustic_weighted = (acoustic * weight[:, None]).sum(dim=0)
                pooled = torch.cat((pooled, self.acoustic_proj(torch.cat((acoustic_weighted, acoustic_mean)))), dim=0)
            logit = self.classifier(pooled).squeeze(-1)
            return (logit, weight) if return_attention else logit

    return ComponentAblation


@contextmanager
def component_override(pipeline, arm: str):
    original = pipeline.SpeakerAttentionClassifier
    pipeline.SpeakerAttentionClassifier = make_ablation_head(original, arm)
    try:
        yield
    finally:
        pipeline.SpeakerAttentionClassifier = original


def verify_reference_splits(ctx) -> tuple[list[dict], pd.DataFrame]:
    """Regenerate notebook splits and require equality to every frozen audit."""
    config, pipeline = ctx.config, ctx.pipeline
    plans, rows = [], []
    all_speakers = set(ctx.meta.physical_speaker_id)
    for seed in config.seeds:
        seen = set()
        for outer, train_ids, test_ids in pipeline.grouped_speaker_splits(ctx.meta, seed, config.outer_folds):
            audit_path = ctx.run_dir / "folds" / "speech_only" / f"seed{seed}_outer{outer}" / "fold_audit.json"
            audit = json.loads(audit_path.read_text())
            if (set(audit["train_physical_speakers"]) != train_ids
                    or set(audit["test_physical_speakers"]) != test_ids
                    or not audit.get("complete")):
                raise AssertionError(f"Frozen outer membership differs: {audit_path}")
            if not train_ids.isdisjoint(test_ids) or train_ids | test_ids != all_speakers or seen & test_ids:
                raise AssertionError("Outer physical-speaker leakage or incomplete partition")
            seen |= test_ids
            train_meta = ctx.meta[ctx.meta.physical_speaker_id.isin(train_ids)]
            inner_splits = list(pipeline.grouped_speaker_splits(train_meta, seed + 731, config.inner_folds))
            frozen_inner = {item["inner_fold"]: item for item in audit["inner_folds"]}
            inner_seen = set()
            for inner, inner_train, inner_val in inner_splits:
                if not (inner_train.isdisjoint(inner_val) and inner_train.isdisjoint(test_ids)
                        and inner_val.isdisjoint(test_ids)):
                    raise AssertionError("Nested physical-speaker leakage")
                if inner_train | inner_val != train_ids or inner_seen & inner_val:
                    raise AssertionError("Inner partitions do not cover outer training speakers once")
                inner_seen |= inner_val
                saved = frozen_inner[inner]
                if set(saved["train_physical_speakers"]) != inner_train or set(saved["validation_physical_speakers"]) != inner_val:
                    raise AssertionError(f"Frozen inner membership differs: {audit_path}, inner {inner}")
                for role, ids in (("train", inner_train), ("validation", inner_val), ("test", test_ids)):
                    for speaker in sorted(ids):
                        rows.append(dict(seed=seed, outer_fold=outer, inner_fold=inner, role=role,
                                         physical_speaker_id=speaker, reference_audit=str(audit_path)))
            if inner_seen != train_ids or len(frozen_inner) != len(inner_splits):
                raise AssertionError("Inner folds missing or duplicated")
            plans.append(dict(seed=seed, outer=outer, train_speakers=train_ids, test_speakers=test_ids))
        if seen != all_speakers:
            raise AssertionError("Outer test speakers do not cover the reference cohort")
    return plans, pd.DataFrame(rows)


def _metric_values(y: np.ndarray, probability: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Exact notebook metrics along the final axis, vectorized across draws.

    ROC-AUC includes half credit for ties; PR-AUC is sklearn's noninterpolated
    average precision, with equal-score recordings entering at one threshold.
    ECE uses the identical ten fixed-width probability bins as V8.
    """
    y = np.broadcast_to(y, probability.shape)
    positive = y == 1
    negative = ~positive
    decision = probability >= thresholds
    tp = (decision & positive).sum(axis=-1)
    fp = (decision & negative).sum(axis=-1)
    fn = (~decision & positive).sum(axis=-1)
    tn = (~decision & negative).sum(axis=-1)
    npos, nneg = positive.sum(axis=-1), negative.sum(axis=-1)
    if np.any(npos == 0) or np.any(nneg == 0):
        raise ValueError("Paired stratified metrics require both diagnoses")
    sensitivity, specificity = tp / npos, tn / nneg
    auc_sum = np.zeros(probability.shape[:-1])
    # At most 15 TORGO speakers, keeping the bootstrap footprint small.
    for index in range(probability.shape[-1]):
        difference = probability[..., index, None] - probability
        concordance = (difference > 0) + .5 * (difference == 0)
        auc_sum += positive[..., index] * (concordance * negative).sum(axis=-1)
    order = np.argsort(-probability, axis=-1, kind="stable")
    sorted_p = np.take_along_axis(probability, order, axis=-1)
    sorted_y = np.take_along_axis(positive, order, axis=-1)
    cumulative_tp = np.cumsum(sorted_y, axis=-1)
    group_end = np.concatenate((sorted_p[..., :-1] != sorted_p[..., 1:],
                                np.ones(sorted_p.shape[:-1] + (1,), dtype=bool)), axis=-1)
    ap, preceding_tp = np.zeros(probability.shape[:-1]), np.zeros(probability.shape[:-1])
    for index in range(probability.shape[-1]):
        at_end = group_end[..., index]
        current_tp = cumulative_tp[..., index]
        ap += np.where(at_end, (current_tp - preceding_tp) / npos * current_tp / (index + 1), 0)
        preceding_tp = np.where(at_end, current_tp, preceding_tp)
    bins = np.minimum((probability * 10).astype(int), 9)
    ece = sum(np.abs(((y - probability) * (bins == bin_id)).sum(axis=-1))
              / probability.shape[-1] for bin_id in range(10))
    denominator = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denominator, out=np.zeros_like(tp, dtype=float), where=denominator != 0)
    return np.stack(((tp + tn) / probability.shape[-1], (sensitivity + specificity) / 2,
                     sensitivity, specificity, f1, auc_sum / (npos * nneg), ap,
                     np.mean((probability - y) ** 2, axis=-1), ece), axis=-1)


def paired_statistics(predictions: pd.DataFrame, *, resamples: int = 2000,
                      bootstrap_seed: int = 20261013) -> dict[str, pd.DataFrame | dict]:
    """Stratified paired cluster bootstrap of mean-of-seed performance.

    One diagnosis-stratified physical-speaker draw is shared across every arm
    and every seed. Seeds are repeated fits, never treated as new subjects.
    """
    if resamples < 2:
        raise ValueError("At least two bootstrap resamples required")
    needed = {"model_arm", "seed", "physical_speaker_id", "label", "probability", "threshold"}
    if not needed.issubset(predictions.columns):
        raise ValueError(f"Missing columns: {sorted(needed - set(predictions.columns))}")
    if predictions.duplicated(["model_arm", "seed", "physical_speaker_id"]).any():
        raise AssertionError("Duplicate speaker in arm/seed")
    if set(predictions.model_arm) != set(ARMS):
        raise AssertionError("All four arms are required for a complete component comparison")
    identities = predictions[["physical_speaker_id", "label"]].drop_duplicates().sort_values("physical_speaker_id")
    if identities.physical_speaker_id.duplicated().any() or set(identities.label) != {0, 1}:
        raise AssertionError("Inconsistent physical-speaker diagnosis")
    speakers = identities.physical_speaker_id.tolist()
    seeds = sorted(predictions.seed.unique())
    arrays = []
    per_seed_rows = []
    y = identities.label.to_numpy(dtype=int)
    for arm in ARMS:
        arm_arrays = []
        for seed in seeds:
            frame = predictions[(predictions.model_arm == arm) & (predictions.seed == seed)].set_index("physical_speaker_id")
            if set(frame.index) != set(speakers):
                raise AssertionError("All arms/seeds must contain the same physical speakers")
            frame = frame.loc[speakers]
            if not np.array_equal(frame.label.to_numpy(dtype=int), y):
                raise AssertionError("Label mismatch in paired cohort")
            p, t = frame.probability.to_numpy(float), frame.threshold.to_numpy(float)
            if not np.isfinite(p).all() or not np.isfinite(t).all() or np.any((p < 0) | (p > 1)):
                raise ValueError("Invalid probabilities or thresholds")
            values = _metric_values(y, p, t)
            per_seed_rows.append(dict(model_arm=arm, model_name=ARMS[arm][2], seed=int(seed),
                                      n_speakers=len(speakers), **dict(zip(METRICS, values))))
            arm_arrays.append(np.stack((p, t)))
        arrays.append(arm_arrays)
    arrays = np.asarray(arrays)  # arm, seed, (probability, threshold), speaker
    rng = np.random.default_rng(bootstrap_seed)
    draws = np.concatenate([rng.choice(np.flatnonzero(y == diagnosis),
                                      size=(resamples, np.count_nonzero(y == diagnosis)), replace=True)
                            for diagnosis in (0, 1)], axis=1)
    per_seed = pd.DataFrame(per_seed_rows)
    boot = np.empty((resamples, len(ARMS), len(METRICS)))
    for ai, arm in enumerate(ARMS):
        sampled_p = arrays[ai, :, 0, :][:, draws]
        sampled_t = arrays[ai, :, 1, :][:, draws]
        boot[:, ai, :] = _metric_values(y[draws], sampled_p, sampled_t).mean(axis=0)
    means = np.array([per_seed[per_seed.model_arm == arm][list(METRICS)].mean().to_numpy() for arm in ARMS])
    summary, ci, paired, factorial, distributions = [], [], [], [], []
    for ai, arm in enumerate(ARMS):
        rows = per_seed[per_seed.model_arm == arm]
        for mi, metric in enumerate(METRICS):
            low, high = np.percentile(boot[:, ai, mi], [2.5, 97.5])
            summary.append(dict(model_arm=arm, model_name=ARMS[arm][2], metric=metric,
                                mean=means[ai, mi], sd=rows[metric].std(ddof=1), n_seeds=len(seeds),
                                lower_95=low, upper_95=high, n_speakers=len(speakers)))
            ci.append(dict(model_arm=arm, metric=metric, lower_95=low, upper_95=high,
                           n_resamples=resamples, resampling_unit="physical_speaker",
                           stratified_by_diagnosis=True, same_draw_across_arms_and_seeds=True))
            for repetition in range(resamples):
                distributions.append(dict(repetition=repetition, model_arm=arm, metric=metric,
                                          value=boot[repetition, ai, mi]))
            if arm != "B2" and metric in DIFFERENCES:
                delta = boot[:, ai, mi] - boot[:, 3, mi]
                low, high = np.percentile(delta, [2.5, 97.5])
                paired.append(dict(model_arm=arm, reference="B2", metric=metric,
                                   difference="ablation_minus_reference", delta=means[ai, mi] - means[3, mi],
                                   lower_95=low, upper_95=high, n_resamples=resamples))
    contrasts = {
        "layer_mix_main_effect": np.array([-.5, -.5, .5, .5]),
        "attention_main_effect": np.array([-.5, .5, -.5, .5]),
        "layer_mix_attention_interaction": np.array([1., -1., -1., 1.]),
    }
    for effect, coefficient in contrasts.items():
        for mi, metric in enumerate(METRICS):
            delta = boot[:, :, mi] @ coefficient
            low, high = np.percentile(delta, [2.5, 97.5])
            factorial.append(dict(effect=effect, metric=metric, delta=means[:, mi] @ coefficient,
                                  lower_95=low, upper_95=high, n_resamples=resamples))
    draw_table = pd.DataFrame(draws, columns=[f"draw_{i}" for i in range(len(speakers))])
    draw_table = draw_table.map(lambda index: speakers[index])
    draw_table.insert(0, "repetition", np.arange(resamples))
    return dict(per_seed_metrics=per_seed, summary=pd.DataFrame(summary),
                speaker_bootstrap_ci=pd.DataFrame(ci), paired_differences=pd.DataFrame(paired),
                factorial_effects=pd.DataFrame(factorial), bootstrap_draws=draw_table,
                bootstrap_distributions=pd.DataFrame(distributions),
                bootstrap_protocol=dict(seed=bootstrap_seed, resamples=resamples,
                                        estimand="mean of per-seed metrics", ci="percentile 95%",
                                        resampling_unit="physical_speaker", stratified_by_diagnosis=True,
                                        same_draw_across_arms_and_seeds=True,
                                        interpretation="Brier decreases indicate improvement; other metrics increase."))


def run(ctx, *, arms=("A1", "A2", "B1"), resamples: int = 2000):
    """Execute resumable TORGO-only experiments after the snapshot is verified."""
    if not getattr(ctx, "frozen_verified", False):
        raise RuntimeError("Freeze and verify the existing V8 reference before running ablations")
    if any(arm not in {"A1", "A2", "B1"} for arm in arms):
        raise ValueError("Only A1/A2/B1 may be trained; B2 always reuses the frozen reference")
    if ctx.config.use_fusion or ctx.config.selected_layers != 4:
        raise ValueError("The requested ablation requires the frozen speech-only last-four reference")
    if tuple(ctx.config.seeds) != (42, 43, 44, 45, 46):
        raise ValueError("Publication ablations require the five original V8 seeds")
    pipeline, output = ctx.pipeline, ctx.output_dir / "ablations"
    output.mkdir(parents=True, exist_ok=True)
    plans, split_rows = verify_reference_splits(ctx)
    split_path = output / "split_membership.csv"
    pipeline._atomic_csv(split_path, split_rows)
    implementation_sha = pipeline._file_sha256(Path(__file__))
    protocol = dict(reference_run_signature=ctx.run_signature, config=asdict(ctx.config),
                    ablation_implementation_sha256=implementation_sha,
                    notebook_sha256=getattr(ctx, "notebook_sha256", None),
                    feature_signature=ctx.manifest.get("feature_signature", ctx.manifest.get("feature_spec")),
                    device=str(ctx.device), reference_arm="B2", reference_retrained=False,
                    inner_folds_identical_to_frozen_reference=True,
                    mean_pooling="concatenate context with itself, preserving classifier dimensions",
                    inference="unchanged V8 deterministic speaker-hash chunking, utterance-weighted mean logit",
                    unrelated_parameters="identical initialization, unused component parameters frozen",
                    external_data_used_for_any_development=False,
                    split_manifest=str(split_path), split_manifest_sha256=pipeline._file_sha256(split_path))
    protocol_path = output / "protocol.json"
    if protocol_path.exists():
        prior = json.loads(protocol_path.read_text())
        if prior != protocol:
            raise RuntimeError("Ablation protocol changed; refusing to overwrite existing experiment")
    else:
        pipeline._atomic_json(protocol_path, protocol)
    reference = pd.read_csv(ctx.run_dir / "v8_speaker_oof_predictions.csv")
    reference = reference[reference.model_arm.eq("speech_only")].copy()
    reference["model_arm"] = "B2"
    all_predictions = [reference]
    started = time.monotonic()
    for arm in arms:
        arm_dir = output / arm
        arm_signature = pipeline._canonical_hash(dict(protocol=protocol, arm=arm,
                                                       layer_mix=ARMS[arm][0], attention=ARMS[arm][1]))
        arm_predictions = []
        with component_override(pipeline, arm):
            for position, plan in enumerate(plans):
                print(f"Publication {arm} fold {position + 1}/{len(plans)}; elapsed {time.monotonic()-started:.1f}s", flush=True)
                fold = pipeline._run_outer_fold(ctx.config, ctx.meta, ctx.speech, ctx.acoustic,
                                                **plan, use_fusion=False, run_dir=arm_dir,
                                                run_signature=arm_signature, device=ctx.device)
                fold = fold.copy()
                fold["model_arm"] = arm
                fold["ablation_run_signature"] = arm_signature
                arm_predictions.append(fold)
        combined = pd.concat(arm_predictions, ignore_index=True)
        pipeline._atomic_csv(arm_dir / "speaker_oof_predictions.csv", combined)
        all_predictions.append(combined)
    # A selective invocation can resume previous complete arms, without refitting.
    for arm in {"A1", "A2", "B1"} - set(arms):
        saved = output / arm / "speaker_oof_predictions.csv"
        if saved.exists():
            all_predictions.append(pd.read_csv(saved))
    predictions = pd.concat(all_predictions, ignore_index=True)
    checkpoint_rows = []
    for index, row in predictions.iterrows():
        base = ctx.run_dir if row.model_arm == "B2" else output / row.model_arm
        fold_dir = base / "folds" / "speech_only" / f"seed{int(row.seed)}_outer{int(row.outer_fold)}"
        checkpoint = fold_dir / "final" / "best.pt"
        audit_path = fold_dir / "fold_audit.json"
        predictions.loc[index, "checkpoint_path"] = str(checkpoint.resolve())
        predictions.loc[index, "checkpoint_sha256"] = pipeline._file_sha256(checkpoint)
        predictions.loc[index, "split_manifest"] = str(audit_path.resolve())
        predictions.loc[index, "split_manifest_sha256"] = pipeline._file_sha256(audit_path)
        predictions.loc[index, "model_name"] = ARMS[row.model_arm][2]
    predictions["run_signature"] = ctx.run_signature
    predictions["corpus"] = "TORGO"
    predictions["analysis"] = "component_ablation"
    prediction_path = output / "speaker_oof_predictions.csv"
    pipeline._atomic_csv(prediction_path, predictions)
    for (arm, seed, outer), frame in predictions.groupby(["model_arm", "seed", "outer_fold"]):
        row = frame.iloc[0]
        checkpoint_rows.append(dict(model_arm=arm, seed=int(seed), outer_fold=int(outer),
                                    checkpoint_path=row.checkpoint_path, checkpoint_sha256=row.checkpoint_sha256,
                                    split_manifest=row.split_manifest, split_manifest_sha256=row.split_manifest_sha256))
    pipeline._atomic_csv(output / "checkpoint_index.csv", pd.DataFrame(checkpoint_rows))
    complete = set(predictions.model_arm) == set(ARMS)
    if complete:
        statistics = paired_statistics(predictions, resamples=resamples)
        for name, value in statistics.items():
            if isinstance(value, pd.DataFrame):
                pipeline._atomic_csv(output / f"{name}.csv", value)
            else:
                pipeline._atomic_json(output / f"{name}.json", value)
    pipeline._atomic_json(output / "status.json", dict(complete=complete,
        completed_arms=sorted(predictions.model_arm.unique()), reference_retrained=False,
        elapsed_seconds=time.monotonic()-started, prediction_path=str(prediction_path),
        prediction_sha256=pipeline._file_sha256(prediction_path), protocol_path=str(protocol_path),
        run_signature=ctx.run_signature, all_five_seeds=all(
            set(group.seed) == set(ctx.config.seeds) for _, group in predictions.groupby("model_arm"))))
    try:
        from .common import publish_result
    except ImportError:
        publish_result = None
    if publish_result is not None:
        publish_result(ctx, "ablations", predictions, group_cols=["corpus", "analysis", "model_arm", "seed"])
    return predictions
