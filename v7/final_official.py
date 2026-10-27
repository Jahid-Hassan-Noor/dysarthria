"""Fresh zero-shot inference from the prespecified, frozen V7 Stream-1 fits.

The corrected architecture ablation is a comparison, not an external-model
selection procedure. This module never fits a parameter or chooses a threshold
using official UASpeech. Historical external outputs are left untouched.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .corrected_analysis import METRIC_COLUMNS, _single_metric
from .external import CHANNELS, PRIMARY_CHANNEL, _flush, load_frozen_models, load_official_metadata
from .methodology import OUT, SEEDS, SIGNATURE, THRESHOLD_SIGNATURE, metrics


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def lock_final_stream1():
    """Lock the existing TORGO-only deployment recipe before new external inference."""
    budget=json.loads((OUT/"v7_training_budget_decision.json").read_text())
    if budget["selected_max_epochs"]!=20 or budget["samples_per_epoch"]!=512:
        raise RuntimeError("Frozen deployment budget differs from TORGO convergence decision")
    records=load_frozen_models()
    lock={"model":"Prespecified V7 Stream 1 Only (CNN-BiLSTM)",
        "selection_basis":"V7 prespecification; no official UASpeech selection",
        "source_corpus":"TORGO only","seeds":list(SEEDS),
        "pipeline_signature":SIGNATURE,"threshold_signature":THRESHOLD_SIGNATURE,
        "training_budget_decision":"v7_training_budget_decision.json",
        "training_budget_sha256":_sha(OUT/"v7_training_budget_decision.json"),
        "checkpoint_sha256":{str(r["seed"]):_sha(r["checkpoint"]) for r in records},
        "locked_thresholds":{str(r["seed"]):r["threshold"] for r in records},
        "locked_temperatures":{str(r["seed"]):r["temperature"] for r in records}}
    path=OUT/"v7_final_model_lock.json"
    if path.exists() and json.loads(path.read_text())!=lock:
        raise RuntimeError("Final model lock changed; inspect the existing lock")
    path.write_text(json.dumps(lock,indent=2)+"\n")
    return records,lock


def _predict_channel(channel,meta,records,lock,batch_size=48):
    expected=meta[meta.microphone.eq(channel)].sort_values("file_path").copy()
    feature_dir=OUT/"external_features"
    manifest=json.loads((feature_dir/f"{channel}.manifest.json").read_text())
    file_hash=hashlib.sha256("\n".join(expected.file_path).encode()).hexdigest()
    if manifest["signature"]!=SIGNATURE or manifest["file_path_sha256"]!=file_hash or manifest["n"]!=len(expected):
        raise RuntimeError(f"Official feature cache provenance failed for {channel}")
    target=OUT/f"v7_final_official_{channel}_predictions.csv"
    manifest_path=target.with_suffix(".manifest.json")
    spec={"model_lock_sha256":_sha(OUT/"v7_final_model_lock.json"),
        "metadata_sha256":_sha(OUT.parent.parent/"v6"/"outputs"/"official_uaspeech_external_results"/"uaspeech_official_metadata.csv"),
        "channel":channel,"file_path_sha256":file_hash,
        "feature_manifest_sha256":_sha(feature_dir/f"{channel}.manifest.json"),
        "checkpoint_sha256":lock["checkpoint_sha256"]}
    if target.exists():
        if not manifest_path.exists() or json.loads(manifest_path.read_text())!=spec:
            raise RuntimeError(f"Stale final official prediction file: {target}")
        frame=pd.read_csv(target,low_memory=False)
        if len(frame)!=len(expected) or list(frame.file_path)!=list(expected.file_path):
            raise RuntimeError(f"Official prediction rows changed: {target}")
        return frame
    features=np.load(feature_dir/f"{channel}_logmel.npy",mmap_mode="r")
    lengths=np.load(feature_dir/f"{channel}_lengths.npy",mmap_mode="r")
    if len(features)!=len(expected) or len(lengths)!=len(expected):
        raise RuntimeError(f"Official feature shape changed for {channel}")
    columns=("file_path","physical_speaker_id","label","gender","severity",
        "intelligibility","block","word_code","microphone")
    output=[];batch=[];rows=[]
    for i,record in enumerate(expected.to_dict("records")):
        row={k:record[k] for k in columns}
        row["valid_frames"]=int(lengths[i])
        batch.append(features[i]);rows.append(row)
        if len(rows)==batch_size:
            _flush(records,batch,rows);output.extend(rows);batch=[];rows=[]
    if rows:_flush(records,batch,rows);output.extend(rows)
    frame=pd.DataFrame(output)
    frame.to_csv(target,index=False)
    manifest_path.write_text(json.dumps(spec,indent=2)+"\n")
    print("Fresh frozen official inference",channel,len(frame),flush=True)
    return frame


def run_final_official(bootstrap_reps=1000):
    """Infer all M2-M8 channels, score M3, and retain speaker-cluster CIs."""
    records,lock=lock_final_stream1()
    meta=load_official_metadata()
    predictions=pd.concat([_predict_channel(c,meta,records,lock) for c in CHANNELS],
        ignore_index=True)
    predictions.to_csv(OUT/"v7_final_official_uaspeech_predictions.csv",index=False)
    primary=predictions[predictions.microphone.eq(PRIMARY_CHANNEL)].copy()
    if primary.empty or primary.physical_speaker_id.nunique()<2:
        raise RuntimeError("Missing official M3 speaker evaluation population")
    rows=[];speaker_frames={}
    for rec in records:
        seed=rec["seed"];col=f"prob_seed{seed}"
        frame=primary[["physical_speaker_id","label",col]].rename(
            columns={col:"probability"})
        speaker=frame.groupby("physical_speaker_id",as_index=False).agg(
            label=("label","first"),probability=("probability","mean"))
        speaker["seed"]=seed;speaker["threshold"]=rec["threshold"]
        speaker_frames[seed]=speaker
        for level,part in (("utterance",frame),("physical_speaker",speaker)):
            rows.append(dict(seed=seed,level=level,channel=PRIMARY_CHANNEL,
                threshold=rec["threshold"],temperature=rec["temperature"],
                model_lock_sha256=_sha(OUT/"v7_final_model_lock.json"),
                source_prediction_file="v7_final_official_uaspeech_predictions.csv",
                **metrics(part.label,part.probability,rec["threshold"])))
    pd.concat(speaker_frames.values(),ignore_index=True).to_csv(
        OUT/"v7_final_official_uaspeech_speakers.csv",index=False)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_final_official_uaspeech_metrics.csv",index=False)
    make_official_master_table(result)
    summary=[]
    for level,part in result.groupby("level"):
        for metric in METRIC_COLUMNS:
            summary.append(dict(level=level,metric=metric,mean=part[metric].mean(),
                sd=part[metric].std(ddof=1),n_seeds=len(part),
                source_metrics_file="v7_final_official_uaspeech_metrics.csv"))
    pd.DataFrame(summary).to_csv(OUT/"v7_final_official_uaspeech_mean_sd.csv",index=False)
    # One paired resample is shared across all seeds for each replicate.
    reference=speaker_frames[SEEDS[0]].sort_values("physical_speaker_id")
    ordered={seed:frame.sort_values("physical_speaker_id") for seed,frame in speaker_frames.items()}
    if any(not frame.physical_speaker_id.equals(reference.physical_speaker_id)
            for frame in ordered.values()):
        raise RuntimeError("Official speaker set differs across seeds")
    rng=np.random.default_rng(20260929);samples=[]
    for _ in range(bootstrap_reps):
        ix=rng.integers(0,len(reference),len(reference))
        if len(np.unique(reference.label.to_numpy()[ix]))<2:continue
        samples.append(ix)
    ci=[]
    for metric in METRIC_COLUMNS:
        for seed,frame in ordered.items():
            y=frame.label.to_numpy(int);p=frame.probability.to_numpy(float)
            t=float(frame.threshold.iloc[0])
            values=np.asarray([_single_metric(y[ix],p[ix],t,metric) for ix in samples],float)
            values=values[np.isfinite(values)]
            ci.append(dict(level="physical_speaker",seed=seed,metric=metric,
                point=_single_metric(y,p,t,metric),ci_low=np.percentile(values,2.5) if len(values) else np.nan,
                ci_high=np.percentile(values,97.5) if len(values) else np.nan,
                n_valid_bootstrap=len(values),n_physical_speakers=len(frame)))
        five=[]
        for ix in samples:
            value=np.mean([_single_metric(frame.label.to_numpy(int)[ix],
                frame.probability.to_numpy(float)[ix],float(frame.threshold.iloc[0]),metric)
                for frame in ordered.values()])
            five.append(value)
        five=np.asarray(five,float);five=five[np.isfinite(five)]
        ci.append(dict(level="physical_speaker",seed="five_seed_mean",metric=metric,
            point=float(np.mean([_single_metric(frame.label,frame.probability,
                float(frame.threshold.iloc[0]),metric) for frame in ordered.values()])),
            ci_low=np.percentile(five,2.5) if len(five) else np.nan,
            ci_high=np.percentile(five,97.5) if len(five) else np.nan,
            n_valid_bootstrap=len(five),n_physical_speakers=len(reference)))
    pd.DataFrame(ci).to_csv(OUT/"v7_final_official_speaker_cluster_ci.csv",index=False)
    predictions["content_key"]=(predictions.physical_speaker_id.astype(str)+"|"+
        predictions.block.astype(str)+"|"+predictions.word_code.astype(str))
    keys=set.intersection(*[set(predictions.loc[predictions.microphone.eq(c),"content_key"])
        for c in CHANNELS])
    matched=predictions[predictions.content_key.isin(keys)].copy()
    if matched.empty:raise RuntimeError("No M2-M8 matched speaker/content subset")
    matched.to_csv(OUT/"v7_final_official_matched_channel_predictions.csv",index=False)
    channel_rows=[]
    for channel,part in matched.groupby("microphone"):
        for rec in records:
            seed=rec["seed"];col=f"prob_seed{seed}"
            utter=part[["physical_speaker_id","label",col]].rename(columns={col:"probability"})
            speaker=utter.groupby("physical_speaker_id",as_index=False).agg(
                label=("label","first"),probability=("probability","mean"))
            for level,frame in (("utterance",utter),("physical_speaker",speaker)):
                channel_rows.append(dict(seed=seed,channel=channel,level=level,
                    n_physical_speakers=frame.physical_speaker_id.nunique(),
                    source_prediction_file="v7_final_official_matched_channel_predictions.csv",
                    **metrics(frame.label,frame.probability,rec["threshold"])))
    pd.DataFrame(channel_rows).to_csv(OUT/"v7_final_official_matched_channel_metrics.csv",index=False)
    audit={"model_lock_sha256":_sha(OUT/"v7_final_model_lock.json"),
        "official_metadata_sha256":_sha(OUT.parent.parent/"v6"/"outputs"/"official_uaspeech_external_results"/"uaspeech_official_metadata.csv"),
        "official_uaspeech_used_for_training":False,
        "official_uaspeech_used_for_fine_tuning":False,
        "official_uaspeech_used_for_threshold_selection":False,
        "official_uaspeech_used_for_calibration":False,
        "official_uaspeech_used_for_model_selection":False,
        "seeds":list(SEEDS),"primary_channel":PRIMARY_CHANNEL,
        "matched_content_keys":len(keys),
        "matched_physical_speakers":matched.physical_speaker_id.nunique(),
        "prediction_source":"v7_final_official_uaspeech_predictions.csv"}
    (OUT/"v7_final_official_zero_shot_audit.json").write_text(json.dumps(audit,indent=2)+"\n")
    verify_historical_official_reproduction()
    return result


def make_official_master_table(result=None):
    """Five seed rows plus a directly traceable mean ± SD row per level."""
    if result is None:
        result=pd.read_csv(OUT/"v7_final_official_uaspeech_metrics.csv")
    rows=[]
    for level,part in result.groupby("level"):
        if len(part)!=5 or set(part.seed)!=set(SEEDS):
            raise RuntimeError(f"Official table lacks five seeds at {level}")
        for record in part.sort_values("seed").to_dict("records"):
            rows.append(dict(level=level,seed=f"seed {record['seed']}",
                n_evaluation_units=int(record["n"]),
                source_metrics_file="v7_final_official_uaspeech_metrics.csv",
                **{metric:f"{record[metric]:.3f}" if np.isfinite(record[metric]) else "NA"
                    for metric in METRIC_COLUMNS}))
        rows.append(dict(level=level,seed="mean ± SD",
            n_evaluation_units=int(part.n.iloc[0]),
            source_metrics_file="v7_final_official_uaspeech_metrics.csv",
            **{metric:f"{part[metric].mean():.3f} ± {part[metric].std(ddof=1):.3f}"
                if part[metric].notna().sum()>1 else "NA"
                for metric in METRIC_COLUMNS}))
    table=pd.DataFrame(rows)
    table.to_csv(OUT/"v7_final_official_uaspeech_master_table.csv",index=False)
    return table


def verify_historical_official_reproduction():
    """Compare fresh frozen inference with preserved V7 channel predictions."""
    rows=[]
    columns=["file_path"]+[f"prob_seed{seed}" for seed in SEEDS]
    for channel in CHANNELS:
        historical=OUT/"external_predictions"/f"official_{channel}.csv"
        current=OUT/f"v7_final_official_{channel}_predictions.csv"
        if not historical.exists() or not current.exists():
            raise FileNotFoundError(f"Missing historical or fresh official channel {channel}")
        old=pd.read_csv(historical,usecols=columns)
        new=pd.read_csv(current,usecols=columns)
        joined=old.merge(new,on="file_path",suffixes=("_historical","_final"),
            validate="one_to_one")
        if len(joined)!=len(old) or len(joined)!=len(new):
            raise AssertionError(f"Official channel {channel} row sets differ")
        for seed in SEEDS:
            delta=np.abs(joined[f"prob_seed{seed}_historical"]-
                joined[f"prob_seed{seed}_final"])
            rows.append(dict(channel=channel,seed=seed,n=len(joined),
                max_absolute_probability_difference=float(delta.max()),
                historical_file=f"external_predictions/official_{channel}.csv",
                final_file=f"v7_final_official_{channel}_predictions.csv"))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_final_official_historical_reproduction.csv",index=False)
    return result
