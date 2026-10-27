"""Read-only official UASpeech inference and V7 reporting.

No function in this file fits parameters using official UASpeech labels.
"""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .methodology import (OUT, ROOT, SEEDS, SIGNATURE, THRESHOLD_SIGNATURE, Stream1Model,
                          _logmel, apply_temperature, device, metrics,
                          speaker_aggregate, speaker_cluster_ci)

PRIMARY_CHANNEL = "M3"  # inherited from the V6 prespecified official evaluation
CHANNELS = tuple(f"M{i}" for i in range(2, 9))
OFFICIAL_META = ROOT / "v6" / "outputs" / "official_uaspeech_external_results" / "uaspeech_official_metadata.csv"


def load_official_metadata():
    if not OFFICIAL_META.exists():
        raise FileNotFoundError(OFFICIAL_META)
    df = pd.read_csv(OFFICIAL_META, low_memory=False)
    expected = {"file_path", "physical_speaker_id", "label", "severity", "microphone", "block", "word_code", "soundfile_info_ok"}
    if not expected.issubset(df.columns):
        raise AssertionError("Official metadata columns changed")
    df = df[df.soundfile_info_ok.astype(str).str.lower().eq("true")].copy()
    df = df[df.microphone.isin(CHANNELS)].copy()
    if not df.file_path.map(lambda p: Path(p).is_file()).all():
        raise FileNotFoundError("Official audio path in metadata is missing")
    if df.groupby("physical_speaker_id").label.nunique().max() != 1:
        raise AssertionError("Official physical speaker has conflicting labels")
    source_xls=ROOT/"UASpeech"/"UASpeech_normalized_FM"/"doc"/"speaker_wordlist.xls"
    if not source_xls.is_file():
        raise FileNotFoundError(f"Official severity/intelligibility source missing: {source_xls}")
    allowed={"high","mid","low","very_low","not_applicable_control","not_available"}
    if not set(df.severity.dropna()).issubset(allowed):
        raise AssertionError("Unexpected official severity category")
    if df.groupby("physical_speaker_id").severity.nunique().max()!=1:
        raise AssertionError("Official severity varies within physical speaker")
    # Verify this is the official normalized release already used in V6.
    variants = {Path(p).parents[3].name for p in df.file_path}
    if variants != {"UASpeech_normalized_C", "UASpeech_normalized_FM"}:
        raise AssertionError(f"Unexpected UASpeech release variants: {variants}")
    return df


def load_frozen_models():
    records = []
    for seed in SEEDS:
        path = OUT / f"deployment_seed{seed}" / "checkpoint.pt"
        if not path.is_file():
            raise RuntimeError(f"Missing TORGO-only frozen checkpoint for seed {seed}: {path}")
        b = torch.load(path, map_location="cpu", weights_only=False)
        if (b.get("pipeline_signature") != SIGNATURE or b.get("threshold_search_signature") != THRESHOLD_SIGNATURE
                or b.get("source_corpus") != "TORGO only" or b.get("seed") != seed):
            raise RuntimeError(f"Frozen checkpoint provenance failed for seed {seed}")
        model = Stream1Model().to(device()).eval()
        model.load_state_dict(b["model_state_dict"])
        records.append(dict(seed=seed,model=model,threshold=float(b["threshold"]),temperature=float(b["temperature"]),checkpoint=str(path)))
    return records


def _external_logmel(path):
    """Match V6 external preprocessing: load/trim/normalize, mic='unknown'."""
    import librosa
    audio,_ = librosa.load(path,sr=16000)
    audio,_ = librosa.effects.trim(audio,top_db=25)
    if len(audio)==0: audio=np.zeros(800,dtype=np.float32)
    audio=librosa.util.normalize(audio).astype(np.float32)
    return _logmel(audio)


def precompute_official_features(meta=None):
    """Cache fixed V6 external features; reads audio only, never fits labels."""
    meta=load_official_metadata() if meta is None else meta
    cache=OUT/"external_features";cache.mkdir(exist_ok=True)
    for channel in CHANNELS:
        expected=meta[meta.microphone.eq(channel)].sort_values("file_path")
        signature=hashlib.sha256("\n".join(expected.file_path).encode()).hexdigest()
        feature_path=cache/f"{channel}_logmel.npy";length_path=cache/f"{channel}_lengths.npy"
        manifest=cache/f"{channel}.manifest.json";progress=cache/f"{channel}.progress.json"
        spec={"signature":SIGNATURE,"file_path_sha256":signature,"n":len(expected),
              "preprocessing":"V6 official mic unknown; 25 dB trim; normalize; 80x128 log-mel"}
        if manifest.exists() and json.loads(manifest.read_text())==spec and feature_path.exists() and length_path.exists():
            print("Official feature cache verified",channel,len(expected));continue
        start=0
        if progress.exists() and feature_path.exists() and length_path.exists():
            state=json.loads(progress.read_text())
            if state.get("spec")==spec:start=int(state["next_index"])
        mode="r+" if start else "w+"
        features=np.lib.format.open_memmap(feature_path,mode=mode,dtype=np.float32,shape=(len(expected),80,128))
        lengths=np.lib.format.open_memmap(length_path,mode=mode,dtype=np.int16,shape=(len(expected),))
        for i,path in enumerate(expected.file_path.iloc[start:],start=start):
            lm,n=_external_logmel(path);features[i]=lm;lengths[i]=n
            if (i+1)%500==0:
                features.flush();lengths.flush();progress.write_text(json.dumps({"spec":spec,"next_index":i+1}))
                print("Cached",channel,i+1,"/",len(expected),flush=True)
        features.flush();lengths.flush();manifest.write_text(json.dumps(spec,indent=2))
        if progress.exists():progress.unlink()
        print("Official feature cache completed",channel,len(expected),flush=True)
    return cache


def _flush(records, batch, rows):
    x=torch.as_tensor(np.stack(batch),device=device()).unsqueeze(1)
    ln=torch.as_tensor(np.array([r["valid_frames"] for r in rows]),device=device())
    with torch.no_grad():
        for rec in records:
            raw=torch.softmax(rec["model"](x,ln),1)[:,1].cpu().numpy()
            calibrated=apply_temperature(raw,rec["temperature"])
            for row,p,praw in zip(rows,calibrated,raw):
                row[f"prob_seed{rec['seed']}"]=float(p)
                row[f"prob_raw_seed{rec['seed']}"]=float(praw)


def predict_official(meta=None, records=None, batch_size=48):
    """All seven channels; checkpoint signature prevents stale prediction reuse."""
    meta=load_official_metadata() if meta is None else meta
    records=load_frozen_models() if records is None else records
    cache=OUT/"external_predictions";cache.mkdir(exist_ok=True)
    feature_cache=OUT/"external_features"
    checkpoint_hashes={str(r["seed"]):hashlib.sha256(Path(r["checkpoint"]).read_bytes()).hexdigest() for r in records}
    frames=[]
    for channel in CHANNELS:
        target=cache/f"official_{channel}.csv"
        manifest=cache/f"official_{channel}.manifest.json"
        expected=meta[meta.microphone.eq(channel)].copy().sort_values("file_path")
        expected_manifest={"pipeline_signature":SIGNATURE,"checkpoint_sha256":checkpoint_hashes,
                           "file_path_sha256":hashlib.sha256("\n".join(expected.file_path).encode()).hexdigest(),
                           "preprocessing":"V6 official, mic unknown, trim 25 dB, normalize, 80-bin log-mel"}
        columns=[f"prob_seed{s}" for s in SEEDS]
        if target.exists():
            done=pd.read_csv(target,low_memory=False)
            if manifest.exists() and json.loads(manifest.read_text())==expected_manifest and set(done.file_path)==set(expected.file_path) and all(c in done for c in columns):
                frames.append(done);print("Loaded verified channel",channel,len(done));continue
            raise RuntimeError(f"Incomplete/stale external cache at {target}; inspect before rerun")
        output=[];batch=[];rows=[]
        feature_manifest=feature_cache/f"{channel}.manifest.json"
        use_features=feature_manifest.exists()
        if use_features:
            feature_spec=json.loads(feature_manifest.read_text())
            if feature_spec.get("signature")!=SIGNATURE or feature_spec.get("file_path_sha256")!=expected_manifest["file_path_sha256"]:
                raise RuntimeError(f"Stale official feature cache: {feature_manifest}")
            cached_lm=np.load(feature_cache/f"{channel}_logmel.npy",mmap_mode="r")
            cached_lengths=np.load(feature_cache/f"{channel}_lengths.npy",mmap_mode="r")
        for i,record in enumerate(expected.to_dict("records")):
            lm,length=(cached_lm[i],int(cached_lengths[i])) if use_features else _external_logmel(record["file_path"])
            row={k:record[k] for k in ("file_path","physical_speaker_id","label","gender","severity","intelligibility","block","word_code","microphone")}
            row["valid_frames"]=length;batch.append(lm);rows.append(row)
            if len(batch)>=batch_size:
                _flush(records,batch,rows);output.extend(rows);batch=[];rows=[]
        if rows:_flush(records,batch,rows);output.extend(rows)
        df=pd.DataFrame(output);df.to_csv(target,index=False)
        manifest.write_text(json.dumps(expected_manifest,indent=2))
        frames.append(df);print("Computed channel",channel,len(df))
    allpred=pd.concat(frames,ignore_index=True)
    allpred.to_csv(OUT/"official_all_channel_predictions.csv",index=False)
    return allpred


def _metric_rows(df, records, channel, level, experiment="Official UASpeech zero-shot"):
    out=[]
    for rec in records:
        prob=f"prob_seed{rec['seed']}"
        x=df[["physical_speaker_id","label",prob]].rename(columns={prob:"probability"})
        if level=="physical_speaker":x=speaker_aggregate(x)
        out.append(dict(experiment=experiment,seed=rec["seed"],channel=channel,
                        level=level,n_physical_speakers=df.physical_speaker_id.nunique(),
                        threshold=rec["threshold"],**metrics(x.label,x.probability,rec["threshold"])))
    return out


def evaluate_official(predictions, records):
    primary=predictions[predictions.microphone.eq(PRIMARY_CHANNEL)].copy()
    results=[];ci=[]
    for level in ("utterance","physical_speaker"):
        results.extend(_metric_rows(primary,records,PRIMARY_CHANNEL,level))
    for rec in records:
        col=f"prob_seed{rec['seed']}"
        frame=primary[["physical_speaker_id","label",col]].rename(columns={col:"probability"})
        for metric in ("ROC_AUC","PR_AUC","Balanced_Accuracy","F1_dysarthric"):
            point,low,high=speaker_cluster_ci(frame,metric,rec["threshold"],seed=9000+rec["seed"])
            ci.append(dict(seed=rec["seed"],metric=metric,level="physical_speaker",point=point,ci_low=low,ci_high=high,
                           n_physical_speakers=frame.physical_speaker_id.nunique()))
    df=pd.DataFrame(results);df.to_csv(OUT/"official_per_seed_metrics.csv",index=False)
    ci_df=pd.DataFrame(ci);ci_df.to_csv(OUT/"official_speaker_cluster_ci.csv",index=False)
    # CI for the five-seed mean: one shared resample of physical speakers is
    # applied to every seed, preserving the paired external test population.
    speaker_data={}
    for rec in records:
        col=f"prob_seed{rec['seed']}"
        speaker_data[rec["seed"]]=speaker_aggregate(primary[["physical_speaker_id","label",col]].rename(columns={col:"probability"})).sort_values("physical_speaker_id")
    reference=speaker_data[SEEDS[0]]
    if any(not speaker_data[s].physical_speaker_id.equals(reference.physical_speaker_id) for s in SEEDS):
        raise AssertionError("External speaker set differs across frozen seeds")
    rng=np.random.default_rng(20260928); mean_ci=[]
    for metric in ("ROC_AUC","PR_AUC","Balanced_Accuracy","F1_dysarthric"):
        point=float(np.mean([metrics(speaker_data[r["seed"]].label,speaker_data[r["seed"]].probability,r["threshold"])[metric] for r in records]))
        samples=[]
        for _ in range(2000):
            ix=rng.integers(0,len(reference),len(reference))
            if len(np.unique(reference.label.to_numpy()[ix]))<2:continue
            values=[metrics(speaker_data[r["seed"]].label.to_numpy()[ix],speaker_data[r["seed"]].probability.to_numpy()[ix],r["threshold"])[metric] for r in records]
            samples.append(float(np.mean(values)))
        mean_ci.append(dict(metric=metric,point_five_seed_mean=point,ci_low=np.percentile(samples,2.5),
                            ci_high=np.percentile(samples,97.5),n_physical_speakers=len(reference),
                            resampling_unit="physical speaker",n_bootstrap=len(samples)))
    pd.DataFrame(mean_ci).to_csv(OUT/"official_five_seed_mean_speaker_cluster_ci.csv",index=False)
    # Same speaker/content tokens across all seven channels. This prevents
    # microphone comparisons from changing the evaluated population.
    predictions=predictions.copy()
    predictions["content_key"]=predictions.physical_speaker_id.astype(str)+"|"+predictions.block.astype(str)+"|"+predictions.word_code.astype(str)
    keys=set.intersection(*[set(predictions.loc[predictions.microphone.eq(c),"content_key"]) for c in CHANNELS])
    matched=predictions[predictions.content_key.isin(keys)].copy()
    if matched.empty:raise RuntimeError("No common speaker/content subset for M2-M8")
    matched.to_csv(OUT/"official_matched_channel_predictions.csv",index=False)
    channel_rows=[]
    for c in CHANNELS:
        part=matched[matched.microphone.eq(c)]
        channel_rows.extend(_metric_rows(part,records,c,"utterance"))
        channel_rows.extend(_metric_rows(part,records,c,"physical_speaker"))
    channels=pd.DataFrame(channel_rows);channels.to_csv(OUT/"official_matched_channel_metrics.csv",index=False)
    # Descriptive only: severity labels come unchanged from official metadata.
    severity=[]
    for rec in records:
        col=f"prob_seed{rec['seed']}"
        for group,part in primary[primary.label.eq(1)].groupby("severity"):
            sp=part.groupby("physical_speaker_id")[col].mean()
            severity.append(dict(seed=rec["seed"],severity=group,n_physical_speakers=len(sp),
                                 mean_speaker_probability=sp.mean(),sd_speaker_probability=sp.std(),
                                 detection_rate=(part[col]>=rec["threshold"]).mean()))
    pd.DataFrame(severity).to_csv(OUT/"official_severity_descriptive.csv",index=False)
    audit=dict(training_on_official_uaspeech=False,fine_tuning_on_official_uaspeech=False,
               threshold_selection_on_official_uaspeech=False,calibration_on_official_uaspeech=False,
               model_selection_on_official_uaspeech=False,seeds=list(SEEDS),primary_channel=PRIMARY_CHANNEL,
               matched_channel_content_keys=len(keys),
               matched_channel_physical_speakers=matched.physical_speaker_id.nunique(),
               official_metadata_sha256=hashlib.sha256(OFFICIAL_META.read_bytes()).hexdigest(),
               official_speaker_wordlist_sha256=hashlib.sha256((ROOT/"UASpeech"/"UASpeech_normalized_FM"/"doc"/"speaker_wordlist.xls").read_bytes()).hexdigest(),
               frozen_checkpoint_sha256={str(r["seed"]):hashlib.sha256(Path(r["checkpoint"]).read_bytes()).hexdigest() for r in records})
    (OUT/"official_zero_shot_audit.json").write_text(json.dumps(audit,indent=2))
    return df,ci_df,channels


def evaluate_identical_outer_checkpoints(meta=None, batch_size=48):
    """Paired domain comparison with the same predesignated outer-fold-0 weights.

    Fold zero is fixed for every seed before reading external outcomes. Its
    held-out TORGO speakers give the internal side; the very same frozen
    checkpoint, temperature, and threshold give the official UASpeech side.
    This diagnostic is separate from the five all-TORGO deployment models.
    """
    meta=load_official_metadata() if meta is None else meta
    expected=meta[meta.microphone.eq(PRIMARY_CHANNEL)].sort_values("file_path").copy()
    tor=pd.read_csv(OUT/"torgo_nested_fold_metrics.csv")
    tor=tor[tor.level.eq("utterance") & tor.outer_fold.eq(0)].set_index("seed")
    if set(tor.index)!=set(SEEDS):
        raise RuntimeError("Paired comparison requires all five TORGO outer-fold-0 results")
    records=[]; hashes={}
    for seed in SEEDS:
        path=OUT/f"seed{seed}_outer0"/"checkpoint.pt"
        bundle=torch.load(path,map_location="cpu",weights_only=False)
        if (bundle.get("pipeline_signature")!=SIGNATURE or
                bundle.get("threshold_search_signature")!=THRESHOLD_SIGNATURE or
                bundle.get("source_corpus")!="TORGO only" or
                bundle.get("seed")!=seed or bundle.get("outer_fold")!=0):
            raise RuntimeError(f"Paired checkpoint provenance failed: {path}")
        if abs(float(bundle["threshold"])-float(tor.loc[seed,"threshold"]))>1e-12:
            raise AssertionError(f"TORGO metric and checkpoint threshold differ: {seed}")
        model=Stream1Model().to(device()).eval()
        model.load_state_dict(bundle["model_state_dict"])
        records.append(dict(seed=seed,model=model,threshold=float(bundle["threshold"]),
                            temperature=float(bundle["temperature"]),checkpoint=str(path)))
        hashes[str(seed)]=hashlib.sha256(path.read_bytes()).hexdigest()
    cache=OUT/"paired_outer_official_predictions.csv"
    manifest=OUT/"paired_outer_official_predictions.manifest.json"
    spec={"pipeline_signature":SIGNATURE,"threshold_search_signature":THRESHOLD_SIGNATURE,
          "channel":PRIMARY_CHANNEL,"outer_fold":0,"checkpoint_sha256":hashes,
          "file_path_sha256":hashlib.sha256("\n".join(expected.file_path).encode()).hexdigest()}
    if cache.exists() and manifest.exists() and json.loads(manifest.read_text())==spec:
        pred=pd.read_csv(cache,low_memory=False)
        if set(pred.file_path)!=set(expected.file_path):
            raise RuntimeError("Paired prediction cache has unexpected files")
    else:
        feature_path=OUT/"external_features"/f"{PRIMARY_CHANNEL}_logmel.npy"
        length_path=OUT/"external_features"/f"{PRIMARY_CHANNEL}_lengths.npy"
        feature_spec=json.loads((OUT/"external_features"/f"{PRIMARY_CHANNEL}.manifest.json").read_text())
        if feature_spec["file_path_sha256"]!=spec["file_path_sha256"]:
            raise RuntimeError("Paired M3 features have unexpected file order")
        features=np.load(feature_path,mmap_mode="r")
        lengths=np.load(length_path,mmap_mode="r")
        output=[];batch=[];rows=[]
        for i,record in enumerate(expected.to_dict("records")):
            row={k:record[k] for k in ("file_path","physical_speaker_id","label","microphone")}
            row["valid_frames"]=int(lengths[i]);batch.append(features[i]);rows.append(row)
            if len(batch)>=batch_size:
                _flush(records,batch,rows);output.extend(rows);batch=[];rows=[]
        if rows:_flush(records,batch,rows);output.extend(rows)
        pred=pd.DataFrame(output);pred.to_csv(cache,index=False)
        manifest.write_text(json.dumps(spec,indent=2))
    fields=["Accuracy","Balanced_Accuracy","Sensitivity","Specificity","F1_dysarthric",
            "ROC_AUC","PR_AUC","Brier","ECE"]
    pairs=[]
    for rec in records:
        seed=rec["seed"]
        observed=metrics(pred.label,pred[f"prob_seed{seed}"],rec["threshold"])
        row=dict(seed=seed,outer_fold=0,checkpoint_sha256=hashes[str(seed)],
                 TORGO_n=int(tor.loc[seed,"n"]),Official_UASpeech_n=len(pred),
                 threshold=rec["threshold"],temperature=rec["temperature"])
        row.update({f"TORGO_{m}":float(tor.loc[seed,m]) for m in fields})
        row.update({f"Official_UASpeech_{m}":observed[m] for m in fields})
        pairs.append(row)
    paired=pd.DataFrame(pairs)
    paired.to_csv(OUT/"paired_identical_checkpoint_metrics.csv",index=False)
    return paired,pred
