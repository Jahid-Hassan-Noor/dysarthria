"""V7 speaker-safe TORGO protocol. V6 artifacts are read-only inputs.

The official UASpeech corpus is absent from this module by design. Training,
epoch selection, calibration and threshold selection accept TORGO rows only.
"""
from __future__ import annotations

import hashlib
import json
import pickle
import random
import re
import shutil
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize_scalar
from sklearn.metrics import (accuracy_score, average_precision_score,
                             balanced_accuracy_score, brier_score_loss,
                             confusion_matrix, f1_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "v7" / "outputs"
OUT.mkdir(parents=True, exist_ok=True)
OFFICIAL_TORGO_ROSTER = {
    "F01", "F03", "F04", "M01", "M02", "M03", "M04", "M05",
    "FC01", "FC02", "FC03", "MC01", "MC02", "MC03", "MC04",
}
TORGO_DOCUMENTATION = "https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html"
SEEDS = (42, 43, 44, 45, 46)
SNR_PROBS = ((None, .40), (30, .15), (25, .15), (20, .15), (15, .10), (10, .05))
SIGNATURE = "v7_physical_speaker_nested_stream1_v2_effective_gain"
THRESHOLD_SIGNATURE = "exact_speaker_weighted_unique_score_search_v1"


def _cache_paths():
    runs = list((ROOT / "v6" / "outputs" / "cache" / "dysarthria_detection_v6").glob("run_*"))
    valid = [p for p in runs if (p / "dual_stream_features.npz").exists()
             and (p / "stages" / "metadata.pkl").exists()]
    if len(valid) != 1:
        raise RuntimeError(f"Expected one complete V6 feature cache; found {valid}")
    return valid[0]


def load_torgo():
    """Load TORGO features and verify every flattened session against paths."""
    run = _cache_paths()
    with (run / "stages" / "metadata.pkl").open("rb") as f:
        raw = pickle.load(f)["metadata"].copy()
    z = np.load(run / "dual_stream_features.npz", allow_pickle=True)
    meta = pd.DataFrame(z["meta_values"], columns=z["meta_columns"].tolist())
    meta["label"] = meta["label"].astype(int)
    meta["valid_frames"] = meta["valid_frames"].astype(int)
    if len(meta) != len(z["logmels"]):
        raise AssertionError("TORGO feature/metadata row mismatch")
    if not set(meta.file_path).issubset(set(raw.file_path)):
        raise AssertionError("Feature cache contains paths absent from TORGO inventory")
    return meta, z["logmels"], run


def audit_physical_speakers(meta, save=True):
    """Only accepts session IDs present in the documented 15-speaker roster.

    The original TORGO hierarchy is speaker/Session; this Kaggle mirror has
    flattened names wav_{headMic,arrayMic}_{speaker}S{session}. Both the parent
    folder and WAV basename must agree with the cached session ID.
    """
    rows = []
    for sid, sub in meta.groupby("speaker_id", sort=True):
        sid = str(sid)
        match = re.fullmatch(r"((?:FC|MC|F|M)\d{2})(?:S(\d{2}))?", sid)
        if not match or match.group(1) not in OFFICIAL_TORGO_ROSTER:
            raise AssertionError(f"Unverified TORGO session ID: {sid}")
        physical, session = match.group(1), match.group(2)
        labels = set(sub.label.astype(int)); genders = set(sub.gender.astype(str))
        expected_label = 0 if "C" in physical else 1
        expected_gender = "female" if physical.startswith("F") else "male"
        if labels != {expected_label} or genders != {expected_gender}:
            raise AssertionError(f"Label/gender conflict for {sid}: {labels}/{genders}")
        folders, mics = set(), set()
        for path in sub.file_path.astype(str):
            p = Path(path)
            folder = p.parent.name
            m = re.fullmatch(r"wav_(headMic|arrayMic)_(.+)", folder)
            if not m or m.group(2) != sid or not p.name.startswith(folder + "_"):
                raise AssertionError(f"TORGO path/session disagreement: {path}")
            if p.parent.parent.name != ("F" if expected_gender == "female" else "M") + ("_Con" if expected_label == 0 else "_Dys"):
                raise AssertionError(f"TORGO path/group disagreement: {path}")
            if not p.is_file():
                raise FileNotFoundError(path)
            folders.add(folder); mics.add(m.group(1))
        rows.append(dict(session_id=sid, physical_speaker_id=physical,
                         session_number=session or "unsuffixed", label=expected_label,
                         gender=expected_gender, n_utterances=len(sub),
                         microphones=",".join(sorted(mics)),
                         observed_folders=";".join(sorted(folders)),
                         source_document=TORGO_DOCUMENTATION))
    audit = pd.DataFrame(rows)
    observed = set(audit.physical_speaker_id)
    if observed != OFFICIAL_TORGO_ROSTER:
        raise AssertionError(f"Roster mismatch: missing={OFFICIAL_TORGO_ROSTER-observed}, extra={observed-OFFICIAL_TORGO_ROSTER}")
    meta = meta.copy()
    meta["physical_speaker_id"] = meta.speaker_id.map(audit.set_index("session_id").physical_speaker_id)
    if meta.physical_speaker_id.isna().any():
        raise AssertionError("Unmapped TORGO row")
    if save:
        audit.to_csv(OUT / "torgo_physical_speaker_mapping_audit.csv", index=False)
        meta[["file_path", "speaker_id", "physical_speaker_id", "label", "gender", "mic_type"]].to_csv(
            OUT / "torgo_verified_manifest.csv", index=False)
    return meta, audit


def historical_v6_leakage_audit():
    """Read V6's saved session split without modifying its historical cache."""
    run=_cache_paths()
    with (run/"stages"/"speaker_split.pkl").open("rb") as f:
        split_map=pickle.load(f)["split_map"]
    rows=[]
    for session,part in sorted(split_map.items()):
        match=re.fullmatch(r"((?:FC|MC|F|M)\d{2})(?:S\d{2})?",session)
        if not match or match.group(1) not in OFFICIAL_TORGO_ROSTER:
            raise AssertionError(f"Unverified historical session: {session}")
        rows.append(dict(session_id=session,physical_speaker_id=match.group(1),v6_split=part))
    df=pd.DataFrame(rows)
    df["physical_overlap_across_v6_splits"]=df.groupby("physical_speaker_id").v6_split.transform("nunique")>1
    df.to_csv(OUT/"v6_historical_physical_leakage_audit.csv",index=False)
    return df


def _speaker_table(meta):
    sp = meta.groupby("physical_speaker_id", as_index=False).agg(label=("label", "first"), gender=("gender", "first"))
    if not (meta.groupby("physical_speaker_id").label.nunique() == 1).all():
        raise AssertionError("Conflicting labels within physical speaker")
    return sp


def grouped_splits(meta, seed, n_splits=5):
    sp = _speaker_table(meta)
    class_counts = sp.label.value_counts()
    n_splits = min(n_splits, int(class_counts.min()))
    if n_splits < 2:
        raise RuntimeError("Too few physical speakers for grouped evaluation")
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr, te) in enumerate(sgkf.split(sp, sp.label, sp.physical_speaker_id)):
        tr_sp, te_sp = set(sp.iloc[tr].physical_speaker_id), set(sp.iloc[te].physical_speaker_id)
        if not tr_sp.isdisjoint(te_sp):
            raise AssertionError("Train/test physical speaker overlap")
        train = np.flatnonzero(meta.physical_speaker_id.isin(tr_sp))
        test = np.flatnonzero(meta.physical_speaker_id.isin(te_sp))
        yield fold, train, test


def subgroup_counts(meta, indices, seed, outer_fold, partition):
    sub = _speaker_table(meta.iloc[indices])
    counts = {f"{gender}_{diagnosis}": int(((sub.gender == gender) & (sub.label == label)).sum())
              for gender in ("female", "male") for diagnosis, label in (("control", 0), ("dysarthric", 1))}
    return dict(seed=seed, outer_fold=outer_fold, partition=partition, **counts)


def assert_disjoint(meta, train, val, test):
    train_physical_speakers = set(meta.iloc[train].physical_speaker_id)
    val_physical_speakers = set(meta.iloc[val].physical_speaker_id)
    test_physical_speakers = set(meta.iloc[test].physical_speaker_id)
    assert train_physical_speakers.isdisjoint(val_physical_speakers)
    assert train_physical_speakers.isdisjoint(test_physical_speakers)
    assert val_physical_speakers.isdisjoint(test_physical_speakers)


def split_audit(meta):
    rows = []
    for seed in SEEDS:
        for outer, otr, ote in grouped_splits(meta, seed):
            inner_meta = meta.iloc[otr].reset_index(drop=True)
            for inner, itr, iva in grouped_splits(inner_meta, seed + 731, n_splits=3):
                tr, va = otr[itr], otr[iva]
                assert_disjoint(meta, tr, va, ote)
                rows += [subgroup_counts(meta, ix, seed, outer, f"inner{inner}_{part}")
                         for part, ix in (("train", tr), ("validation", va))]
            rows.append(subgroup_counts(meta, ote, seed, outer, "outer_test"))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "torgo_grouped_split_audit.csv", index=False)
    return df


def write_split_manifests(meta):
    """Persist exact physical-speaker membership for every outer/inner fold."""
    for seed in SEEDS:
        for outer,tr,te in grouped_splits(meta,seed):
            base=OUT/f"seed{seed}_outer{outer}";base.mkdir(exist_ok=True)
            inner_meta=meta.iloc[tr].reset_index(drop=True)
            inner_rows=[]
            for inner,tr_local,va_local in grouped_splits(inner_meta,seed+731,n_splits=3):
                tr_i,va_i=tr[tr_local],tr[va_local]
                assert_disjoint(meta,tr_i,va_i,te)
                inner_rows.append(dict(inner_fold=inner,
                    train_physical_speakers=sorted(set(meta.iloc[tr_i].physical_speaker_id)),
                    validation_physical_speakers=sorted(set(meta.iloc[va_i].physical_speaker_id))))
            record=dict(seed=seed,outer_fold=outer,pipeline_signature=SIGNATURE,
                        train_physical_speakers=sorted(set(meta.iloc[tr].physical_speaker_id)),
                        test_physical_speakers=sorted(set(meta.iloc[te].physical_speaker_id)),
                        inner_folds=inner_rows)
            (base/"split_manifest.json").write_text(json.dumps(record,indent=2))
        deployment_base=OUT/f"deployment_seed{seed}";deployment_base.mkdir(exist_ok=True)
        deployment_inner=[]
        for inner,tr,va in grouped_splits(meta,seed+731,n_splits=5):
            if not set(meta.iloc[tr].physical_speaker_id).isdisjoint(set(meta.iloc[va].physical_speaker_id)):
                raise AssertionError("Deployment inner physical speaker overlap")
            deployment_inner.append(dict(inner_fold=inner,
                train_physical_speakers=sorted(set(meta.iloc[tr].physical_speaker_id)),
                validation_physical_speakers=sorted(set(meta.iloc[va].physical_speaker_id))))
        (deployment_base/"split_manifest.json").write_text(json.dumps(dict(
            seed=seed,outer_fold="deployment",pipeline_signature=SIGNATURE,
            final_train_physical_speakers=sorted(set(meta.physical_speaker_id)),
            inner_folds=deployment_inner),indent=2))


def metrics(y, p, threshold):
    y, p = np.asarray(y, int), np.asarray(p, float)
    pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    div = lambda a, b: float(a / b) if b else float("nan")
    two = len(np.unique(y)) == 2
    return dict(Accuracy=accuracy_score(y, pred), Balanced_Accuracy=div(div(tp, tp+fn)+div(tn, tn+fp), 2) if two else float("nan"),
                Sensitivity=div(tp, tp+fn), Specificity=div(tn, tn+fp),
                Precision=div(tp, tp+fp), NPV=div(tn, tn+fn),
                F1_dysarthric=f1_score(y, pred, zero_division=0),
                Macro_F1=f1_score(y, pred, average="macro", labels=[0, 1], zero_division=0),
                ROC_AUC=roc_auc_score(y, p) if two else float("nan"),
                PR_AUC=average_precision_score(y, p) if two else float("nan"),
                Brier=brier_score_loss(y, p), ECE=ece(y, p), n=len(y))


def ece(y, p, bins=10):
    y, p = np.asarray(y), np.asarray(p)
    ids = np.minimum((p * bins).astype(int), bins-1)
    return float(sum((ids == b).mean() * abs(y[ids == b].mean() - p[ids == b].mean())
                     for b in range(bins) if (ids == b).any()))


def threshold_table(y, p, speaker_ids):
    """Equal physical-speaker weight; all candidate policies are predeclared."""
    y, p, sp = np.asarray(y, int), np.asarray(p, float), np.asarray(speaker_ids)
    counts = pd.Series(sp).value_counts()
    weights = np.array([1 / counts[s] for s in sp]); weights /= weights.sum()
    unique, inverse=np.unique(p,return_inverse=True)
    pos=np.bincount(inverse,weights=weights*(y==1),minlength=len(unique))
    neg=np.bincount(inverse,weights=weights*(y==0),minlength=len(unique))
    tp=np.cumsum(pos[::-1])[::-1];fp=np.cumsum(neg[::-1])[::-1]
    total_pos=pos.sum();total_neg=neg.sum()
    if total_pos<=0 or total_neg<=0:
        raise RuntimeError("TORGO threshold selection needs both diagnoses")
    sens=tp/total_pos;spec=(total_neg-fp)/total_neg
    precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
    f1=np.divide(2*precision*sens,precision+sens,out=np.zeros_like(sens),where=(precision+sens)>0)
    table=pd.DataFrame(dict(threshold=unique,F1=f1,Sensitivity=sens,Specificity=spec,
        Youden_J=sens+spec-1,Balanced_Accuracy=(sens+spec)/2))
    # Include the all-negative operating point. The lowest unique score already
    # represents all-positive. Every possible score boundary is considered.
    table.loc[len(table)]=dict(threshold=np.nextafter(float(unique[-1]),np.inf),
        F1=0.0,Sensitivity=0.0,Specificity=1.0,Youden_J=0.0,Balanced_Accuracy=.5)
    policies = {"F1_optimal": "F1", "Youden_J": "Youden_J", "Balanced_Accuracy_optimal": "Balanced_Accuracy"}
    choices = {}
    for name, col in policies.items():
        choices[name] = table.sort_values([col, "Specificity", "threshold"], ascending=[False, False, False]).iloc[0].to_dict()
    feasible = table[table.Sensitivity >= .90 - 1e-12]
    if feasible.empty:
        raise RuntimeError("No validation threshold meets sensitivity >= 0.90")
    choices["Specificity_at_sensitivity_0.90"] = feasible.sort_values(["Specificity", "threshold"], ascending=[False, False]).iloc[0].to_dict()
    return pd.DataFrame([dict(policy=k, **v) for k, v in choices.items()])


def calibrate_temperature(y, p, speaker_ids):
    y, p, sp = np.asarray(y, float), np.asarray(p, float), np.asarray(speaker_ids)
    counts = pd.Series(sp).value_counts(); w = np.array([1/counts[s] for s in sp]); w /= w.sum()
    logits = np.log(np.clip(p, 1e-7, 1-1e-7) / np.clip(1-p, 1e-7, 1-1e-7))
    def objective(log_t):
        x = logits / np.exp(log_t)
        return float(np.sum(w * (np.logaddexp(0, x) - y*x)))
    result = minimize_scalar(objective, bounds=(np.log(.05), np.log(20)), method="bounded")
    if not result.success:
        raise RuntimeError("TORGO validation temperature fit failed")
    return float(np.exp(result.x))


def apply_temperature(p, temperature):
    p = np.clip(np.asarray(p, float), 1e-7, 1-1e-7)
    logit = np.log(p/(1-p))/temperature
    return 1/(1+np.exp(-logit))


def speaker_aggregate(frame, probability="probability"):
    return frame.groupby("physical_speaker_id", as_index=False).agg(
        label=("label", "first"), probability=(probability, "mean"))


def speaker_cluster_ci(frame, metric="ROC_AUC", threshold=.5, reps=2000, seed=42):
    sp = speaker_aggregate(frame)
    y, p = sp.label.to_numpy(), sp.probability.to_numpy()
    point = metrics(y, p, threshold)[metric]
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(reps):
        ix = rng.integers(0, len(sp), len(sp))
        if len(np.unique(y[ix])) < 2: continue
        vals.append(metrics(y[ix], p[ix], threshold)[metric])
    if not vals: return point, float("nan"), float("nan")
    return point, float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


class SpectroTemporalEncoder(nn.Module):
    """Architecture copied unchanged from V6 Stream-1 for comparability."""
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(nn.Conv2d(1,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2,2),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1,None)))
        self.bilstm = nn.LSTM(128,128,num_layers=2,batch_first=True,bidirectional=True,dropout=.3)
        self.proj = nn.Linear(256,256); self.norm = nn.LayerNorm(256)
    def forward(self, x, lengths):
        feat = self.conv_block(x).squeeze(2).transpose(1,2)
        out,_ = self.bilstm(feat)
        reduced = torch.clamp((lengths.float()/4).ceil().long(), min=1,max=out.size(1))
        mask = (torch.arange(out.size(1),device=out.device)[None,:] < reduced[:,None]).unsqueeze(-1).float()
        return self.norm(self.proj((out*mask).sum(1)/mask.sum(1).clamp_min(1e-6)))


class Stream1Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpectroTemporalEncoder()
        self.dysarthria_head = nn.Sequential(nn.Linear(256,128),nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,2))
        self.gender_head = nn.Sequential(nn.Linear(256,64),nn.GELU(),nn.Dropout(.2),nn.Linear(64,2))
    def forward(self, logmel, lengths):
        emb = self.encoder(logmel, lengths)
        return self.dysarthria_head(emb)


def device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")


@lru_cache(maxsize=256)
def _base_audio(path, mic):
    audio, _ = librosa.load(path, sr=16000)
    audio, _ = librosa.effects.trim(audio, top_db=35 if mic == "head" else 25)
    if len(audio) == 0: audio = np.zeros(800, dtype=np.float32)
    audio = librosa.util.normalize(audio)
    if mic == "array":
        audio = np.where(np.abs(audio) < .01*np.max(np.abs(audio)), 0, audio)
    return audio.astype(np.float32)


def _logmel(audio):
    mel = librosa.feature.melspectrogram(y=audio,sr=16000,n_mels=80,hop_length=160,win_length=400,fmax=8000)
    lm = librosa.power_to_db(mel,ref=np.max)
    lm = (lm-lm.mean())/(lm.std()+1e-8)
    length = min(128,max(1,1+len(audio)//160))
    lm = np.pad(lm[:,:128],((0,0),(0,max(0,128-lm.shape[1]))))
    return lm.astype(np.float32), length


def augment_torgo_audio(audio, rng):
    """Training only; the SNR schedule is fixed without using UASpeech."""
    x = audio.copy(); x *= rng.uniform(.8,1.2)
    shift = int(rng.integers(-320,321)); x = np.roll(x,shift)
    if shift > 0: x[:shift] = 0
    elif shift < 0: x[shift:] = 0
    # Smooth random frequency response: low/high shelf applied in Fourier domain.
    if rng.random() < .5:
        f = np.fft.rfftfreq(len(x),1/16000)
        tilt = rng.uniform(-.15,.15)
        response = np.maximum(.5,1+tilt*np.log2(np.maximum(f,100)/1000))
        x = np.fft.irfft(np.fft.rfft(x)*response,n=len(x)).astype(np.float32)
    snr = rng.choice([v for v,_ in SNR_PROBS],p=[v for _,v in SNR_PROBS])
    if snr is not None:
        power = np.mean(x*x)+1e-12
        x += rng.normal(0,np.sqrt(power/(10**(float(snr)/10))),size=len(x)).astype(np.float32)
    return x.astype(np.float32)


def _train_batch(meta, indices, rng):
    lm, lengths, y = [], [], []
    for i in indices:
        row = meta.iloc[int(i)]
        audio = augment_torgo_audio(_base_audio(str(row.file_path),str(row.mic_type)),rng)
        a,n = _logmel(audio)
        # A pure waveform gain is cancelled by V6's per-utterance z-score.
        # A mild post-normalization feature gain makes this perturbation real.
        a *= rng.uniform(.90,1.10)
        # SpecAugment, training only.
        if rng.random() < .5:
            t0 = int(rng.integers(0,128)); tw = int(rng.integers(0,21)); a[:,t0:min(128,t0+tw)] = a.mean()
        if rng.random() < .5:
            f0 = int(rng.integers(0,80)); fw = int(rng.integers(0,16)); a[f0:min(80,f0+fw),:] = a.mean()
        lm.append(a); lengths.append(n); y.append(int(row.label))
    return np.stack(lm), np.array(lengths), np.array(y)


def _speaker_balanced_indices(meta, eligible, rng, n):
    sub = meta.iloc[eligible]
    by_class = {lab: {} for lab in (0,1)}
    for speaker, group in sub.groupby("physical_speaker_id"):
        by_class[int(group.label.iloc[0])][speaker] = group.index.to_numpy()
    if not all(by_class.values()): raise RuntimeError("Training fold needs both classes")
    sampled = []
    for _ in range(n):
        lab = int(rng.integers(0,2))
        speaker = rng.choice(list(by_class[lab]))
        sampled.append(int(rng.choice(by_class[lab][speaker])))
    return np.asarray(sampled,int)


def predict(model, meta, features, indices, batch_size=64, dev=None):
    dev = dev or device(); model.eval(); out=[]
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            ix = indices[start:start+batch_size]
            x = torch.as_tensor(features[ix],device=dev).unsqueeze(1)
            lengths = torch.as_tensor(meta.iloc[ix].valid_frames.to_numpy(dtype=int).copy(),device=dev)
            out.append(torch.softmax(model(x,lengths),1)[:,1].cpu().numpy())
    return np.concatenate(out)


def train_one(meta, features, train_indices, seed, name, epochs=20, val_indices=None,
              patience=4, samples_per_epoch=512, batch_size=32, history_path=None):
    """Validation speakers select only the epoch; final fits use fixed epochs."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    dev=device(); model=Stream1Model().to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed); best_loss=float("inf"); best_epoch=epochs; best_state=None; stale=0
    history=[]
    for epoch in range(1,epochs+1):
        model.train(); order=_speaker_balanced_indices(meta,train_indices,rng,samples_per_epoch)
        train_losses=[]
        for start in range(0,len(order),batch_size):
            lm,ln,y=_train_batch(meta,order[start:start+batch_size],rng)
            logits=model(torch.as_tensor(lm,device=dev).unsqueeze(1),torch.as_tensor(ln,device=dev))
            loss=F.cross_entropy(logits,torch.as_tensor(y,device=dev))
            train_losses.append(float(loss.detach().cpu()))
            opt.zero_grad();loss.backward();opt.step()
        if val_indices is not None:
            p=predict(model,meta,features,val_indices,dev=dev)
            y=meta.iloc[val_indices].label.to_numpy(dtype=int)
            speakers=meta.iloc[val_indices].physical_speaker_id.to_numpy()
            counts=pd.Series(speakers).value_counts();w=np.array([1/counts[s] for s in speakers]);w/=w.sum()
            loss=float(-np.sum(w*(y*np.log(np.clip(p,1e-7,1))+(1-y)*np.log(np.clip(1-p,1e-7,1)))))
            val_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else float("nan")
            val_f1=f1_score(y,p>=.5,zero_division=0)
            if loss < best_loss-1e-5:
                best_loss=loss;best_epoch=epoch;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};stale=0
            else:
                stale+=1
            if history_path is not None:
                history.append(dict(epoch=epoch,train_loss=float(np.mean(train_losses)),
                    val_loss=loss,val_roc_auc=val_auc,val_f1_at_0_5=val_f1,
                    best_epoch_so_far=best_epoch,stale_epochs=stale,
                    max_epochs=epochs,patience=patience))
                pd.DataFrame(history).to_csv(history_path,index=False)
            if stale>=patience:break
    if best_state is not None:model.load_state_dict(best_state)
    return model,best_epoch


def _save_bundle(model, seed, fold, train_speakers, inner_speakers, epoch, temp, threshold, path):
    bundle=dict(model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
                seed=seed,outer_fold=fold,train_physical_speakers=sorted(train_speakers),
                inner_validation_physical_speakers=sorted(inner_speakers),selected_epochs=epoch,
                temperature=temp,threshold=threshold,pipeline_signature=SIGNATURE,
                threshold_search_signature=THRESHOLD_SIGNATURE,
                threshold_policy="max specificity subject to sensitivity >= 0.90",
                source_corpus="TORGO only",
                training_recipe={"architecture":"V6 Stream-1 CNN-BiLSTM",
                    "optimizer":"AdamW", "learning_rate":5e-4,"weight_decay":1e-4,
                    "sampling":"diagnosis -> physical speaker -> utterance",
                    "snr_distribution":list(SNR_PROBS),
                    "augmentation":"waveform gain/noise/time shift/mic response; post-zscore feature gain; SpecAugment",
                    "calibration":"speaker-weighted TORGO inner OOF temperature scaling"})
    torch.save(bundle,path)
    return bundle


def _nested_fit(meta, features, development_indices, seed, fold_name, outer_test=None,
                epochs=20, samples_per_epoch=512):
    dev_meta=meta.iloc[development_indices].reset_index(drop=True)
    oof=[]; chosen_epochs=[]; validation_speakers=[]
    for inner,tr_local,va_local in grouped_splits(dev_meta,seed+731,n_splits=3 if outer_test is not None else 5):
        tr=development_indices[tr_local];va=development_indices[va_local]
        if outer_test is not None:assert_disjoint(meta,tr,va,outer_test)
        model,best_epoch=train_one(meta,features,tr,seed+inner,f"{fold_name}_inner{inner}",epochs,va,
                                   samples_per_epoch=samples_per_epoch)
        p=predict(model,meta,features,va)
        oof.append(pd.DataFrame(dict(index=va,physical_speaker_id=meta.iloc[va].physical_speaker_id.to_numpy(),
                                     label=meta.iloc[va].label.to_numpy(dtype=int),probability_raw=p)))
        chosen_epochs.append(best_epoch)
        validation_speakers.extend(sorted(set(meta.iloc[va].physical_speaker_id)))
        del model
    oof=pd.concat(oof,ignore_index=True).sort_values("index")
    if len(oof)!=len(development_indices) or oof["index"].duplicated().any():
        raise AssertionError("Inner OOF does not cover each TORGO development row exactly once")
    temp=calibrate_temperature(oof.label,oof.probability_raw,oof.physical_speaker_id)
    oof["probability"]=apply_temperature(oof.probability_raw,temp)
    policies=threshold_table(oof.label,oof.probability,oof.physical_speaker_id)
    threshold=float(policies.set_index("policy").loc["Specificity_at_sensitivity_0.90","threshold"])
    final_epochs=int(np.median(chosen_epochs))
    model,_=train_one(meta,features,development_indices,seed+10000,fold_name,epochs=final_epochs,
                      samples_per_epoch=samples_per_epoch)
    return model,oof,policies,temp,threshold,final_epochs,validation_speakers


def run_nested_torgo(meta,features,seeds=SEEDS,epochs=20,samples_per_epoch=512):
    """Five outer grouped folds per seed; outer speakers never tune the model."""
    rows=[];predictions=[];subgroups=[];subgroup_metrics=[]
    for seed in seeds:
        for fold,tr,te in grouped_splits(meta,seed):
            name=f"seed{seed}_outer{fold}"; base=OUT/name;base.mkdir(exist_ok=True)
            done=base/"evaluation.csv"
            if done.exists():
                bundle=torch.load(base/"checkpoint.pt",map_location="cpu",weights_only=False)
                if bundle.get("pipeline_signature")!=SIGNATURE:
                    raise RuntimeError(f"Stale grouped checkpoint: {base}")
                rows.append(pd.read_csv(done))
                pred=pd.read_csv(base/"predictions.csv");predictions.append(pred)
                subgroups.append(subgroup_counts(meta,te,seed,fold,"outer_test"))
                for gender in ("female","male"):
                    for label,diagnosis_name in ((0,"control"),(1,"dysarthric")):
                        part=pred[pred.gender.eq(gender)&pred.label.eq(label)]
                        if len(part):
                            subgroup_metrics.append(dict(seed=seed,outer_fold=fold,
                                subgroup=f"{gender}_{diagnosis_name}",n_physical_speakers=part.physical_speaker_id.nunique(),
                                ROC_AUC=float("nan"),PR_AUC=float("nan"),estimable=False,
                                note="One diagnosis class; discrimination metrics non-estimable"))
                continue
            model,oof,policies,temp,threshold,ep,val_sp=_nested_fit(meta,features,tr,seed,name,te,epochs,samples_per_epoch)
            raw=predict(model,meta,features,te);p=apply_temperature(raw,temp)
            pred=meta.iloc[te][["file_path","physical_speaker_id","label","gender","mic_type"]].copy()
            pred["probability_raw"]=raw;pred["probability"]=p;pred["seed"]=seed;pred["outer_fold"]=fold
            pred.to_csv(base/"predictions.csv",index=False);oof.to_csv(base/"inner_oof.csv",index=False)
            policies.to_csv(base/"threshold_policies.csv",index=False)
            _save_bundle(model,seed,fold,set(meta.iloc[tr].physical_speaker_id),val_sp,ep,temp,threshold,base/"checkpoint.pt")
            for level,df in (("utterance",pred),("physical_speaker",speaker_aggregate(pred))):
                result=dict(seed=seed,outer_fold=fold,level=level,threshold=threshold,temperature=temp,
                            **metrics(df.label,df.probability,threshold))
                rows.append(pd.DataFrame([result]))
            pd.concat(rows[-2:]).to_csv(done,index=False)
            predictions.append(pred)
            subgroups.append(subgroup_counts(meta,te,seed,fold,"outer_test"))
            for gender in ("female","male"):
                for label,diagnosis_name in ((0,"control"),(1,"dysarthric")):
                    part=pred[(pred.gender.eq(gender)) & (pred.label.eq(label))]
                    if len(part):
                        r=metrics(part.label,part.probability,threshold)
                        subgroup_metrics.append(dict(seed=seed,outer_fold=fold,subgroup=f"{gender}_{diagnosis_name}",
                            n_physical_speakers=part.physical_speaker_id.nunique(),ROC_AUC=r["ROC_AUC"],
                            PR_AUC=r["PR_AUC"],estimable=False,
                            note="One diagnosis class; discrimination metrics non-estimable"))
            print("Completed",name)
    summary=pd.concat(rows,ignore_index=True);summary.to_csv(OUT/"torgo_nested_fold_metrics.csv",index=False)
    pd.concat(predictions,ignore_index=True).to_csv(OUT/"torgo_nested_predictions.csv",index=False)
    pd.DataFrame(subgroups).to_csv(OUT/"torgo_outer_subgroup_counts.csv",index=False)
    pd.DataFrame(subgroup_metrics).to_csv(OUT/"torgo_outer_subgroup_metrics.csv",index=False)
    return summary


def train_deployment(meta,features,seeds=SEEDS,epochs=20,samples_per_epoch=512):
    """All TORGO speakers train final frozen models; grouped OOF locks settings."""
    all_idx=np.arange(len(meta)); rows=[]
    for seed in seeds:
        base=OUT/f"deployment_seed{seed}";base.mkdir(exist_ok=True)
        if (base/"checkpoint.pt").exists():
            bundle=torch.load(base/"checkpoint.pt",map_location="cpu",weights_only=False)
            if bundle.get("pipeline_signature")!=SIGNATURE:raise RuntimeError("Deployment signature mismatch")
            rows.append(dict(seed=seed,threshold=bundle["threshold"],temperature=bundle["temperature"],checkpoint=str(base/"checkpoint.pt")))
            continue
        model,oof,policies,temp,threshold,ep,val_sp=_nested_fit(meta,features,all_idx,seed,"deployment",None,epochs,samples_per_epoch)
        _save_bundle(model,seed,"deployment",set(meta.physical_speaker_id),val_sp,ep,temp,threshold,base/"checkpoint.pt")
        oof.to_csv(base/"torgo_grouped_oof.csv",index=False);policies.to_csv(base/"threshold_policies.csv",index=False)
        rows.append(dict(seed=seed,threshold=threshold,temperature=temp,checkpoint=str(base/"checkpoint.pt")))
        print("Frozen TORGO deployment model",seed)
    df=pd.DataFrame(rows);df.to_csv(OUT/"deployment_manifest.csv",index=False)
    return df


def run_cross_microphone(meta,features,seed=42,epochs=20,samples_per_epoch=512):
    """Physical-speaker outer folds; train and inner validation share one mic."""
    rows=[]
    for fold,tr,te in grouped_splits(meta,seed):
        for train_mic in ("head","array"):
            train=tr[meta.iloc[tr].mic_type.to_numpy()==train_mic]
            name=f"crossmic_seed{seed}_fold{fold}_{train_mic}"
            base=OUT/name;base.mkdir(exist_ok=True)
            if (base/"metrics.csv").exists():
                rows.append(pd.read_csv(base/"metrics.csv"));continue
            model,oof,policies,temp,thr,ep,val_sp=_nested_fit(meta,features,train,seed,name,te,epochs,samples_per_epoch)
            policies.to_csv(base/"threshold_policies.csv",index=False)
            oof.to_csv(base/"same_mic_inner_oof.csv",index=False)
            _save_bundle(model,seed,fold,set(meta.iloc[train].physical_speaker_id),val_sp,ep,temp,thr,base/"checkpoint.pt")
            r=[]
            for test_mic in ("head","array"):
                test=te[meta.iloc[te].mic_type.to_numpy()==test_mic]
                p=apply_temperature(predict(model,meta,features,test),temp)
                r.append(dict(seed=seed,outer_fold=fold,train_mic=train_mic,test_mic=test_mic,
                              n_physical_speakers=meta.iloc[test].physical_speaker_id.nunique(),
                              threshold=thr,**metrics(meta.iloc[test].label,p,thr)))
            pd.DataFrame(r).to_csv(base/"metrics.csv",index=False);rows.append(pd.DataFrame(r))
    df=pd.concat(rows,ignore_index=True);df.to_csv(OUT/"torgo_cross_microphone_results.csv",index=False)
    return df


def run_noise_robustness(meta,seed=42,snr_levels=(30,20,10,5,0),batch_size=48):
    """Held-out outer speakers only; locked TORGO inner-validation settings."""
    target=OUT/"torgo_noise_results.csv"
    if target.exists():
        cached=pd.read_csv(target)
        expected={(seed,fold,int(snr)) for fold,_,_ in grouped_splits(meta,seed) for snr in snr_levels}
        observed={(int(r.seed),int(r.outer_fold),int(r.SNR_dB)) for r in cached.itertuples()}
        if observed==expected and "threshold_search_signature" in cached and (cached.threshold_search_signature==THRESHOLD_SIGNATURE).all():
            return cached
    rows=[];dev=device()
    for fold,_,te in grouped_splits(meta,seed):
        path=OUT/f"seed{seed}_outer{fold}"/"checkpoint.pt"
        if not path.exists():raise RuntimeError(f"Train grouped outer fold first: {path}")
        bundle=torch.load(path,map_location="cpu",weights_only=False)
        if bundle.get("threshold_search_signature")!=THRESHOLD_SIGNATURE:
            raise RuntimeError(f"Exact TORGO threshold lock required before noise evaluation: {path}")
        model=Stream1Model().to(dev).eval();model.load_state_dict(bundle["model_state_dict"])
        for snr in snr_levels:
            y=[];sp=[];p=[];rng=np.random.default_rng(seed*1000+fold*10+snr)
            lm=[];ln=[]
            def flush():
                if not lm:return
                with torch.no_grad():
                    pred=torch.softmax(model(torch.as_tensor(np.stack(lm),device=dev).unsqueeze(1),
                        torch.as_tensor(np.asarray(ln),device=dev)),1)[:,1].cpu().numpy()
                p.extend(apply_temperature(pred,bundle["temperature"]).tolist());lm.clear();ln.clear()
            for i in te:
                row=meta.iloc[i];audio=_base_audio(str(row.file_path),str(row.mic_type)).copy()
                power=np.mean(audio**2)+1e-12
                audio+=rng.normal(0,np.sqrt(power/(10**(snr/10))),size=len(audio)).astype(np.float32)
                a,length=_logmel(audio);lm.append(a);ln.append(length)
                y.append(int(row.label));sp.append(row.physical_speaker_id)
                if len(lm)>=batch_size:flush()
            flush()
            rows.append(dict(seed=seed,outer_fold=fold,SNR_dB=snr,n_physical_speakers=len(set(sp)),
                             threshold_search_signature=THRESHOLD_SIGNATURE,
                             **metrics(y,p,bundle["threshold"])))
            print("Noise",seed,fold,snr)
    df=pd.DataFrame(rows);df.to_csv(target,index=False)
    return df


def summarize_grouped_validation_calibration():
    """Raw vs TORGO-validation-only temperature scaling fit diagnostics.

    These inner OOF scores fit temperature; use the separate outer-test table
    for an unbiased estimate of calibration transfer to unseen speakers.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    rows=[];frames=[]
    for seed in SEEDS:
        for fold in range(5):
            p=OUT/f"seed{seed}_outer{fold}"/"inner_oof.csv"
            if not p.exists():raise RuntimeError(f"Missing grouped inner OOF: {p}")
            d=pd.read_csv(p);d["seed"]=seed;d["outer_fold"]=fold;frames.append(d)
            for variant,col in (("uncalibrated","probability_raw"),("temperature_scaled","probability")):
                result=metrics(d.label,d[col],.5)
                rows.append(dict(seed=seed,outer_fold=fold,variant=variant,
                                 n_physical_speakers=d.physical_speaker_id.nunique(),
                                 Brier=result["Brier"],ECE=result["ECE"],
                                 ROC_AUC=result["ROC_AUC"],PR_AUC=result["PR_AUC"]))
    summary=pd.DataFrame(rows);summary.to_csv(OUT/"torgo_grouped_validation_calibration_comparison.csv",index=False)
    # Seed 42 only for the readable reliability comparison. All seed/fold
    # numeric metrics are retained in the table above.
    d=pd.concat(frames,ignore_index=True)
    d=d[d.seed.eq(42)]
    fig,ax=plt.subplots(figsize=(6,5.5))
    ax.plot([0,1],[0,1],"--",color="gray",label="Perfect calibration")
    for name,col,color in (("Uncalibrated","probability_raw","#C45C47"),
                           ("TORGO temperature scaled","probability","#245A81")):
        obs,mean=calibration_curve(d.label,d[col],n_bins=10,strategy="uniform")
        ax.plot(mean,obs,"o-",color=color,label=f"{name} (ECE {ece(d.label,d[col]):.3f}; Brier {brier_score_loss(d.label,d[col]):.3f})")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted dysarthria probability",
           ylabel="Observed dysarthria rate",title="TORGO grouped validation calibration — seed 42")
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/"torgo_grouped_validation_calibration.png",dpi=300,bbox_inches="tight")
    fig.savefig(OUT/"torgo_grouped_validation_calibration.pdf",bbox_inches="tight")
    plt.close(fig)
    return summary


def summarize_outer_calibration_comparison():
    """Compare raw and TORGO-calibrated scores on untouched outer speakers."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve
    path=OUT/"torgo_nested_predictions.csv"
    if not path.is_file():raise RuntimeError(f"Missing grouped outer predictions: {path}")
    predictions=pd.read_csv(path)
    rows=[]
    for (seed,fold),group in predictions.groupby(["seed","outer_fold"]):
        for variant,col in (("uncalibrated","probability_raw"),("temperature_scaled","probability")):
            observed=metrics(group.label,group[col],.5)
            rows.append(dict(seed=seed,outer_fold=fold,variant=variant,
                             n_physical_speakers=group.physical_speaker_id.nunique(),
                             Brier=observed["Brier"],ECE=observed["ECE"],
                             ROC_AUC=observed["ROC_AUC"],PR_AUC=observed["PR_AUC"]))
    result=pd.DataFrame(rows)
    if len(result)!=len(SEEDS)*5*2:raise RuntimeError("Incomplete outer calibration comparison")
    result.to_csv(OUT/"torgo_outer_calibration_comparison.csv",index=False)
    # One prespecified seed gives a readable reliability plot; all folds are pooled
    # only for visualization, and each physical speaker appears once in this seed.
    d=predictions[predictions.seed.eq(SEEDS[0])]
    if d.groupby("physical_speaker_id").outer_fold.nunique().max()!=1:
        raise AssertionError("Speaker repeated across outer folds")
    fig,ax=plt.subplots(figsize=(6,5.5))
    ax.plot([0,1],[0,1],"--",color="gray",label="Perfect calibration")
    for name,col,color in (("Uncalibrated","probability_raw","#C45C47"),
                           ("TORGO temperature scaled","probability","#245A81")):
        obs,mean=calibration_curve(d.label,d[col],n_bins=10,strategy="uniform")
        ax.plot(mean,obs,"o-",color=color,
                label=f"{name} (ECE {ece(d.label,d[col]):.3f}; Brier {brier_score_loss(d.label,d[col]):.3f})")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted dysarthria probability",
           ylabel="Observed dysarthria rate",title=f"TORGO held-out physical speakers — seed {SEEDS[0]}")
    ax.legend(fontsize=8);fig.tight_layout()
    fig.savefig(OUT/"torgo_outer_calibration.png",dpi=300,bbox_inches="tight")
    fig.savefig(OUT/"torgo_outer_calibration.pdf",bbox_inches="tight")
    plt.close(fig)
    return result


def summarize_gender_subgroups():
    """Sex-stratified discrimination only when both diagnoses are represented."""
    rows=[]
    for seed in SEEDS:
        for fold in range(5):
            base=OUT/f"seed{seed}_outer{fold}"
            p=base/"predictions.csv";ckpt=base/"checkpoint.pt"
            if not p.exists() or not ckpt.exists():raise RuntimeError(f"Missing outer-fold subgroup inputs: {base}")
            d=pd.read_csv(p);b=torch.load(ckpt,map_location="cpu",weights_only=False)
            for gender,part in d.groupby("gender"):
                has_both=part.label.nunique()==2
                result=metrics(part.label,part.probability,b["threshold"])
                rows.append(dict(seed=seed,outer_fold=fold,gender=gender,
                    n_control_physical_speakers=part[part.label.eq(0)].physical_speaker_id.nunique(),
                    n_dysarthric_physical_speakers=part[part.label.eq(1)].physical_speaker_id.nunique(),
                    ROC_AUC=result["ROC_AUC"] if has_both else float("nan"),
                    PR_AUC=result["PR_AUC"] if has_both else float("nan"),
                    Balanced_Accuracy=result["Balanced_Accuracy"] if has_both else float("nan"),
                    estimable=has_both,
                    note="Descriptive only; no fairness conclusion" if has_both else "One diagnosis class; discrimination non-estimable"))
    df=pd.DataFrame(rows);df.to_csv(OUT/"torgo_gender_subgroup_metrics.csv",index=False)
    return df


def summarize_threshold_policies():
    """Compare four prespecified policies on TORGO grouped inner OOF only."""
    rows=[]
    for seed in SEEDS:
        for fold in range(5):
            path=OUT/f"seed{seed}_outer{fold}"/"threshold_policies.csv"
            if not path.exists():raise RuntimeError(f"Missing threshold policy table: {path}")
            d=pd.read_csv(path);d["seed"]=seed;d["outer_fold"]=fold;rows.append(d)
    long=pd.concat(rows,ignore_index=True)
    long.to_csv(OUT/"torgo_grouped_validation_threshold_policies.csv",index=False)
    cols=["threshold","F1","Sensitivity","Specificity","Youden_J","Balanced_Accuracy"]
    summary=long.groupby("policy")[cols].agg(["mean","std"])
    summary.columns=[f"{a}_{b}" for a,b in summary.columns]
    summary=summary.reset_index();summary.to_csv(OUT/"torgo_grouped_validation_threshold_policy_summary.csv",index=False)
    return summary


def relock_exact_thresholds():
    """Apply exact TORGO OOF score-boundary search to saved frozen fits.

    This changes only TORGO-derived thresholds and downstream TORGO metrics;
    model weights, temperatures, and predictions remain untouched. Run before
    any official UASpeech inference.
    """
    archive=OUT/"historical_approximate_threshold"
    if not archive.exists():
        archive.mkdir()
        for seed in SEEDS:
            for fold in range(5):
                source=OUT/f"seed{seed}_outer{fold}"
                target=archive/source.name;target.mkdir()
                for name in ("checkpoint.pt","evaluation.csv","threshold_policies.csv"):
                    shutil.copy2(source/name,target/name)
            source=OUT/f"deployment_seed{seed}"
            target=archive/source.name;target.mkdir()
            for name in ("checkpoint.pt","threshold_policies.csv"):
                shutil.copy2(source/name,target/name)
        for name in ("torgo_nested_fold_metrics.csv","torgo_noise_results.csv"):
            source=OUT/name
            if source.exists():shutil.copy2(source,archive/name)
        (archive/"README.txt").write_text(
            "Preserved pilot V7 threshold outputs from the approximate 5000-candidate search. "
            "Excluded from final V7 summaries and figures. Final thresholds are recomputed "
            "exactly at every distinct TORGO grouped-validation probability.\n")
    all_eval=[];deployment_rows=[]
    for seed in SEEDS:
        for fold in range(5):
            base=OUT/f"seed{seed}_outer{fold}"
            oof=pd.read_csv(base/"inner_oof.csv")
            policies=threshold_table(oof.label,oof.probability,oof.physical_speaker_id)
            thr=float(policies.set_index("policy").loc["Specificity_at_sensitivity_0.90","threshold"])
            path=base/"checkpoint.pt";bundle=torch.load(path,map_location="cpu",weights_only=False)
            if bundle.get("pipeline_signature")!=SIGNATURE:raise RuntimeError(f"Stale TORGO fit: {path}")
            if bundle.get("threshold_search_signature")!=THRESHOLD_SIGNATURE or abs(float(bundle["threshold"])-thr)>1e-12:
                bundle["threshold"]=thr;bundle["threshold_search_signature"]=THRESHOLD_SIGNATURE
                tmp=path.with_suffix(".tmp");torch.save(bundle,tmp);tmp.replace(path)
            policies.to_csv(base/"threshold_policies.csv",index=False)
            pred=pd.read_csv(base/"predictions.csv")
            for level,frame in (("utterance",pred),("physical_speaker",speaker_aggregate(pred))):
                all_eval.append(dict(seed=seed,outer_fold=fold,level=level,threshold=thr,
                    temperature=bundle["temperature"],**metrics(frame.label,frame.probability,thr)))
            pd.DataFrame(all_eval[-2:]).to_csv(base/"evaluation.csv",index=False)
        base=OUT/f"deployment_seed{seed}";path=base/"checkpoint.pt"
        if not path.exists():raise RuntimeError(f"Deployment checkpoint missing: {path}")
        oof=pd.read_csv(base/"torgo_grouped_oof.csv")
        policies=threshold_table(oof.label,oof.probability,oof.physical_speaker_id)
        thr=float(policies.set_index("policy").loc["Specificity_at_sensitivity_0.90","threshold"])
        bundle=torch.load(path,map_location="cpu",weights_only=False)
        if bundle.get("pipeline_signature")!=SIGNATURE:raise RuntimeError(f"Stale TORGO deployment fit: {path}")
        if bundle.get("threshold_search_signature")!=THRESHOLD_SIGNATURE or abs(float(bundle["threshold"])-thr)>1e-12:
            bundle["threshold"]=thr;bundle["threshold_search_signature"]=THRESHOLD_SIGNATURE
            tmp=path.with_suffix(".tmp");torch.save(bundle,tmp);tmp.replace(path)
        policies.to_csv(base/"threshold_policies.csv",index=False)
        deployment_rows.append(dict(seed=seed,threshold=thr,temperature=bundle["temperature"],checkpoint=str(path)))
    summary=pd.DataFrame(all_eval);summary.to_csv(OUT/"torgo_nested_fold_metrics.csv",index=False)
    deployment=pd.DataFrame(deployment_rows);deployment.to_csv(OUT/"deployment_manifest.csv",index=False)
    (OUT/"threshold_search_audit.json").write_text(json.dumps(dict(
        threshold_search_signature=THRESHOLD_SIGNATURE,
        source="TORGO grouped inner out-of-fold predictions only",
        n_outer_models=25,n_deployment_models=5,official_uaspeech_labels_used=False),indent=2))
    return summary,deployment
