"""Additive V7 analyses. Historical V7/V6 artifacts are never overwritten."""
from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score,
    brier_score_loss, confusion_matrix, f1_score, roc_auc_score)
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedGroupKFold

from .methodology import (_cache_paths, apply_temperature, assert_disjoint,
    calibrate_temperature, threshold_table, train_one, predict)

from .methodology import (OUT, SEEDS, audit_physical_speakers, ece, grouped_splits,
    load_torgo, metrics, speaker_aggregate)

METRIC_COLUMNS = ("Accuracy", "Balanced_Accuracy", "Sensitivity", "Specificity",
    "Precision", "NPV", "F1_dysarthric", "Macro_F1", "ROC_AUC", "PR_AUC",
    "Brier", "ECE")
PAPER_BASELINE_COLUMNS=("Accuracy","Balanced_Accuracy","Sensitivity","Specificity",
    "F1_dysarthric","Macro_F1","ROC_AUC","PR_AUC","Brier","ECE")
PAPER_BASELINE_RENAME={"model":"Model","Balanced_Accuracy":"Balanced Accuracy",
    "F1_dysarthric":"F1","Macro_F1":"Macro F1","ROC_AUC":"ROC-AUC",
    "PR_AUC":"PR-AUC"}


def _single_metric(y,p,threshold,metric):
    """Bootstrap inner loop computes only the requested statistic."""
    y=np.asarray(y,int);p=np.asarray(p,float)
    if metric=="ROC_AUC":
        return roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan
    if metric=="PR_AUC":
        return average_precision_score(y,p) if len(np.unique(y))==2 else np.nan
    if metric=="Brier":return float(np.mean((p-y)**2))
    if metric=="ECE":return ece(y,p)
    pred=p>=np.asarray(threshold,float)
    tp=int(np.sum((y==1)&pred));tn=int(np.sum((y==0)&~pred))
    fp=int(np.sum((y==0)&pred));fn=int(np.sum((y==1)&~pred))
    ratio=lambda a,b:float(a/b) if b else np.nan
    if metric=="Accuracy":return (tp+tn)/len(y)
    if metric=="Sensitivity":return ratio(tp,tp+fn)
    if metric=="Specificity":return ratio(tn,tn+fp)
    if metric=="Balanced_Accuracy":
        return (ratio(tp,tp+fn)+ratio(tn,tn+fp))/2
    if metric=="Precision":return ratio(tp,tp+fp)
    if metric=="NPV":return ratio(tn,tn+fn)
    f1_pos=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0
    if metric=="F1_dysarthric":return f1_pos
    f1_neg=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.0
    if metric=="Macro_F1":return (f1_pos+f1_neg)/2
    raise ValueError(metric)


def _variable_threshold_metrics(frame):
    y = frame.label.to_numpy(int)
    p = frame.probability.to_numpy(float)
    pred = (p >= frame.threshold.to_numpy(float)).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a / b) if b else float("nan")
    two = len(np.unique(y)) == 2
    return dict(Accuracy=accuracy_score(y, pred),
        Balanced_Accuracy=(ratio(tp,tp+fn)+ratio(tn,tn+fp))/2 if two else np.nan,
        Sensitivity=ratio(tp,tp+fn), Specificity=ratio(tn,tn+fp),
        Precision=ratio(tp,tp+fp), NPV=ratio(tn,tn+fn),
        F1_dysarthric=f1_score(y,pred,zero_division=0),
        Macro_F1=f1_score(y,pred,average="macro",labels=[0,1],zero_division=0),
        ROC_AUC=roc_auc_score(y,p) if two else np.nan,
        PR_AUC=average_precision_score(y,p) if two else np.nan,
        Brier=brier_score_loss(y,p), ECE=ece(y,p), n=len(frame))


def pool_stream1_oof(bootstrap_reps=2000):
    """Pool existing physical-speaker-safe V7 outer predictions across all folds."""
    meta, _, _ = load_torgo()
    meta, _ = audit_physical_speakers(meta, save=False)
    expected = meta[["file_path", "physical_speaker_id", "label"]].copy()
    frames = []
    for seed in SEEDS:
        for fold, tr, te in grouped_splits(meta, seed):
            base = OUT / f"seed{seed}_outer{fold}"
            manifest = json.loads((base / "split_manifest.json").read_text())
            train_sp = set(meta.iloc[tr].physical_speaker_id)
            test_sp = set(meta.iloc[te].physical_speaker_id)
            assert train_sp.isdisjoint(test_sp)
            assert train_sp == set(manifest["train_physical_speakers"])
            assert test_sp == set(manifest["test_physical_speakers"])
            bundle = torch.load(base / "checkpoint.pt", map_location="cpu", weights_only=False)
            assert set(bundle["train_physical_speakers"]) == train_sp
            p = pd.read_csv(base / "predictions.csv")
            assert len(p) == len(te)
            assert set(p.file_path) == set(meta.iloc[te].file_path)
            assert set(p.physical_speaker_id) == test_sp
            assert p.file_path.is_unique
            p["threshold"] = float(bundle["threshold"])
            frames.append(p)
    utterance = pd.concat(frames, ignore_index=True)
    for seed, part in utterance.groupby("seed"):
        assert part.file_path.is_unique
        assert set(part.file_path) == set(expected.file_path)
        assert (part.groupby("physical_speaker_id").outer_fold.nunique() == 1).all()
        assert part.physical_speaker_id.nunique() == 15
    utterance.to_csv(OUT / "v7_pooled_oof_torgo_predictions.csv", index=False)
    speakers = utterance.groupby(["seed", "physical_speaker_id"], as_index=False).agg(
        label=("label", "first"), gender=("gender", "first"),
        probability=("probability", "mean"), probability_raw=("probability_raw", "mean"),
        threshold=("threshold", "first"), outer_fold=("outer_fold", "first"),
        n_utterances=("file_path", "size"))
    assert speakers.groupby("seed").size().eq(15).all()
    speakers.to_csv(OUT / "v7_pooled_oof_torgo_speakers.csv", index=False)
    rows = []
    for seed in SEEDS:
        for level, data in (("utterance", utterance), ("physical_speaker", speakers)):
            part = data[data.seed.eq(seed)]
            rows.append(dict(seed=seed, level=level, **_variable_threshold_metrics(part)))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "v7_pooled_oof_speaker_metrics.csv", index=False)
    summary = []
    for level, data in result.groupby("level"):
        for col in METRIC_COLUMNS:
            v = data[col].dropna()
            summary.append(dict(level=level, metric=col, mean=v.mean(),
                sd=v.std(ddof=1), n_seeds=len(v)))
    pd.DataFrame(summary).to_csv(OUT / "v7_pooled_oof_mean_sd.csv", index=False)
    rng = np.random.default_rng(4207)
    ci = []
    # Resample whole physical speakers within each seed, keeping fold-locked thresholds.
    for seed in SEEDS:
        part = speakers[speakers.seed.eq(seed)].reset_index(drop=True)
        for metric in METRIC_COLUMNS:
            vals = []
            for _ in range(bootstrap_reps):
                sample = part.iloc[rng.integers(0, len(part), len(part))]
                if sample.label.nunique() < 2:
                    continue
                vals.append(_single_metric(sample.label,sample.probability,
                    sample.threshold,metric))
            v = np.asarray(vals, float)
            finite = v[np.isfinite(v)]
            ci.append(dict(seed=seed, metric=metric,
            point=_single_metric(part.label,part.probability,part.threshold,metric),
                ci_low=np.percentile(finite,2.5) if len(finite) else np.nan,
                ci_high=np.percentile(finite,97.5) if len(finite) else np.nan,
                n_valid_bootstrap=len(finite)))
    pd.DataFrame(ci).to_csv(OUT / "v7_pooled_oof_speaker_cluster_ci.csv", index=False)
    return result


def convergence_evidence():
    """Summarize what historical V7 checkpoints prove; do not invent curves."""
    existing=OUT/"v7_convergence_summary.csv"
    if existing.exists():
        previous=pd.read_csv(existing)
        if len(previous)==25 and previous.epochs_run.notna().all():
            return previous
    rows = []
    for seed in SEEDS:
        for fold in range(5):
            path = OUT / f"seed{seed}_outer{fold}" / "checkpoint.pt"
            b = torch.load(path, map_location="cpu", weights_only=False)
            rows.append(dict(seed=seed, fold=fold, epochs_run=np.nan,
                best_epoch=np.nan, best_val_metric=np.nan, hit_max_epoch=np.nan,
                early_stopped=np.nan, converged=np.nan,
                final_fit_epochs=b["selected_epochs"],
                evidence="Per-epoch inner-fold histories not saved; final-fit epoch count is median inner best epoch."))
    result = pd.DataFrame(rows)
    result.to_csv(OUT / "v7_convergence_summary.csv", index=False)
    return result


def replay_convergence_curves():
    """Replay the 75 grouped inner fits solely to recover missing learning curves.

    This is a diagnostic replay; historical outer checkpoints and predictions are
    retained, and official UASpeech is never consulted.
    """
    meta, features, _ = load_torgo()
    meta, _ = audit_physical_speakers(meta,save=False)
    rows=[]
    for seed in SEEDS:
        for fold,outer_train,outer_test in grouped_splits(meta,seed):
            local=meta.iloc[outer_train].reset_index(drop=True)
            for inner,tri,vai in grouped_splits(local,seed+731,n_splits=3):
                train,val=outer_train[tri],outer_train[vai]
                assert_disjoint(meta,train,val,outer_test)
                path=OUT/f"v7_corrected_convergence_seed{seed}_outer{fold}_inner{inner}.csv"
                if not path.exists():
                    model,_=train_one(meta,features,train,seed+inner,
                        f"seed{seed}_outer{fold}_inner{inner}",epochs=20,val_indices=val,
                        samples_per_epoch=512,history_path=path)
                    del model
                h=pd.read_csv(path)
                best=h.loc[h.val_loss.idxmin()]
                hit_max=int(h.epoch.iloc[-1])==20
                early_stop=not hit_max and len(h)>=5 and int(h.stale_epochs.iloc[-1])>=4
                trailing=h.tail(min(3,len(h)))
                improving_loss=len(trailing)>1 and trailing.val_loss.iloc[-1]<trailing.val_loss.iloc[0]
                improving_auc=len(trailing)>1 and trailing.val_roc_auc.iloc[-1]>trailing.val_roc_auc.iloc[0]+.01
                rows.append(dict(seed=seed,fold=fold,inner_fold=inner,
                    epochs_run=int(h.epoch.iloc[-1]),best_epoch=int(best.epoch),
                    best_val_metric=float(best.val_loss),
                    best_val_metric_name="speaker_weighted_log_loss",
                    best_val_roc_auc=float(h.val_roc_auc.max()),
                    best_val_f1=float(h.val_f1_at_0_5.max()),
                    hit_max_epoch=hit_max,early_stopped=early_stop,
                    converged=bool(early_stop and not improving_loss and not improving_auc),
                    final_val_loss=float(h.val_loss.iloc[-1]),
                    final_val_roc_auc=float(h.val_roc_auc.iloc[-1]),
                    curve_file=path.name))
                pd.DataFrame(rows).to_csv(OUT/"v7_convergence_inner_summary.csv",index=False)
                print("Convergence replay",seed,fold,inner,"epochs",len(h),flush=True)
    details=pd.DataFrame(rows)
    fold_summary=details.groupby(["seed","fold"],as_index=False).agg(
        epochs_run=("epochs_run","max"),best_epoch=("best_epoch","median"),
        best_val_metric=("best_val_metric","mean"),
        hit_max_epoch=("hit_max_epoch","any"),
        early_stopped=("early_stopped","all"),converged=("converged","all"),
        n_inner_folds=("inner_fold","size"))
    selected=[]
    for row in fold_summary.itertuples():
        b=torch.load(OUT/f"seed{row.seed}_outer{row.fold}"/"checkpoint.pt",
            map_location="cpu",weights_only=False)
        selected.append(int(b["selected_epochs"]))
    fold_summary["historical_final_fit_epochs"]=selected
    fold_summary["replay_matches_historical_epoch_choice"]=(
        fold_summary.best_epoch.astype(int)==fold_summary.historical_final_fit_epochs)
    fold_summary["converged"] &= fold_summary.replay_matches_historical_epoch_choice
    fold_summary.to_csv(OUT/"v7_convergence_summary.csv",index=False)
    return fold_summary


def training_budget_decision():
    """Lock the epoch-budget decision from TORGO curves alone."""
    path=OUT/"v7_convergence_inner_summary.csv"
    if not path.exists():raise RuntimeError("No grouped TORGO convergence curves")
    details=pd.read_csv(path)
    if len(details)!=75:
        raise RuntimeError(f"Need 75 grouped inner curves; found {len(details)}")
    hit=details[details.hit_max_epoch]
    still_improving=0
    for row in hit.itertuples():
        h=pd.read_csv(OUT/row.curve_file)
        tail=h.tail(3)
        if (tail.val_loss.iloc[-1]<tail.val_loss.iloc[0] or
            tail.val_roc_auc.iloc[-1]>tail.val_roc_auc.iloc[0]+.01):
            still_improving+=1
    increase=(len(hit)>=19 and still_improving>=10)
    result=dict(source="TORGO grouped inner validation only",n_inner_fits=len(details),
        hit_20_epoch_cap=int(len(hit)),still_improving_at_cap=int(still_improving),
        early_stopped=int(details.early_stopped.sum()),
        unstable_by_curve_criterion=int((~details.converged.astype(bool)).sum()),
        previous_max_epochs=20,selected_max_epochs=75 if increase else 20,
        samples_per_epoch=512,
        decision="Increase maximum epochs and rerun TORGO models" if increase else
            "Retain 20-epoch maximum; no evidence that the cap truncates many grouped validation fits",
        official_uaspeech_consulted=False)
    (OUT/"v7_training_budget_decision.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


def _classical_fit(meta, x, indices, model, seed):
    """All trainable feature transforms are fitted inside this training subset."""
    y = meta.iloc[indices].label.to_numpy(int)
    speaker = meta.iloc[indices].physical_speaker_id.to_numpy()
    counts = pd.Series(speaker).value_counts()
    weight = np.array([1 / counts[s] for s in speaker])
    weight *= len(weight) / weight.sum()
    if model == "SVM (RBF)":
        estimator = SVC(kernel="rbf", class_weight="balanced", random_state=seed)
    elif model == "Random Forest":
        estimator = RandomForestClassifier(n_estimators=300, class_weight="balanced",
            random_state=seed, n_jobs=4)
    else:
        raise ValueError(model)
    pipe = make_pipeline(StandardScaler(), SelectKBest(f_classif, k=60), estimator)
    pipe.fit(x[indices], y, svc__sample_weight=weight) if model == "SVM (RBF)" else pipe.fit(
        x[indices], y, randomforestclassifier__sample_weight=weight)
    return pipe


def _classical_score(pipe, x, indices, model):
    if model == "SVM (RBF)":
        return pipe.decision_function(x[indices])
    return pipe.predict_proba(x[indices])[:,1]


def _stack_fit_predict(meta, x, train, target, seed):
    """Grouped OOF level-one training; no utterance-level stacking CV."""
    local = meta.iloc[train].reset_index(drop=True)
    oof = np.full((len(train),2), np.nan)
    for inner, tri, vai in grouped_splits(local, seed+1731, n_splits=3):
        a, b = train[tri], train[vai]
        assert set(meta.iloc[a].physical_speaker_id).isdisjoint(set(meta.iloc[b].physical_speaker_id))
        for j,name in enumerate(("SVM (RBF)", "Random Forest")):
            pipe = _classical_fit(meta,x,a,name,seed+inner)
            oof[vai,j] = _classical_score(pipe,x,b,name)
    assert np.isfinite(oof).all()
    y = meta.iloc[train].label.to_numpy(int)
    speaker = meta.iloc[train].physical_speaker_id.to_numpy()
    counts = pd.Series(speaker).value_counts()
    weight = np.array([1/counts[s] for s in speaker])
    meta_model = make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=seed))
    meta_model.fit(oof,y,logisticregression__sample_weight=weight)
    target_scores = np.column_stack([_classical_score(
        _classical_fit(meta,x,train,name,seed),x,target,name)
        for name in ("SVM (RBF)","Random Forest")])
    return meta_model.predict_proba(target_scores)[:,1]


def run_corrected_classical_baselines():
    """Five-seed physical-speaker outer folds with grouped inner calibration."""
    meta, _, _ = load_torgo()
    meta, _ = audit_physical_speakers(meta,save=False)
    feature_path=_cache_paths()/"classical_features.npy"
    manifest=json.loads(feature_path.with_suffix(".npy.manifest.json").read_text())
    columns=["file_path","mic_type","label","speaker_id","gender"]
    records=[[str(v) for v in row] for row in meta[columns].itertuples(index=False,name=None)]
    digest=hashlib.sha256(json.dumps(records,sort_keys=False,ensure_ascii=True).encode()).hexdigest()
    if not manifest.get("complete") or digest!=manifest.get("source_signature"):
        raise AssertionError("Classical feature rows do not match verified TORGO metadata")
    x = np.load(feature_path,mmap_mode="r")
    assert len(x)==len(meta) and x.shape[1]==126 and np.isfinite(x).all()
    names=("SVM (RBF)","Random Forest","Stacking Ensemble")
    all_predictions=[]
    for seed in SEEDS:
        for fold,outer_train,outer_test in grouped_splits(meta,seed):
            for name in names:
                safe=name.lower().replace(" ","_").replace("(","").replace(")","")
                path=OUT/f"v7_corrected_classical_seed{seed}_fold{fold}_{safe}.csv"
                if path.exists():
                    saved=pd.read_csv(path)
                    assert set(saved.file_path)==set(meta.iloc[outer_test].file_path)
                    assert set(saved.physical_speaker_id)==set(meta.iloc[outer_test].physical_speaker_id)
                    assert set(meta.iloc[outer_train].physical_speaker_id).isdisjoint(
                        set(saved.physical_speaker_id))
                    all_predictions.append(saved)
                    continue
                inner_meta=meta.iloc[outer_train].reset_index(drop=True)
                oof=np.full(len(outer_train),np.nan)
                for inner,tri,vai in grouped_splits(inner_meta,seed+731,n_splits=3):
                    train,val=outer_train[tri],outer_train[vai]
                    assert_disjoint(meta,train,val,outer_test)
                    if name=="Stacking Ensemble":
                        score=_stack_fit_predict(meta,x,train,val,seed+inner)
                    else:
                        pipe=_classical_fit(meta,x,train,name,seed+inner)
                        score=_classical_score(pipe,x,val,name)
                    oof[vai]=score
                assert np.isfinite(oof).all()
                y=meta.iloc[outer_train].label.to_numpy(int)
                sp=meta.iloc[outer_train].physical_speaker_id.to_numpy()
                if name=="SVM (RBF)":
                    # Sigmoid is trained on physical-speaker-held-out OOF scores.
                    counts=pd.Series(sp).value_counts()
                    weight=np.array([1/counts[s] for s in sp])
                    cal=LogisticRegression(max_iter=1000,random_state=seed).fit(
                        oof[:,None],y,sample_weight=weight)
                    val_prob=cal.predict_proba(oof[:,None])[:,1]
                else:
                    cal=None
                    val_prob=np.clip(oof,1e-7,1-1e-7)
                temp=calibrate_temperature(y,val_prob,sp)
                val_cal=apply_temperature(val_prob,temp)
                policies=threshold_table(y,val_cal,sp)
                threshold=float(policies.set_index("policy").loc[
                    "Specificity_at_sensitivity_0.90","threshold"])
                if name=="Stacking Ensemble":
                    raw=_stack_fit_predict(meta,x,outer_train,outer_test,seed+10000)
                else:
                    pipe=_classical_fit(meta,x,outer_train,name,seed+10000)
                    raw=_classical_score(pipe,x,outer_test,name)
                if cal is not None:
                    raw=cal.predict_proba(raw[:,None])[:,1]
                pred=meta.iloc[outer_test][["file_path","physical_speaker_id","label","gender","mic_type"]].copy()
                pred["probability_raw"]=raw
                pred["probability"]=apply_temperature(raw,temp)
                pred["threshold"]=threshold
                pred["temperature"]=temp
                pred["seed"]=seed
                pred["outer_fold"]=fold
                pred["model"]=name
                pred.to_csv(path,index=False)
                policies.to_csv(path.with_suffix(".threshold_policies.csv"),index=False)
                all_predictions.append(pred)
                print("Completed classical",seed,fold,name,flush=True)
    result=pd.concat(all_predictions,ignore_index=True)
    result.to_csv(OUT/"v7_corrected_classical_predictions.csv",index=False)
    metric_rows=[]
    for (name,seed),part in result.groupby(["model","seed"]):
        assert part.file_path.is_unique and part.physical_speaker_id.nunique()==15
        speaker=part.groupby("physical_speaker_id",as_index=False).agg(
            label=("label","first"),probability=("probability","mean"),
            threshold=("threshold","first"))
        metric_rows.append(dict(model=name,seed=seed,level="utterance",**_variable_threshold_metrics(part)))
        metric_rows.append(dict(model=name,seed=seed,level="physical_speaker",**_variable_threshold_metrics(speaker)))
    metric_df=pd.DataFrame(metric_rows)
    metric_df.to_csv(OUT/"v7_corrected_classical_metrics.csv",index=False)
    return metric_df


def backfill_classical_grouped_validation_scores(seeds=SEEDS,folds=range(5),
    names=("SVM (RBF)","Random Forest","Stacking Ensemble")):
    """Save and verify the original train-only OOF calibration inputs.

    This deterministic replay does not refit or replace any saved outer-test
    prediction. It checks that recovered TORGO-only temperatures and primary
    thresholds equal those used by the completed classical comparison.
    """
    meta,_,_=load_torgo()
    meta,_=audit_physical_speakers(meta,save=False)
    x=np.load(_cache_paths()/"classical_features.npy",mmap_mode="r")
    index=[]
    for seed in seeds:
        for fold,outer_train,outer_test in grouped_splits(meta,seed):
            if fold not in folds:continue
            local=meta.iloc[outer_train].reset_index(drop=True)
            for name in names:
                safe=name.lower().replace(" ","_").replace("(","").replace(")","")
                base=OUT/f"v7_corrected_classical_seed{seed}_fold{fold}_{safe}"
                path=base.with_suffix(".inner_oof.csv")
                outer_path=base.with_suffix(".csv")
                if not outer_path.exists():raise RuntimeError(f"Missing completed classical outer fold: {outer_path}")
                if path.exists():
                    frame=pd.read_csv(path)
                    if len(frame)!=len(outer_train) or set(frame.file_path)!=set(meta.iloc[outer_train].file_path):
                        raise RuntimeError(f"Stale classical validation OOF file: {path}")
                    index.append(dict(seed=seed,outer_fold=fold,model=name,
                        n_inner_oof=len(frame),source_file=path.name))
                    continue
                score=np.full(len(outer_train),np.nan)
                inner_assignment=np.full(len(outer_train),-1,int)
                for inner,tri,vai in grouped_splits(local,seed+731,n_splits=3):
                    train,val=outer_train[tri],outer_train[vai]
                    assert_disjoint(meta,train,val,outer_test)
                    if name=="Stacking Ensemble":
                        fold_score=_stack_fit_predict(meta,x,train,val,seed+inner)
                    else:
                        model=_classical_fit(meta,x,train,name,seed+inner)
                        fold_score=_classical_score(model,x,val,name)
                    score[vai]=fold_score;inner_assignment[vai]=inner
                if not np.isfinite(score).all() or (inner_assignment<0).any():
                    raise RuntimeError("Classical grouped validation coverage failed")
                y=meta.iloc[outer_train].label.to_numpy(int)
                sp=meta.iloc[outer_train].physical_speaker_id.to_numpy()
                if name=="SVM (RBF)":
                    counts=pd.Series(sp).value_counts()
                    weight=np.array([1/counts[s] for s in sp])
                    sigmoid=LogisticRegression(max_iter=1000,random_state=seed).fit(
                        score[:,None],y,sample_weight=weight)
                    probability_raw=sigmoid.predict_proba(score[:,None])[:,1]
                else:probability_raw=np.clip(score,1e-7,1-1e-7)
                temperature=calibrate_temperature(y,probability_raw,sp)
                probability=apply_temperature(probability_raw,temperature)
                policies=threshold_table(y,probability,sp)
                threshold=float(policies.set_index("policy").loc[
                    "Specificity_at_sensitivity_0.90","threshold"])
                outer=pd.read_csv(outer_path)
                if not np.allclose(outer.temperature,temperature,atol=1e-10) or\
                        not np.allclose(outer.threshold,threshold,atol=1e-10):
                    raise AssertionError(f"Classical TORGO calibration replay differs: {seed}/{fold}/{name}")
                original_policy=pd.read_csv(base.with_suffix(".threshold_policies.csv"))
                if not np.allclose(original_policy.threshold,policies.threshold,atol=1e-10):
                    raise AssertionError(f"Classical threshold policy replay differs: {seed}/{fold}/{name}")
                frame=meta.iloc[outer_train][["file_path","physical_speaker_id","label"]].copy()
                frame["inner_fold"]=inner_assignment
                frame["raw_score"]=score
                frame["probability_raw"]=probability_raw
                frame["probability"]=probability
                frame["temperature"]=temperature
                frame["seed"]=seed
                frame["outer_fold"]=fold
                frame["model"]=name
                frame.to_csv(path,index=False)
                index.append(dict(seed=seed,outer_fold=fold,model=name,
                    n_inner_oof=len(frame),source_file=path.name))
                print("Classical grouped OOF verified",seed,fold,name,flush=True)
    result=pd.DataFrame(index)
    result.to_csv(OUT/"v7_corrected_classical_inner_oof_index.csv",index=False)
    return result


def make_interim_baseline_table():
    """Four-model interim table; intentionally not the seven-model primary table."""
    classical_path=OUT/"v7_corrected_classical_metrics.csv"
    if not classical_path.exists():
        raise RuntimeError("Classical five-seed rerun is incomplete")
    classical=pd.read_csv(classical_path)
    assert len(classical)==3*5*2
    classical=classical[classical.level.eq("physical_speaker")]
    stream1=pd.read_csv(OUT/"v7_pooled_oof_speaker_metrics.csv")
    stream1=stream1[stream1.level.eq("physical_speaker")].copy()
    stream1["model"]="Stream 1 Only (CNN-BiLSTM) — existing V7"
    data=pd.concat([classical,stream1],ignore_index=True)
    rows=[]
    for model,part in data.groupby("model"):
        stream1_row=model.startswith("Stream 1 Only")
        row={"Model":model,"n_seeds":len(part),
            "source_metrics_file":("v7_pooled_oof_speaker_metrics.csv" if stream1_row else
                "v7_corrected_classical_metrics.csv"),
            "source_prediction_file":("v7_pooled_oof_torgo_predictions.csv" if stream1_row else
                "v7_corrected_classical_predictions.csv")}
        for col in METRIC_COLUMNS:
            row[col+"_mean"]=part[col].mean()
            row[col+"_sd"]=part[col].std(ddof=1)
            row[col]=f"{row[col+'_mean']:.3f} ± {row[col+'_sd']:.3f}"
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_interim_physical_speaker_baseline_results.csv",index=False)
    return result


def make_primary_baseline_table():
    """Release the seven-model table only after all matched neural runs exist."""
    classical=pd.read_csv(OUT/"v7_corrected_classical_metrics.csv")
    neural=pd.read_csv(OUT/"v7_corrected_architecture_ablation.csv")
    classical=classical[classical.level.eq("physical_speaker")].copy()
    neural=neural[neural.level.eq("physical_speaker")].copy()
    expected_classical={"SVM (RBF)","Random Forest","Stacking Ensemble"}
    expected_neural={"Stream 1 Only (CNN-BiLSTM)","Stream 2 Only (Transformer)",
        "Dual-Stream (No Pretraining)","Dual-Stream (Full)"}
    if set(classical.model)!=expected_classical or set(neural.model)!=expected_neural:
        raise RuntimeError("Seven matched baseline model families are not complete")
    data=pd.concat([classical,neural],ignore_index=True)
    if any(len(part)!=5 or set(part.seed)!=set(SEEDS) for _,part in data.groupby("model")):
        raise RuntimeError("A baseline lacks one or more required seeds")
    rows=[]
    for model,part in data.groupby("model"):
        is_neural=model in expected_neural
        row={"model":model,"n_seeds":5,
            "source_metrics_file":"v7_corrected_architecture_ablation.csv" if is_neural else
                "v7_corrected_classical_metrics.csv",
            "source_prediction_file":"v7_corrected_architecture_predictions.csv" if is_neural else
                "v7_corrected_classical_predictions.csv"}
        for metric in METRIC_COLUMNS:
            row[metric+"_mean"]=part[metric].mean()
            row[metric+"_sd"]=part[metric].std(ddof=1)
            row[metric]=f"{row[metric+'_mean']:.3f} ± {row[metric+'_sd']:.3f}"
        rows.append(row)
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_corrected_baseline_results.csv",index=False)
    paper=result[["model",*PAPER_BASELINE_COLUMNS]].rename(columns=PAPER_BASELINE_RENAME)
    paper.to_csv(OUT/"v7_corrected_baseline_paper_table.csv",index=False)
    neural_rows=result[result.model.isin(expected_neural)].copy()
    neural_rows.to_csv(OUT/"v7_corrected_architecture_summary.csv",index=False)
    rankings=[]
    for seed,part in neural.groupby("seed"):
        ordered=part.sort_values("ROC_AUC",ascending=False)
        for rank,row in enumerate(ordered.itertuples(),1):
            rankings.append(dict(seed=seed,rank=rank,model=row.model,ROC_AUC=row.ROC_AUC,
                source_prediction_file="v7_corrected_architecture_predictions.csv"))
    ranking=pd.DataFrame(rankings)
    ranking.to_csv(OUT/"v7_corrected_architecture_rankings.csv",index=False)
    (OUT/"v7_corrected_architecture_ranking_audit.json").write_text(json.dumps(dict(
        metric="pooled physical-speaker ROC_AUC",seeds=list(SEEDS),
        winner_varies_across_seeds=ranking[ranking["rank"].eq(1)].model.nunique()>1,
        winners_by_seed=ranking[ranking["rank"].eq(1)].set_index("seed").model.to_dict()),
        indent=2)+"\n")
    return result


def pooled_subgroup_analysis():
    """Subgroup estimates use each physical speaker once per seed."""
    path=OUT/"v7_pooled_oof_torgo_speakers.csv"
    if not path.exists():raise RuntimeError("Pooled speaker predictions missing")
    speaker=pd.read_csv(path)
    rows=[]
    for seed,part in speaker.groupby("seed"):
        for gender in ("female","male"):
            subset=part[part.gender.eq(gender)]
            r=_variable_threshold_metrics(subset)
            rows.append(dict(seed=seed,subgroup=gender,
                n_physical_speakers=len(subset),estimable=subset.label.nunique()==2,
                **r))
            for diagnosis,label in (("control",0),("dysarthric",1)):
                one=subset[subset.label.eq(label)]
                if one.empty:continue
                r=_variable_threshold_metrics(one)
                rows.append(dict(seed=seed,subgroup=f"{gender} {diagnosis}",
                    n_physical_speakers=len(one),estimable=False,**r))
    result=pd.DataFrame(rows)
    assert result.loc[~result.estimable,"ROC_AUC"].isna().all()
    result["ROC_AUC_status"]=np.where(result.estimable,"estimable","not estimable")
    result.to_csv(OUT/"v7_corrected_pooled_subgroup_metrics.csv",index=False)
    return result


def fold_subgroup_estimability_analysis():
    """Retain one-class fold AUC as NA; summarize only estimable folds."""
    pred=pd.read_csv(OUT/"v7_pooled_oof_torgo_predictions.csv")
    speaker=pred.groupby(["seed","outer_fold","physical_speaker_id"],as_index=False).agg(
        label=("label","first"),gender=("gender","first"),
        probability=("probability","mean"),threshold=("threshold","first"))
    rows=[]
    for (seed,fold),frame in speaker.groupby(["seed","outer_fold"]):
        for gender in ("female","male"):
            part=frame[frame.gender.eq(gender)]
            if part.empty:continue
            for subgroup,subset in [(gender,part)]+[
                (f"{gender} {name}",part[part.label.eq(label)])
                for name,label in (("control",0),("dysarthric",1))]:
                if subset.empty:continue
                score=_variable_threshold_metrics(subset)
                rows.append(dict(seed=seed,outer_fold=fold,subgroup=subgroup,
                    n_physical_speakers=len(subset),
                    estimable=subset.label.nunique()==2,**score))
    result=pd.DataFrame(rows)
    if not result.loc[~result.estimable,"ROC_AUC"].isna().all():
        raise AssertionError("One-class subgroup ROC-AUC must remain not estimable")
    result["ROC_AUC_status"]=np.where(result.estimable,"estimable","not estimable")
    result.to_csv(OUT/"v7_corrected_fold_subgroup_metrics.csv",index=False)
    valid=result[result.estimable].groupby("subgroup",as_index=False).agg(
        n_valid_folds=("ROC_AUC","size"),ROC_AUC_mean=("ROC_AUC","mean"),
        ROC_AUC_sd=("ROC_AUC","std"),Balanced_Accuracy_mean=("Balanced_Accuracy","mean"),
        F1_mean=("F1_dysarthric","mean"))
    valid.to_csv(OUT/"v7_corrected_valid_fold_subgroup_summary.csv",index=False)
    return result,valid


def pooled_calibration_comparison():
    """Evaluate fitted TORGO calibration only on held-out outer speakers."""
    prediction=OUT/"v7_pooled_oof_torgo_predictions.csv"
    if not prediction.exists():raise RuntimeError("Pooled TORGO predictions missing")
    data=pd.read_csv(prediction)
    rows=[]
    for seed,part in data.groupby("seed"):
        for level,frame in (("utterance",part),("physical_speaker",
                part.groupby("physical_speaker_id",as_index=False).agg(
                    label=("label","first"),probability=("probability","mean"),
                    probability_raw=("probability_raw","mean")))):
            for variant,col in (("Uncalibrated","probability_raw"),
                    ("Temperature scaled","probability")):
                y,p=frame.label.to_numpy(int),frame[col].to_numpy(float)
                rows.append(dict(seed=seed,level=level,variant=variant,
                    Brier=brier_score_loss(y,p),ECE=ece(y,p),
                    ROC_AUC=roc_auc_score(y,p),PR_AUC=average_precision_score(y,p),
                    n=len(frame)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_corrected_pooled_calibration_comparison.csv",index=False)
    return result


def classical_calibration_comparison():
    """Untouched outer-test calibration transfer for all corrected classical fits."""
    path=OUT/"v7_corrected_classical_predictions.csv"
    if not path.exists():raise RuntimeError("Classical OOF predictions missing")
    data=pd.read_csv(path)
    rows=[]
    for (model,seed),part in data.groupby(["model","seed"]):
        speaker=part.groupby("physical_speaker_id",as_index=False).agg(
            label=("label","first"),probability=("probability","mean"),
            probability_raw=("probability_raw","mean"))
        for level,frame in (("utterance",part),("physical_speaker",speaker)):
            for variant,col in (("Uncalibrated","probability_raw"),
                    ("Temperature scaled","probability")):
                y,p=frame.label.to_numpy(int),frame[col].to_numpy(float)
                rows.append(dict(model=model,seed=seed,level=level,variant=variant,
                    Brier=brier_score_loss(y,p),ECE=ece(y,p),
                    ROC_AUC=roc_auc_score(y,p),PR_AUC=average_precision_score(y,p),
                    n=len(frame)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_corrected_classical_calibration_comparison.csv",index=False)
    return result


def pooled_five_seed_cluster_ci(reps=2000):
    """Paired speaker bootstrap for the mean of the five seedwise estimates."""
    data=pd.read_csv(OUT/"v7_pooled_oof_torgo_speakers.csv")
    frames=[data[data.seed.eq(seed)].sort_values("physical_speaker_id").reset_index(drop=True)
        for seed in SEEDS]
    assert all(f.physical_speaker_id.tolist()==frames[0].physical_speaker_id.tolist() for f in frames)
    arrays=[(f.label.to_numpy(int),f.probability.to_numpy(float),
        f.threshold.to_numpy(float)) for f in frames]
    rng=np.random.default_rng(4250)
    rows=[]
    for metric in METRIC_COLUMNS:
        point=np.mean([_single_metric(y,p,t,metric) for y,p,t in arrays])
        vals=[]
        for _ in range(reps):
            ix=rng.integers(0,len(frames[0]),len(frames[0]))
            if frames[0].iloc[ix].label.nunique()<2:continue
            seed_values=np.array([_single_metric(y[ix],p[ix],t[ix],metric)
                for y,p,t in arrays],float)
            if np.isfinite(seed_values).all():vals.append(seed_values.mean())
        rows.append(dict(metric=metric,point_five_seed_mean=point,
            ci_low=np.percentile(vals,2.5) if vals else np.nan,
            ci_high=np.percentile(vals,97.5) if vals else np.nan,
            n_valid_bootstrap=len(vals)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_pooled_oof_five_seed_speaker_ci.csv",index=False)
    return result


def threshold_policy_sensitivity_analysis():
    """Compare prespecified TORGO-only policies without selecting on test labels."""
    policies=pd.read_csv(OUT/"torgo_grouped_validation_threshold_policies.csv")
    pred=pd.read_csv(OUT/"v7_pooled_oof_torgo_predictions.csv")
    expected={"F1_optimal","Youden_J","Balanced_Accuracy_optimal",
        "Specificity_at_sensitivity_0.90"}
    if len(policies)!=len(SEEDS)*5*len(expected) or set(policies.policy)!=expected:
        raise RuntimeError("Expected four grouped-validation policies per outer fold")
    if policies.duplicated(["seed","outer_fold","policy"]).any():
        raise RuntimeError("Duplicate TORGO validation threshold policy")
    validation=policies.groupby("policy",as_index=False).agg(
        n_folds=("threshold","size"),threshold_mean=("threshold","mean"),
        threshold_sd=("threshold","std"),
        validation_F1_mean=("F1","mean"),
        validation_Balanced_Accuracy_mean=("Balanced_Accuracy","mean"),
        validation_Sensitivity_mean=("Sensitivity","mean"),
        validation_Specificity_mean=("Specificity","mean"))
    validation["source_file"]="torgo_grouped_validation_threshold_policies.csv"
    validation["primary_policy"]=validation.policy.eq("Specificity_at_sensitivity_0.90")
    validation.to_csv(OUT/"v7_corrected_threshold_policy_validation_summary.csv",index=False)
    rows=[]
    for policy in sorted(expected):
        threshold=policies[policies.policy.eq(policy)][["seed","outer_fold","threshold"]]
        data=pred.drop(columns="threshold").merge(threshold,on=["seed","outer_fold"],
            validate="many_to_one")
        if len(data)!=len(pred):raise RuntimeError("Missing fold threshold for outer prediction")
        for seed,part in data.groupby("seed"):
            speaker=part.groupby("physical_speaker_id",as_index=False).agg(
                label=("label","first"),probability=("probability","mean"),
                threshold=("threshold","first"))
            if len(speaker)!=15:raise RuntimeError("Missing held-out TORGO speaker")
            for level,frame in (("utterance",part),("physical_speaker",speaker)):
                rows.append(dict(policy=policy,seed=seed,level=level,
                    primary_policy=policy=="Specificity_at_sensitivity_0.90",
                    source_prediction_file="v7_pooled_oof_torgo_predictions.csv",
                    source_threshold_file="torgo_grouped_validation_threshold_policies.csv",
                    **_variable_threshold_metrics(frame)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_corrected_threshold_policy_outer_metrics.csv",index=False)
    summary=[]
    for (policy,level),part in result.groupby(["policy","level"]):
        row=dict(policy=policy,level=level,n_seeds=len(part),
            primary_policy=policy=="Specificity_at_sensitivity_0.90")
        for metric in METRIC_COLUMNS:
            row[metric+"_mean"]=part[metric].mean()
            row[metric+"_sd"]=part[metric].std(ddof=1)
        summary.append(row)
    pd.DataFrame(summary).to_csv(OUT/"v7_corrected_threshold_policy_outer_summary.csv",index=False)
    return validation,result


def final_research_audit():
    """Report achieved V7 evidence only; missing experiments remain NO."""
    meta, _, _ = load_torgo()
    meta, mapping = audit_physical_speakers(meta,save=False)
    mapping_ok = len(mapping)==30 and meta.physical_speaker_id.nunique()==15
    leakage_ok = True
    for seed in SEEDS:
        for fold,tr,te in grouped_splits(meta,seed):
            local=meta.iloc[tr].reset_index(drop=True)
            for _,tri,vai in grouped_splits(local,seed+731,n_splits=3):
                try:
                    assert_disjoint(meta,tr[tri],tr[vai],te)
                except AssertionError:
                    leakage_ok=False
    classical_names=("svm_rbf","random_forest","stacking_ensemble")
    classical_done=all((OUT/f"v7_corrected_classical_seed{seed}_fold{fold}_{name}.csv").exists()
        for seed in SEEDS for fold in range(5) for name in classical_names)
    classical_done=classical_done and all((OUT/name).exists() for name in (
        "v7_corrected_classical_predictions.csv","v7_corrected_classical_metrics.csv"))
    if classical_done:
        classical_metric=pd.read_csv(OUT/"v7_corrected_classical_metrics.csv")
        classical_done=(len(classical_metric)==30 and
            set(classical_metric.seed)==set(SEEDS))
    arch_file=OUT/"v7_corrected_architecture_ablation.csv"
    arch_names={"Stream 1 Only (CNN-BiLSTM)","Stream 2 Only (Transformer)",
        "Dual-Stream (No Pretraining)","Dual-Stream (Full)"}
    arch_done=False
    if arch_file.exists():
        arch=pd.read_csv(arch_file)
        arch_done=all(((arch.seed.eq(seed)) & (arch.model.eq(name)) &
            (arch.level.eq("physical_speaker"))).sum()==1
            for seed in SEEDS for name in arch_names)
        arch_done=arch_done and (OUT/"v7_corrected_architecture_predictions.csv").exists()
    convergence_file=OUT/"v7_convergence_summary.csv"
    convergence_done=False
    if convergence_file.exists():
        cv=pd.read_csv(convergence_file)
        convergence_done=(len(cv)==25 and cv[["epochs_run","best_epoch","best_val_metric",
            "hit_max_epoch","early_stopped","converged"]].notna().all().all())
    convergence_done=convergence_done and (OUT/"v7_training_budget_decision.json").exists()
    split_done=all((OUT/name).exists() for name in (
        "v7_split_protocol_comparison.csv","v7_split_protocol_per_seed_metrics.csv",
        "v7_split_protocol_paper_table.csv"))
    if split_done:
        split_rows=pd.read_csv(OUT/"v7_split_protocol_per_seed_metrics.csv")
        split_done=len(split_rows)==10 and set(split_rows.seed)==set(SEEDS)
    pooled_done=all((OUT/name).exists() for name in (
        "v7_pooled_oof_torgo_predictions.csv","v7_pooled_oof_torgo_speakers.csv",
        "v7_pooled_oof_speaker_metrics.csv","v7_pooled_oof_speaker_cluster_ci.csv",
        "v7_pooled_oof_five_seed_speaker_ci.csv"))
    if pooled_done:
        pooled_speakers=pd.read_csv(OUT/"v7_pooled_oof_torgo_speakers.csv")
        pooled_done=(len(pooled_speakers)==75 and
            pooled_speakers.groupby("seed").physical_speaker_id.nunique().eq(15).all())
    official_required=("v7_final_official_uaspeech_metrics.csv",
        "v7_final_official_uaspeech_predictions.csv",
        "v7_final_official_uaspeech_speakers.csv",
        "v7_final_official_speaker_cluster_ci.csv",
        "v7_final_official_zero_shot_audit.json","v7_final_model_lock.json")
    official_done=all((OUT/name).exists() for name in official_required)
    if official_done:
        ext=pd.read_csv(OUT/"v7_final_official_uaspeech_metrics.csv")
        official_done=(len(ext)==10 and set(ext.seed)==set(SEEDS) and
            set(ext.level)=={"utterance","physical_speaker"})
    new_external_audit=OUT/"v7_final_official_zero_shot_audit.json"
    historical_external_audit=OUT/"official_zero_shot_audit.json"
    if new_external_audit.exists():
        external_flags=json.loads(new_external_audit.read_text())
        official_usage={
            "training":"official_uaspeech_used_for_training",
            "fine-tuning":"official_uaspeech_used_for_fine_tuning",
            "threshold selection":"official_uaspeech_used_for_threshold_selection",
            "calibration":"official_uaspeech_used_for_calibration",
            "model selection":"official_uaspeech_used_for_model_selection"}
    else:
        external_flags=json.loads(historical_external_audit.read_text()) if historical_external_audit.exists() else {}
        official_usage={
            "training":"training_on_official_uaspeech",
            "fine-tuning":"fine_tuning_on_official_uaspeech",
            "threshold selection":"threshold_selection_on_official_uaspeech",
            "calibration":"calibration_on_official_uaspeech",
            "model selection":"model_selection_on_official_uaspeech"}
    subgroup_file=OUT/"v7_corrected_pooled_subgroup_metrics.csv"
    subgroup_ok=False
    if subgroup_file.exists():
        sub=pd.read_csv(subgroup_file)
        subgroup_ok=not sub.empty and sub.loc[sub.estimable.eq(False),"ROC_AUC"].isna().all()
    fold_subgroup_file=OUT/"v7_corrected_fold_subgroup_metrics.csv"
    if subgroup_ok and fold_subgroup_file.exists():
        fold_sub=pd.read_csv(fold_subgroup_file)
        subgroup_ok=fold_sub.loc[fold_sub.estimable.eq(False),"ROC_AUC"].isna().all()
    figure_index=OUT/"v7_corrected_publication_figure_index.csv"
    required_figures={
        "v7_corrected_baseline_comparison","v7_corrected_architecture_ablation",
        "v7_corrected_split_protocol","v7_corrected_torgo_pooled_roc",
        "v7_corrected_torgo_pooled_pr","v7_corrected_official_roc",
        "v7_corrected_official_pr","v7_corrected_official_confusion_matrix",
        "v7_corrected_torgo_calibration","v7_corrected_official_speaker_probabilities",
        "v7_corrected_official_speaker_roc_ci","v7_corrected_seed_stability",
        "v7_corrected_generalization_gap","v7_corrected_official_channels",
        "v7_corrected_torgo_cross_microphone","v7_corrected_torgo_noise"}
    figures_done=figure_index.exists() and required_figures.issubset(
        set(pd.read_csv(figure_index).figure))
    status={
        "Physical-speaker mapping verified":"PASS" if mapping_ok else "FAIL",
        "Physical-speaker leakage":"PASS" if leakage_ok else "FAIL",
        "Classical baselines rerun on identical physical-speaker folds":"YES" if classical_done else "NO",
        "Neural architecture ablation rerun on identical physical-speaker folds":"YES" if arch_done else "NO",
        "Five seeds completed":"YES" if classical_done and arch_done else "NO",
        "Training convergence verified":"YES" if convergence_done else "NO",
        "Controlled session-vs-physical-speaker experiment completed":"YES" if split_done else "NO",
        "Pooled out-of-fold TORGO speaker metrics completed":"YES" if pooled_done else "NO",
        "Official UASpeech zero-shot five-seed evaluation completed":"YES" if official_done else "NO",
        **{f"Official UASpeech used for {name}":
            "NO" if external_flags.get(flag) is False else "UNVERIFIED"
            for name,flag in official_usage.items()},
        "Calibration evaluated":"YES" if (OUT/"torgo_outer_calibration_comparison.csv").exists() else "NO",
        "Cross-microphone evaluation":"YES" if (OUT/"torgo_cross_microphone_results.csv").exists() else "NO",
        "Noise robustness evaluation":"YES" if (OUT/"torgo_noise_results.csv").exists() else "NO",
        "Subgroup estimability audit":"PASS" if subgroup_ok else "FAIL",
        "Publication figures regenerated":"YES" if figures_done else "NO"}
    limitations=[f"{key}: {value}" for key,value in status.items()
        if (value in ("NO","FAIL") and not key.startswith("Official UASpeech used for"))
        or value=="UNVERIFIED"]
    inner_index=OUT/"v7_corrected_classical_inner_oof_index.csv"
    if classical_done and (not inner_index.exists() or len(pd.read_csv(inner_index))!=75):
        n_saved=len(list(OUT.glob("v7_corrected_classical_seed*_fold*.inner_oof.csv")))
        limitations.append(f"Classical grouped validation score provenance is being backfilled: {n_saved}/75 model-fold files saved.")
    if arch_done:
        ranking_audit=OUT/"v7_corrected_architecture_ranking_audit.json"
        if ranking_audit.exists():
            rank=json.loads(ranking_audit.read_text())
            if rank.get("winner_varies_across_seeds"):
                limitations.append("The highest-ranked neural architecture varies across the five TORGO seeds; no single-seed winner is treated as definitive.")
        pretraining_histories=list(OUT.glob("v7_arch_dual_full_seed*_fold*_final_pretraining_history.csv"))
        if pretraining_histories:
            pretrain_epochs=max(int(pd.read_csv(path).epoch.max()) for path in pretraining_histories)
            supervised_epochs=int(json.loads((OUT/"v7_training_budget_decision.json").read_text())["selected_max_epochs"])
            limitations.append(f"Dual-Stream (Full) used {pretrain_epochs} training-speaker-only pretraining epochs in addition to the matched {supervised_epochs}-epoch supervised cap; total optimization steps therefore differ by the pretraining treatment.")
    if pooled_done:
        pooled=pd.read_csv(OUT/"v7_pooled_oof_speaker_metrics.csv")
        sp=pooled[pooled.level.eq("physical_speaker")]
        limitations.append(f"TORGO pooled speaker ROC-AUC across seeds: {sp.ROC_AUC.mean():.3f} ± {sp.ROC_AUC.std(ddof=1):.3f} (15 speakers per seed).")
        limitations.append("Five seed estimates reuse the same 15 TORGO speakers; seed SD measures run variation, while speaker bootstrap reflects subject sampling uncertainty.")
        limitations.append(f"The prespecified sensitivity-constrained TORGO threshold yields pooled speaker specificity {sp.Specificity.mean():.3f} across seeds.")
    threshold_summary=OUT/"v7_corrected_threshold_policy_outer_summary.csv"
    if threshold_summary.exists():
        policies=pd.read_csv(threshold_summary)
        speaker_policies=policies[policies.level.eq("physical_speaker")]
        if not speaker_policies.empty:
            low=speaker_policies.Specificity_mean.min()
            high=speaker_policies.Specificity_mean.max()
            limitations.append(f"TORGO-only prespecified threshold policies yield held-out speaker specificity from {low:.3f} to {high:.3f}; the external threshold was never adjusted.")
    if (OUT/"official_per_seed_metrics.csv").exists() and not official_done:
        limitations.append("Existing official UASpeech five-seed outputs predate the requested corrected-model freeze; they remain historical V7 evidence.")
        historical_external=pd.read_csv(OUT/"official_per_seed_metrics.csv")
        historical_external=historical_external[historical_external.level.eq("physical_speaker")]
        if not historical_external.empty:
            limitations.append(f"Historical V7 official UASpeech speaker specificity at TORGO-locked thresholds: {historical_external.Specificity.mean():.3f} across seeds; no external threshold adjustment was made.")
    if official_done:
        final_external=pd.read_csv(OUT/"v7_final_official_uaspeech_metrics.csv")
        final_external=final_external[final_external.level.eq("physical_speaker")]
        if (OUT/"official_per_seed_metrics.csv").exists():
            limitations.append("Official UASpeech had already been evaluated in historical V7; this is a provenance-checked rerun of the same frozen model, not a new prospective external cohort.")
        limitations.append(f"Frozen official UASpeech speaker specificity at TORGO-locked thresholds: {final_external.Specificity.mean():.3f} across seeds; no external threshold adjustment was made.")
        limitations.append(f"Official five-seed estimates reuse the same {int(final_external.n.iloc[0])} physical speakers; the cluster bootstrap resamples speakers jointly across seeds.")
        if new_external_audit.exists():
            channel_speakers=int(external_flags.get("matched_physical_speakers",0))
            if channel_speakers<int(final_external.n.iloc[0]):
                limitations.append(f"Matched M2–M8 channel diagnostics include {channel_speakers} of {int(final_external.n.iloc[0])} primary M3 speakers; channel curves use this common-content subset.")
        limitations.append(f"Official speaker ROC-AUC varies by seed from {final_external.ROC_AUC.min():.3f} to {final_external.ROC_AUC.max():.3f} (SD {final_external.ROC_AUC.std(ddof=1):.3f}).")
        undefined={metric:int(final_external[metric].isna().sum()) for metric in
            ("Sensitivity","Specificity","Precision","NPV","ROC_AUC","PR_AUC")
            if final_external[metric].isna().any()}
        if undefined:
            limitations.append(f"Undefined official speaker metrics from empty predicted or diagnosis classes are retained as missing values: {undefined}.")
    if (OUT/"torgo_noise_results.csv").exists():
        noise=pd.read_csv(OUT/"torgo_noise_results.csv")
        low=noise[noise.SNR_dB.eq(0)]
        if not low.empty:
            limitations.append(f"Existing 0 dB TORGO noise Balanced Accuracy: {low.Balanced_Accuracy.mean():.3f} (seed 42 diagnostic).")
    valid_subgroup=OUT/"v7_corrected_valid_fold_subgroup_summary.csv"
    if valid_subgroup.exists() and fold_subgroup_file.exists():
        sub=pd.read_csv(valid_subgroup)
        limitations.append("Gender subgroup fold estimates use only folds containing both diagnoses; " +
            ", ".join(f"{row.subgroup}: {row.n_valid_folds} valid folds" for row in sub.itertuples()) +
            ". Diagnosis-specific groups have one class and no estimable ROC-AUC or fairness conclusion.")
    if (OUT/"torgo_cross_microphone_results.csv").exists() and (OUT/"torgo_noise_results.csv").exists():
        mic_seeds=set(pd.read_csv(OUT/"torgo_cross_microphone_results.csv").seed)
        noise_seeds=set(pd.read_csv(OUT/"torgo_noise_results.csv").seed)
        if mic_seeds!=set(SEEDS) or noise_seeds!=set(SEEDS):
            limitations.append(f"TORGO cross-microphone and noise diagnostics cover seeds {sorted(mic_seeds)} and {sorted(noise_seeds)}, respectively; they are diagnostic rather than five-seed model-selection evidence.")
    classical_cal=OUT/"v7_corrected_classical_calibration_comparison.csv"
    if classical_cal.exists():
        cc=pd.read_csv(classical_cal)
        rf=cc[(cc.model.eq("Random Forest")) & (cc.level.eq("physical_speaker"))]
        if set(rf.variant)=={"Uncalibrated","Temperature scaled"}:
            raw=rf[rf.variant.eq("Uncalibrated")].Brier.mean()
            scaled=rf[rf.variant.eq("Temperature scaled")].Brier.mean()
            if scaled>raw:
                limitations.append(f"Classical Random Forest TORGO speaker Brier worsened after temperature scaling ({raw:.3f} to {scaled:.3f}); both variants are retained for audit.")
    if convergence_file.exists():
        cv=pd.read_csv(convergence_file)
        if cv.epochs_run.notna().any():
            observed=cv[cv.epochs_run.notna()]
            limitations.append(f"Convergence replay: {int(observed.hit_max_epoch.sum())}/{len(observed)} completed outer folds hit 20 epochs; {int((~observed.converged.astype(bool)).sum())}/{len(observed)} fail the stability-and-replay criterion.")
    result={"status":status,"remaining_limitations":limitations,
        "source_files":{
            "pooled_predictions":"v7_pooled_oof_torgo_predictions.csv",
            "pooled_speakers":"v7_pooled_oof_torgo_speakers.csv",
            "final_official":"v7_final_official_uaspeech_predictions.csv" if official_done else None,
            "historical_official":"official_per_seed_metrics.csv"}}
    (OUT/"v7_final_research_audit.json").write_text(json.dumps(result,indent=2)+"\n")
    print("V7 FINAL RESEARCH AUDIT\n=======================\n")
    for key,value in status.items():print(f"{key}: {value}")
    print("\nREMAINING LIMITATIONS")
    for item in limitations:print("-",item)
    return result


def make_corrected_figures():
    """Generate only figures whose corrected prediction provenance is complete."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    target=OUT/"v7_corrected_publication_figures"
    target.mkdir(exist_ok=True)
    index=[]
    def save(fig,name,source,description):
        fig.tight_layout()
        fig.savefig(target/f"{name}.png",dpi=300,bbox_inches="tight")
        fig.savefig(target/f"{name}.pdf",bbox_inches="tight")
        plt.close(fig)
        index.append(dict(figure=name,source_file=source,description=description,
            png=str(target/f"{name}.png"),pdf=str(target/f"{name}.pdf")))
    path=OUT/"v7_pooled_oof_torgo_predictions.csv"
    if path.exists():
        pred=pd.read_csv(path)
        for level,frame,source in (("physical speaker",
                pd.read_csv(OUT/"v7_pooled_oof_torgo_speakers.csv"),
                "v7_pooled_oof_torgo_speakers.csv"),
                ("utterance",pred,path.name)):
            for kind in ("roc","pr"):
                fig,ax=plt.subplots(figsize=(6,5))
                for seed,part in frame.groupby("seed"):
                    y,p=part.label.to_numpy(int),part.probability.to_numpy(float)
                    if kind=="roc":
                        a,b,_=roc_curve(y,p)
                        ax.set(xlabel="False positive rate",ylabel="True positive rate",
                            title=f"TORGO pooled {level} ROC — held-out speakers")
                    else:
                        b,a,_=precision_recall_curve(y,p)
                        ax.set(xlabel="Recall",ylabel="Precision",
                            title=f"TORGO pooled {level} PR — held-out speakers")
                    ax.plot(a,b,alpha=.6,label=f"Seed {seed}")
                if kind=="roc":ax.plot([0,1],[0,1],"--",color="gray")
                else:ax.axhline(frame[frame.seed.eq(42)].label.mean(),ls="--",color="gray")
                ax.set(xlim=(0,1),ylim=(0,1));ax.legend(fontsize=8)
                suffix="" if level=="physical speaker" else "_utterance"
                save(fig,f"v7_corrected_torgo_pooled_{kind}{suffix}",source,
                    f"Five-seed pooled outer-fold {level} curves")
        fig,ax=plt.subplots(figsize=(6,5))
        seed42=pred[pred.seed.eq(42)]
        for col,label in (("probability_raw","Uncalibrated"),("probability","TORGO temperature scaled")):
            p=seed42[col].to_numpy(float);y=seed42.label.to_numpy(int)
            bins=np.linspace(0,1,11)
            ids=np.minimum(np.digitize(p,bins)-1,9)
            obs=[];means=[]
            for b in range(10):
                mask=ids==b
                if mask.any():obs.append(y[mask].mean());means.append(p[mask].mean())
            ax.plot(means,obs,"o-",label=label)
        ax.plot([0,1],[0,1],"--",color="gray",label="Perfect calibration")
        ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean probability",ylabel="Observed diagnosis rate",
            title="TORGO pooled outer-test calibration (seed 42)");ax.legend(fontsize=8)
        save(fig,"v7_corrected_torgo_reliability_seed42",path.name,
            "Raw and TORGO-calibrated outer-test predictions, seed 42")
        cal_path=OUT/"v7_corrected_pooled_calibration_comparison.csv"
        if cal_path.exists():
            cal=pd.read_csv(cal_path)
            cal=cal[cal.level.eq("physical_speaker")]
            fig,ax=plt.subplots(figsize=(7,5))
            x=np.arange(2);width=.35
            for j,(variant,color) in enumerate((("Uncalibrated","#65717C"),
                    ("Temperature scaled","#245A81"))):
                part=cal[cal.variant.eq(variant)]
                mean=[part[m].mean() for m in ("Brier","ECE")]
                sd=[part[m].std(ddof=1) for m in ("Brier","ECE")]
                ax.bar(x+(j-.5)*width,mean,width,yerr=sd,capsize=3,
                    label=variant,color=color)
            ax.set(xticks=x,xticklabels=["Brier","ECE"],ylim=(0,0.5),
                ylabel="Five-seed mean ± SD (lower is better)",
                title="TORGO pooled speaker calibration comparison")
            ax.legend(fontsize=8)
            save(fig,"v7_corrected_torgo_calibration",cal_path.name,
                "Raw versus TORGO temperature-scaled outer-test speaker calibration")
    baseline=OUT/"v7_corrected_baseline_results.csv"
    if baseline.exists():
        df=pd.read_csv(baseline)
        fig,ax=plt.subplots(figsize=(9,5))
        ordered=df.sort_values("ROC_AUC_mean")
        ax.barh(ordered.model,ordered.ROC_AUC_mean,xerr=ordered.ROC_AUC_sd,
            color="#245A81",capsize=3)
        ax.set(xlim=(0,1),xlabel="Physical-speaker-level pooled ROC-AUC",
            title="Corrected V7 baseline comparison, five seeds")
        save(fig,"v7_corrected_baseline_comparison",baseline.name,
            "Five-seed mean and SD from physical-speaker-safe pooled predictions")
    arch_path=OUT/"v7_corrected_architecture_ablation.csv"
    if arch_path.exists() and baseline.exists():
        arch=pd.read_csv(arch_path)
        arch=arch[arch.level.eq("physical_speaker")]
        names=list(arch.model.drop_duplicates())
        fig,ax=plt.subplots(figsize=(10,5))
        x=np.arange(len(names))
        for seed,part in arch.groupby("seed"):
            values=part.set_index("model").reindex(names).ROC_AUC.to_numpy()
            ax.plot(x,values,"o-",alpha=.55,label=f"Seed {seed}")
        ax.set(xticks=x,xticklabels=[n.replace(" (","\n(") for n in names],
            ylim=(0,1),ylabel="Pooled speaker ROC-AUC",
            title="Matched V7 architecture ablation across five seeds")
        ax.legend(ncol=5,fontsize=8)
        save(fig,"v7_corrected_architecture_ablation",arch_path.name,
            "All five seedwise physical-speaker-safe architecture results")
    interim=OUT/"v7_interim_physical_speaker_baseline_results.csv"
    if interim.exists() and not baseline.exists():
        df=pd.read_csv(interim).sort_values("ROC_AUC_mean")
        fig,ax=plt.subplots(figsize=(9,5))
        ax.barh(df.Model,df.ROC_AUC_mean,xerr=df.ROC_AUC_sd,
            color="#245A81",capsize=3)
        ax.set(xlim=(0,1),xlabel="Pooled physical-speaker ROC-AUC",
            title="Interim V7 comparison — neural ablation pending")
        save(fig,"v7_interim_baseline_comparison",interim.name,
            "Four-model interim comparison; excluded from final figure gate")
    split_path=OUT/"v7_split_protocol_comparison.csv"
    if split_path.exists():
        data=pd.read_csv(split_path).iloc[:2]
        fig,ax=plt.subplots(figsize=(8,5))
        cols=["Accuracy","Balanced_Accuracy","F1_dysarthric","ROC_AUC","PR_AUC"]
        x=np.arange(len(cols));width=.36
        for j,row in enumerate(data.itertuples(index=False)):
            values=[getattr(row,c) for c in cols]
            ax.bar(x+(j-.5)*width,values,width,
                label="Session diagnostic" if j==0 else "Physical speaker primary")
        ax.set(xticks=x,xticklabels=["Accuracy","Balanced\nAccuracy","F1","ROC-AUC","PR-AUC"],
            ylim=(0,1),ylabel="Five-seed mean",
            title="Controlled TORGO split-protocol comparison")
        ax.legend(fontsize=8)
        save(fig,"v7_corrected_split_protocol",split_path.name,
            "Same Stream-1 recipe; grouping key is the only experimental change")
    threshold_path=OUT/"torgo_grouped_validation_threshold_policies.csv"
    outer_threshold_path=OUT/"v7_corrected_threshold_policy_outer_summary.csv"
    if threshold_path.exists() and outer_threshold_path.exists():
        selected=pd.read_csv(threshold_path)
        outer=pd.read_csv(outer_threshold_path)
        outer=outer[outer.level.eq("physical_speaker")]
        names=["F1_optimal","Youden_J","Balanced_Accuracy_optimal",
            "Specificity_at_sensitivity_0.90"]
        fig,axes=plt.subplots(1,2,figsize=(11,4))
        axes[0].boxplot([selected[selected.policy.eq(name)].threshold.to_numpy()
            for name in names],tick_labels=["F1","Youden J","Balanced acc.","Sensitivity≥0.90"])
        axes[0].set(ylabel="TORGO grouped-validation threshold",
            title="TORGO validation thresholds across 25 outer folds")
        x=np.arange(len(names));width=.35
        for j,(metric,label) in enumerate((("Sensitivity","Sensitivity"),
                ("Specificity","Specificity"))):
            part=outer.set_index("policy").reindex(names)
            axes[1].bar(x+(j-.5)*width,part[metric+"_mean"],width,
                yerr=part[metric+"_sd"],capsize=2,label=label)
        axes[1].set(xticks=x,xticklabels=["F1","Youden J","Balanced acc.","Sensitivity≥0.90"],
            ylim=(0,1),ylabel="Pooled held-out speaker metric",
            title="TORGO-only policy sensitivity")
        axes[1].legend(fontsize=8)
        save(fig,"v7_corrected_threshold_policy_sensitivity",
            f"{threshold_path.name}; {outer_threshold_path.name}",
            "Diagnostic comparison; sensitivity-constrained policy remains prespecified")
    official_path=OUT/"v7_final_official_uaspeech_predictions.csv"
    official_metrics_path=OUT/"v7_final_official_uaspeech_metrics.csv"
    if official_path.exists() and official_metrics_path.exists():
        from sklearn.metrics import confusion_matrix
        official=pd.read_csv(official_path,low_memory=False)
        official=official[official.microphone.eq("M3")]
        ext_metrics=pd.read_csv(official_metrics_path)
        if set(ext_metrics.seed)!=set(SEEDS):
            raise RuntimeError("Final official figure source lacks five seeds")
        for kind in ("roc","pr"):
            fig,ax=plt.subplots(figsize=(6,5))
            for seed in SEEDS:
                y=official.label.to_numpy(int)
                p=official[f"prob_seed{seed}"].to_numpy(float)
                if kind=="roc":
                    x,z,_=roc_curve(y,p)
                else:
                    z,x,_=precision_recall_curve(y,p)
                ax.plot(x,z,alpha=.6,label=f"Seed {seed}")
            if kind=="roc":ax.plot([0,1],[0,1],"--",color="gray")
            else:ax.axhline(official.label.mean(),ls="--",color="gray")
            ax.set(xlim=(0,1),ylim=(0,1),
                xlabel="False positive rate" if kind=="roc" else "Recall",
                ylabel="True positive rate" if kind=="roc" else "Precision",
                title=f"Frozen V7 official UASpeech M3 {kind.upper()}")
            ax.legend(fontsize=8)
            save(fig,f"v7_corrected_official_{kind}",official_path.name,
                "Fresh official M3 inference from five TORGO-frozen deployment fits")
        seed=42
        p=official[f"prob_seed{seed}"].to_numpy(float)
        y=official.label.to_numpy(int)
        threshold=float(ext_metrics[(ext_metrics.seed.eq(seed)) &
            (ext_metrics.level.eq("utterance"))].threshold.iloc[0])
        cm=confusion_matrix(y,p>=threshold,labels=[0,1])
        fig,ax=plt.subplots(figsize=(5,5))
        ax.imshow(cm,cmap="Blues")
        for i in range(2):
            for j in range(2):ax.text(j,i,f"{cm[i,j]:,}",ha="center",va="center")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Control","Dysarthric"],
            yticklabels=["Control","Dysarthric"],xlabel="Predicted",ylabel="Actual",
            title="Official UASpeech M3, seed 42, TORGO-locked threshold")
        save(fig,"v7_corrected_official_confusion_matrix",official_path.name,
            "Prespecified seed 42 and TORGO-locked threshold")
        speaker_path=OUT/"v7_final_official_uaspeech_speakers.csv"
        ci_path=OUT/"v7_final_official_speaker_cluster_ci.csv"
        if speaker_path.exists() and ci_path.exists():
            speakers=pd.read_csv(speaker_path)
            sp42=speakers[speakers.seed.eq(42)].sort_values("probability")
            fig,ax=plt.subplots(figsize=(11,5))
            x=np.arange(len(sp42))
            ax.bar(x,sp42.probability,color=np.where(sp42.label.eq(1),"#C45C47","#245A81"))
            ax.axhline(threshold,color="black",ls="--",label="TORGO-locked threshold")
            ax.set(xticks=x,xticklabels=sp42.physical_speaker_id,ylim=(0,1),
                ylabel="Mean dysarthria probability",
                title="Official UASpeech physical speakers, seed 42")
            ax.tick_params(axis="x",rotation=70);ax.legend(fontsize=8)
            save(fig,"v7_corrected_official_speaker_probabilities",speaker_path.name,
                "Official speaker-level probabilities with locked operating point")
            ci=pd.read_csv(ci_path)
            auc=ci[(ci.seed.astype(str).eq("five_seed_mean")) &
                (ci.metric.eq("ROC_AUC"))].iloc[0]
            fig,ax=plt.subplots(figsize=(6,5))
            for seed,part in speakers.groupby("seed"):
                a,b,_=roc_curve(part.label,part.probability)
                ax.plot(a,b,alpha=.6,label=f"Seed {seed}")
            ax.plot([0,1],[0,1],"--",color="gray")
            ax.set(xlim=(0,1),ylim=(0,1),xlabel="False positive rate",
                ylabel="True positive rate",
                title=f"Official speaker ROC: mean AUC {auc.point:.3f}\n"
                      f"95% speaker bootstrap CI {auc.ci_low:.3f}–{auc.ci_high:.3f}")
            ax.legend(fontsize=8)
            save(fig,"v7_corrected_official_speaker_roc_ci",
                f"{speaker_path.name}; {ci_path.name}",
                "Five-seed speaker ROC and paired physical-speaker bootstrap CI")
        pooled_metrics=OUT/"v7_pooled_oof_speaker_metrics.csv"
        if pooled_metrics.exists():
            internal=pd.read_csv(pooled_metrics)
            internal=internal[internal.level.eq("physical_speaker")].set_index("seed")
            external=ext_metrics[ext_metrics.level.eq("physical_speaker")].set_index("seed")
            fig,ax=plt.subplots(figsize=(7,5))
            for seed in SEEDS:
                ax.plot([0,1],[internal.loc[seed,"ROC_AUC"],
                    external.loc[seed,"ROC_AUC"]],"o-",alpha=.6,label=f"Seed {seed}")
            ax.set(xticks=[0,1],xticklabels=["TORGO pooled OOF","Official UASpeech"],
                ylim=(0,1),ylabel="Speaker ROC-AUC",
                title="V7 internal and external five-seed stability")
            ax.legend(fontsize=8)
            save(fig,"v7_corrected_seed_stability",
                f"{pooled_metrics.name}; {official_metrics_path.name}",
                "Descriptive internal/external comparison; different frozen fits")
            fig,ax=plt.subplots(figsize=(8,5))
            x=np.arange(len(SEEDS));width=.25
            for j,metric in enumerate(("ROC_AUC","Balanced_Accuracy","F1_dysarthric")):
                gap=(internal.loc[list(SEEDS),metric]-external.loc[list(SEEDS),metric]).to_numpy()
                ax.bar(x+(j-1)*width,gap,width,label=metric.replace("_"," "))
            ax.axhline(0,color="gray",lw=1)
            ax.set(xticks=x,xticklabels=SEEDS,xlabel="Seed",
                ylabel="TORGO pooled OOF minus official UASpeech",
                title="Descriptive generalization gap (different frozen fits)")
            ax.legend(fontsize=8)
            save(fig,"v7_corrected_generalization_gap",
                f"{pooled_metrics.name}; {official_metrics_path.name}",
                "Descriptive gap; held-out outer fits versus all-TORGO deployment fits")
        channel_path=OUT/"v7_final_official_matched_channel_metrics.csv"
        if channel_path.exists():
            channels=pd.read_csv(channel_path)
            channels=channels[channels.level.eq("physical_speaker")]
            fig,ax=plt.subplots(figsize=(8,5))
            names=[f"M{i}" for i in range(2,9)]
            x=np.arange(len(names));width=.25
            for j,metric in enumerate(("ROC_AUC","PR_AUC","Balanced_Accuracy")):
                mean=channels.groupby("channel")[metric].mean().reindex(names)
                sd=channels.groupby("channel")[metric].std().reindex(names)
                ax.bar(x+(j-1)*width,mean,width,yerr=sd,capsize=2,
                    label=metric.replace("_"," "))
            ax.set(xticks=x,xticklabels=names,ylim=(0,1),xlabel="Official channel",
                ylabel="Five-seed mean ± SD",
                title="M2–M8 matched speaker/content robustness")
            ax.legend(fontsize=8)
            save(fig,"v7_corrected_official_channels",channel_path.name,
                "Official matched-content, physical-speaker channel evaluation")
    mic_path=OUT/"torgo_cross_microphone_results.csv"
    if mic_path.exists():
        mic=pd.read_csv(mic_path)
        matrix=mic.pivot_table(index="train_mic",columns="test_mic",
            values="Balanced_Accuracy",aggfunc="mean").reindex(
            index=["head","array"],columns=["head","array"])
        fig,ax=plt.subplots(figsize=(5,4))
        ax.imshow(matrix,vmin=0,vmax=1,cmap="Blues")
        for i in range(2):
            for j in range(2):ax.text(j,i,f"{matrix.iloc[i,j]:.3f}",ha="center",va="center")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Head","Array"],
            yticklabels=["Head","Array"],xlabel="Test microphone",
            ylabel="Train/validation microphone",title="TORGO cross-microphone diagnostic")
        save(fig,"v7_corrected_torgo_cross_microphone",mic_path.name,
            "Physical-speaker-safe TORGO cross-microphone diagnostic")
    noise_path=OUT/"torgo_noise_results.csv"
    if noise_path.exists():
        noise=pd.read_csv(noise_path)
        fig,ax=plt.subplots(figsize=(7,5))
        for metric in ("ROC_AUC","Balanced_Accuracy","F1_dysarthric"):
            line=noise.groupby("SNR_dB")[metric].mean().reindex([30,20,10,5,0])
            ax.plot(line.index,line.values,"o-",label=metric.replace("_"," "))
        ax.invert_xaxis()
        ax.set(ylim=(0,1),xlabel="SNR (dB)",ylabel="Metric",
            title="TORGO held-out speaker noise diagnostic")
        ax.legend(fontsize=8)
        save(fig,"v7_corrected_torgo_noise",noise_path.name,
            "Physical-speaker-safe TORGO additive-noise diagnostic")
    curves=list(OUT.glob("v7_corrected_convergence_seed*_outer*_inner*.csv"))
    completed_curve_names=set()
    inner_summary=OUT/"v7_convergence_inner_summary.csv"
    if inner_summary.exists():
        completed_curve_names=set(pd.read_csv(inner_summary).curve_file)
    curves=[p for p in curves if p.name in completed_curve_names]
    if curves:
        representative=[]
        for seed in SEEDS:
            eligible=sorted(p for p in curves if f"seed{seed}_" in p.name)
            if eligible:representative.append(eligible[0])
        for path in representative:
            h=pd.read_csv(path)
            fig,ax=plt.subplots(1,2,figsize=(10,4))
            ax[0].plot(h.epoch,h.train_loss,label="Training loss")
            ax[0].plot(h.epoch,h.val_loss,label="Grouped validation loss")
            ax[0].set(xlabel="Epoch",ylabel="Loss",title=path.stem)
            ax[0].legend(fontsize=8)
            ax[1].plot(h.epoch,h.val_roc_auc,label="Validation ROC-AUC")
            ax[1].plot(h.epoch,h.val_f1_at_0_5,label="Validation F1 at 0.5")
            ax[1].set(xlabel="Epoch",ylabel="Metric",ylim=(0,1))
            ax[1].legend(fontsize=8)
            save(fig,path.stem,path.name,"TORGO grouped inner-fold convergence replay")
    summary=OUT/"v7_convergence_summary.csv"
    if summary.exists():
        df=pd.read_csv(summary)
        if df.epochs_run.notna().all():
            fig,ax=plt.subplots(figsize=(8,4))
            x=np.arange(len(df))
            ax.bar(x,df.epochs_run,color=np.where(df.hit_max_epoch,"#C45C47","#245A81"))
            ax.set(xlabel="Seed and outer fold",ylabel="Maximum inner epochs run",
                title="V7 grouped-validation stopping summary")
            ax.set_xticks(x);ax.set_xticklabels([f"{r.seed}/{r.fold}" for r in df.itertuples()],rotation=75,fontsize=7)
            save(fig,"v7_corrected_convergence_summary",summary.name,
                "Blue: early stop; red: maximum epoch")
    result=pd.DataFrame(index)
    result.to_csv(OUT/"v7_corrected_publication_figure_index.csv",index=False)
    return result


def _session_splits(meta,seed,n_splits=5):
    sessions=meta.groupby("speaker_id",as_index=False).agg(label=("label","first"))
    cv=StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    for fold,(tri,tei) in enumerate(cv.split(sessions,sessions.label,sessions.speaker_id)):
        train_sessions=set(sessions.iloc[tri].speaker_id)
        test_sessions=set(sessions.iloc[tei].speaker_id)
        assert train_sessions.isdisjoint(test_sessions)
        train=np.flatnonzero(meta.speaker_id.isin(train_sessions))
        test=np.flatnonzero(meta.speaker_id.isin(test_sessions))
        yield fold,train,test


def run_controlled_split_diagnostic():
    """Identical Stream-1 recipe with only the grouping partition changed.

    Existing physical-speaker V7 predictions are the fixed control. Session
    models are freshly fitted with the same 20 epochs, 512 examples per epoch,
    AdamW, augmentation, TORGO inner OOF temperature and threshold policy.
    """
    budget_path=OUT/"v7_training_budget_decision.json"
    if not budget_path.exists():
        raise RuntimeError("Complete TORGO-only convergence decision before split diagnostic")
    budget=json.loads(budget_path.read_text())
    if budget["selected_max_epochs"]!=20 or budget["samples_per_epoch"]!=512:
        raise RuntimeError("Existing physical-speaker control uses 20 epochs/512 samples; both arms require matching retraining")
    meta,features,_=load_torgo()
    meta,_=audit_physical_speakers(meta,save=False)
    results=[]
    physical=pd.read_csv(OUT/"v7_pooled_oof_torgo_predictions.csv")
    overlap_rows=[]
    for seed in SEEDS:
        session_parts=[]
        for fold,tr,te in _session_splits(meta,seed):
            train_physical=set(meta.iloc[tr].physical_speaker_id)
            test_physical=set(meta.iloc[te].physical_speaker_id)
            overlap_rows.append(dict(seed=seed,outer_fold=fold,
                n_train_sessions=meta.iloc[tr].speaker_id.nunique(),
                n_test_sessions=meta.iloc[te].speaker_id.nunique(),
                n_overlapping_physical_speakers=len(train_physical&test_physical)))
            path=OUT/f"v7_controlled_session_seed{seed}_fold{fold}.csv"
            if path.exists():
                frame=pd.read_csv(path)
                assert set(frame.file_path)==set(meta.iloc[te].file_path)
                session_parts.append(frame)
                continue
            local=meta.iloc[tr].reset_index(drop=True)
            oof=[];chosen=[]
            for inner,tri,vai in _session_splits(local,seed+731,n_splits=3):
                train,val=tr[tri],tr[vai]
                assert set(meta.iloc[train].speaker_id).isdisjoint(set(meta.iloc[val].speaker_id))
                assert set(meta.iloc[train].speaker_id).isdisjoint(set(meta.iloc[te].speaker_id))
                assert set(meta.iloc[val].speaker_id).isdisjoint(set(meta.iloc[te].speaker_id))
                model,best=train_one(meta,features,train,seed+inner,
                    f"controlled_session_seed{seed}_fold{fold}_inner{inner}",
                    epochs=20,val_indices=val,samples_per_epoch=512,
                    history_path=OUT/f"v7_controlled_session_seed{seed}_fold{fold}_inner{inner}_history.csv")
                raw=predict(model,meta,features,val)
                oof.append(pd.DataFrame(dict(index=val,
                    label=meta.iloc[val].label.to_numpy(int),
                    physical_speaker_id=meta.iloc[val].physical_speaker_id.to_numpy(),
                    probability_raw=raw)))
                chosen.append(best)
                del model
            inner_oof=pd.concat(oof,ignore_index=True)
            assert len(inner_oof)==len(tr) and inner_oof["index"].is_unique
            temp=calibrate_temperature(inner_oof.label,inner_oof.probability_raw,
                inner_oof.physical_speaker_id)
            inner_oof["probability"]=apply_temperature(inner_oof.probability_raw,temp)
            policies=threshold_table(inner_oof.label,inner_oof.probability,
                inner_oof.physical_speaker_id)
            threshold=float(policies.set_index("policy").loc[
                "Specificity_at_sensitivity_0.90","threshold"])
            final_epochs=int(np.median(chosen))
            model,_=train_one(meta,features,tr,seed+10000,
                f"controlled_session_seed{seed}_fold{fold}",epochs=final_epochs,
                samples_per_epoch=512)
            raw=predict(model,meta,features,te)
            frame=meta.iloc[te][["file_path","speaker_id","physical_speaker_id",
                "label","gender","mic_type"]].copy()
            frame["probability_raw"]=raw
            frame["probability"]=apply_temperature(raw,temp)
            frame["threshold"]=threshold
            frame["temperature"]=temp
            frame["seed"]=seed
            frame["outer_fold"]=fold
            frame.to_csv(path,index=False)
            inner_oof.to_csv(path.with_suffix(".inner_oof.csv"),index=False)
            policies.to_csv(path.with_suffix(".threshold_policies.csv"),index=False)
            torch.save(dict(seed=seed,fold=fold,protocol="session_disjoint_diagnostic",
                train_sessions=sorted(set(meta.iloc[tr].speaker_id)),
                test_sessions=sorted(set(meta.iloc[te].speaker_id)),
                train_physical_speakers=sorted(set(meta.iloc[tr].physical_speaker_id)),
                test_physical_speakers=sorted(set(meta.iloc[te].physical_speaker_id)),
                threshold=threshold,temperature=temp,selected_epochs=final_epochs,
                model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),
                path.with_suffix(".pt"))
            session_parts.append(frame)
            print("Controlled session diagnostic",seed,fold,flush=True)
        session=pd.concat(session_parts,ignore_index=True)
        assert session.file_path.is_unique and set(session.file_path)==set(meta.file_path)
        p=physical[physical.seed.eq(seed)]
        assert p.file_path.is_unique and set(p.file_path)==set(meta.file_path)
        for protocol,data in (("Historical-style session-disjoint diagnostic",session),
                              ("Physical-speaker-disjoint primary",p)):
            results.append(dict(seed=seed,protocol=protocol,
                **_variable_threshold_metrics(data)))
    pd.DataFrame(results).to_csv(OUT/"v7_split_protocol_per_seed_metrics.csv",index=False)
    pd.DataFrame(overlap_rows).to_csv(OUT/"v7_controlled_session_overlap_audit.csv",index=False)
    rows=[]
    per_seed=pd.DataFrame(results)
    for protocol,group in per_seed.groupby("protocol"):
        row={"Protocol":protocol}
        for col in ("Accuracy","Balanced_Accuracy","F1_dysarthric","ROC_AUC","PR_AUC"):
            row[col]=group[col].mean()
            row[col+"_SD"]=group[col].std(ddof=1)
        rows.append(row)
    a=per_seed[per_seed.protocol.str.startswith("Historical")].set_index("seed")
    b=per_seed[per_seed.protocol.str.startswith("Physical")].set_index("seed")
    difference={"Protocol":"Session minus physical-speaker (signed difference)"}
    for col in ("Accuracy","Balanced_Accuracy","F1_dysarthric","ROC_AUC","PR_AUC"):
        delta=a[col]-b[col]
        difference[col]=delta.mean()
        difference[col+"_SD"]=delta.std(ddof=1)
    rows.append(difference)
    comparison=pd.DataFrame(rows)
    comparison.to_csv(OUT/"v7_split_protocol_comparison.csv",index=False)
    paper=comparison.rename(columns={"Balanced_Accuracy":"Balanced Accuracy",
        "F1_dysarthric":"F1","ROC_AUC":"ROC-AUC","PR_AUC":"PR-AUC"})[
        ["Protocol","Accuracy","Balanced Accuracy","F1","ROC-AUC","PR-AUC"]]
    paper.to_csv(OUT/"v7_split_protocol_paper_table.csv",index=False)
    return comparison


def verify_v7_result_traceability():
    """Recompute saved metric rows from prediction files and audit figure inputs."""
    checks={}
    meta,_,_=load_torgo()
    meta,_=audit_physical_speakers(meta,save=False)
    expected_splits={(seed,fold):(train,test) for seed in SEEDS
        for fold,train,test in grouped_splits(meta,seed)}
    architecture_keys={"Stream 1 Only (CNN-BiLSTM)":"stream1",
        "Stream 2 Only (Transformer)":"stream2",
        "Dual-Stream (No Pretraining)":"dual_no_pretrain",
        "Dual-Stream (Full)":"dual_full"}
    for family,prediction_name,metric_name in (
        ("classical","v7_corrected_classical_predictions.csv",
            "v7_corrected_classical_metrics.csv"),
        ("architecture","v7_corrected_architecture_predictions.csv",
            "v7_corrected_architecture_ablation.csv")):
        prediction_path=OUT/prediction_name;metric_path=OUT/metric_name
        if not prediction_path.exists() or not metric_path.exists():
            checks[family]="INCOMPLETE";continue
        pred=pd.read_csv(prediction_path);reported=pd.read_csv(metric_path)
        for (model,seed,fold),part in pred.groupby(["model","seed","outer_fold"]):
            train,test=expected_splits[(seed,fold)]
            train_speakers=set(meta.iloc[train].physical_speaker_id)
            test_speakers=set(meta.iloc[test].physical_speaker_id)
            if set(part.file_path)!=set(meta.iloc[test].file_path) or\
                    set(part.physical_speaker_id)!=test_speakers or\
                    train_speakers&test_speakers:
                raise AssertionError(f"{family} outer physical-speaker fold differs: {model}/{seed}/{fold}")
            if family=="architecture":
                key=architecture_keys[model]
                base=OUT/f"v7_arch_{key}_seed{seed}_fold{fold}"
                bundle=torch.load(base.with_suffix(".pt"),map_location="cpu",weights_only=False)
                if set(bundle["train_physical_speakers"])!=train_speakers or\
                        set(bundle["test_physical_speakers"])!=test_speakers:
                    raise AssertionError(f"Architecture checkpoint speaker split differs: {base}")
                inner=pd.read_csv(base.with_suffix(".inner_oof.csv"))
                if set(inner["index"])!=set(train) or not inner["index"].is_unique or\
                        set(inner.physical_speaker_id)!=train_speakers:
                    raise AssertionError(f"Architecture inner OOF coverage differs: {base}")
                temp=calibrate_temperature(inner.label,inner.probability_raw,
                    inner.physical_speaker_id)
                if not np.allclose(part.temperature,temp,atol=1e-10) or\
                        not np.allclose(part.threshold,bundle["threshold"],atol=1e-10):
                    raise AssertionError(f"Architecture TORGO-only calibration differs: {base}")
        for (model,seed),part in pred.groupby(["model","seed"]):
            if not part.file_path.is_unique or part.physical_speaker_id.nunique()!=15:
                raise AssertionError(f"{family} lacks unique full speaker OOF coverage: {model}/{seed}")
            speaker=part.groupby("physical_speaker_id",as_index=False).agg(
                label=("label","first"),probability=("probability","mean"),
                threshold=("threshold","first"))
            for level,frame in (("utterance",part),("physical_speaker",speaker)):
                row=reported[(reported.model.eq(model)) & (reported.seed.eq(seed)) &
                    (reported.level.eq(level))]
                if len(row)!=1:raise AssertionError(f"Missing {family} metric row: {model}/{seed}/{level}")
                actual=_variable_threshold_metrics(frame)
                for metric in METRIC_COLUMNS:
                    if not np.isclose(actual[metric],float(row.iloc[0][metric]),
                            atol=1e-9,rtol=1e-8,equal_nan=True):
                        raise AssertionError(f"{family} {model}/{seed}/{level}/{metric} differs from predictions")
        checks[family]=f"PASS: {len(reported)} metric rows traced to {prediction_name}"
    classical_inner_index=OUT/"v7_corrected_classical_inner_oof_index.csv"
    if classical_inner_index.exists():
        index=pd.read_csv(classical_inner_index)
        if len(index)==75 and set(index.seed)==set(SEEDS):
            for row in index.itertuples(index=False):
                frame=pd.read_csv(OUT/row.source_file)
                train,test=expected_splits[(row.seed,row.outer_fold)]
                if len(frame)!=len(train) or set(frame.file_path)!=set(meta.iloc[train].file_path) or\
                        set(frame.physical_speaker_id)&set(meta.iloc[test].physical_speaker_id):
                    raise AssertionError(f"Classical inner OOF split provenance failed: {row.source_file}")
                if frame.groupby("physical_speaker_id").inner_fold.nunique().max()!=1:
                    raise AssertionError(f"Classical inner OOF speaker repeated: {row.source_file}")
                temp=calibrate_temperature(frame.label,frame.probability_raw,
                    frame.physical_speaker_id)
                if not np.allclose(frame.temperature,temp,atol=1e-10):
                    raise AssertionError(f"Classical OOF temperature differs: {row.source_file}")
            checks["classical_validation"]="PASS: 75 grouped inner OOF calibration files traced to outer development speakers"
        else:checks["classical_validation"]="INCOMPLETE"
    else:checks["classical_validation"]="INCOMPLETE"
    official_path=OUT/"v7_final_official_uaspeech_predictions.csv"
    official_metrics_path=OUT/"v7_final_official_uaspeech_metrics.csv"
    if official_path.exists() and official_metrics_path.exists():
        pred=pd.read_csv(official_path,low_memory=False)
        pred=pred[pred.microphone.eq("M3")]
        rows=pd.read_csv(official_metrics_path)
        for row in rows.itertuples(index=False):
            frame=pred[["physical_speaker_id","label",f"prob_seed{row.seed}"]].rename(
                columns={f"prob_seed{row.seed}":"probability"})
            if row.level=="physical_speaker":
                frame=frame.groupby("physical_speaker_id",as_index=False).agg(
                    label=("label","first"),probability=("probability","mean"))
            actual=metrics(frame.label,frame.probability,row.threshold)
            for metric in METRIC_COLUMNS:
                if not np.isclose(actual[metric],float(getattr(row,metric)),
                        atol=1e-9,rtol=1e-8,equal_nan=True):
                    raise AssertionError(f"Official {row.seed}/{row.level}/{metric} differs from predictions")
        checks["official"]=f"PASS: {len(rows)} metric rows traced to {official_path.name}"
    else:checks["official"]="INCOMPLETE"
    reproduction=OUT/"v7_final_official_historical_reproduction.csv"
    if reproduction.exists():
        matched=pd.read_csv(reproduction)
        if len(matched)!=len(SEEDS)*7 or not matched.max_absolute_probability_difference.eq(0).all():
            raise AssertionError("Fresh official inference differs from preserved frozen-model outputs")
        checks["official_reproduction"]="PASS: 35 channel-seed prediction sets exactly reproduce frozen historical outputs"
    else:checks["official_reproduction"]="INCOMPLETE"
    table_path=OUT/"v7_corrected_baseline_results.csv"
    if table_path.exists():
        table=pd.read_csv(table_path)
        for row in table.itertuples(index=False):
            source=pd.read_csv(OUT/row.source_metrics_file)
            part=source[(source.model.eq(row.model)) &
                (source.level.eq("physical_speaker"))]
            if len(part)!=5:raise AssertionError(f"Primary table incomplete for {row.model}")
            for metric in METRIC_COLUMNS:
                if not np.isclose(part[metric].mean(),getattr(row,metric+"_mean"),atol=1e-12):
                    raise AssertionError(f"Primary table mean differs: {row.model}/{metric}")
                if not np.isclose(part[metric].std(ddof=1),getattr(row,metric+"_sd"),atol=1e-12):
                    raise AssertionError(f"Primary table SD differs: {row.model}/{metric}")
        checks["primary_table"]=f"PASS: {len(table)} model rows traced to saved metric and prediction files"
        paper_path=OUT/"v7_corrected_baseline_paper_table.csv"
        expected_paper=table[["model",*PAPER_BASELINE_COLUMNS]].rename(
            columns=PAPER_BASELINE_RENAME)
        if not paper_path.exists() or not pd.read_csv(paper_path).equals(expected_paper):
            raise AssertionError("Paper-facing baseline table differs from traceable primary table")
        checks["primary_paper_table"]="PASS: requested ten-metric seven-model table matches traceable primary table"
    else:checks["primary_table"]="INCOMPLETE"
    official_master=OUT/"v7_final_official_uaspeech_master_table.csv"
    if official_master.exists() and official_metrics_path.exists():
        from .final_official import make_official_master_table
        saved=pd.read_csv(official_master,keep_default_na=False)
        expected=make_official_master_table(pd.read_csv(official_metrics_path))
        if not saved.equals(expected.fillna("")):
            raise AssertionError("Official master table differs from saved per-seed metrics")
        checks["official_master_table"]="PASS: 12 rows traced to official per-seed metrics"
    else:checks["official_master_table"]="INCOMPLETE"
    split_path=OUT/"v7_split_protocol_comparison.csv"
    split_seed_path=OUT/"v7_split_protocol_per_seed_metrics.csv"
    if split_path.exists() and split_seed_path.exists():
        comparison=pd.read_csv(split_path)
        per_seed=pd.read_csv(split_seed_path)
        physical_prediction=pd.read_csv(OUT/"v7_pooled_oof_torgo_predictions.csv")
        for seed in SEEDS:
            session_prediction=pd.concat([pd.read_csv(OUT/f"v7_controlled_session_seed{seed}_fold{fold}.csv")
                for fold in range(5)],ignore_index=True)
            for protocol,frame in (("Historical-style session-disjoint diagnostic",session_prediction),
                ("Physical-speaker-disjoint primary",
                    physical_prediction[physical_prediction.seed.eq(seed)])):
                reported=per_seed[(per_seed.seed.eq(seed)) & (per_seed.protocol.eq(protocol))]
                if len(reported)!=1 or not frame.file_path.is_unique:
                    raise AssertionError(f"Controlled split prediction coverage failed: {seed}/{protocol}")
                actual=_variable_threshold_metrics(frame)
                for metric in METRIC_COLUMNS:
                    if not np.isclose(actual[metric],float(reported.iloc[0][metric]),
                            atol=1e-9,rtol=1e-8,equal_nan=True):
                        raise AssertionError(f"Controlled split {seed}/{protocol}/{metric} differs from predictions")
        session=per_seed[per_seed.protocol.str.startswith("Historical")].set_index("seed")
        physical=per_seed[per_seed.protocol.str.startswith("Physical")].set_index("seed")
        if len(session)!=5 or len(physical)!=5 or not session.index.equals(physical.index):
            raise AssertionError("Controlled split seed pairs incomplete")
        for row in comparison.itertuples(index=False):
            group=(session[["Accuracy","Balanced_Accuracy","F1_dysarthric","ROC_AUC","PR_AUC"]]-
                physical[["Accuracy","Balanced_Accuracy","F1_dysarthric","ROC_AUC","PR_AUC"]]
                if str(row.Protocol).startswith("Session minus") else
                session if str(row.Protocol).startswith("Historical") else physical)
            for metric in ("Accuracy","Balanced_Accuracy","F1_dysarthric","ROC_AUC","PR_AUC"):
                if not np.isclose(group[metric].mean(),getattr(row,metric),atol=1e-12):
                    raise AssertionError(f"Controlled split table mean differs: {row.Protocol}/{metric}")
                if not np.isclose(group[metric].std(ddof=1),getattr(row,metric+"_SD"),atol=1e-12):
                    raise AssertionError(f"Controlled split table SD differs: {row.Protocol}/{metric}")
        checks["split_table"]="PASS: three rows traced to paired per-seed metrics and predictions"
        split_paper=OUT/"v7_split_protocol_paper_table.csv"
        expected_paper=comparison.rename(columns={"Balanced_Accuracy":"Balanced Accuracy",
            "F1_dysarthric":"F1","ROC_AUC":"ROC-AUC","PR_AUC":"PR-AUC"})[
            ["Protocol","Accuracy","Balanced Accuracy","F1","ROC-AUC","PR-AUC"]]
        if not split_paper.exists() or not np.allclose(
                pd.read_csv(split_paper).iloc[:,1:].to_numpy(float),
                expected_paper.iloc[:,1:].to_numpy(float),atol=1e-12) or\
                not pd.read_csv(split_paper).Protocol.equals(expected_paper.Protocol):
            raise AssertionError("Paper-facing split table differs from traceable comparison")
        checks["split_paper_table"]="PASS: requested split-effect table matches paired prediction metrics"
    else:checks["split_table"]="INCOMPLETE"
    index_path=OUT/"v7_corrected_publication_figure_index.csv"
    if index_path.exists():
        index=pd.read_csv(index_path)
        for row in index.itertuples(index=False):
            for name in str(row.source_file).split("; "):
                if not (OUT/name).is_file():raise AssertionError(f"Figure source missing: {name}")
            if not Path(row.png).is_file() or not Path(row.pdf).is_file():
                raise AssertionError(f"Figure artifact missing: {row.figure}")
        checks["figures"]=f"PASS: {len(index)} PNG/PDF pairs have saved source files"
    else:checks["figures"]="INCOMPLETE"
    (OUT/"v7_result_traceability_audit.json").write_text(json.dumps(checks,indent=2)+"\n")
    return checks
