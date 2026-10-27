"""Predeclared duration and recording-artifact logistic sanity controls."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .common import (canonical_hash, external_features, publish_result, read_json, sha256,
                     stage_status, write_csv, write_json)

DURATION_FEATURES=('duration_mean','duration_median','duration_sd','short_fraction')
ARTIFACT_FEATURES=DURATION_FEATURES+('rms_mean','rms_sd','peak_mean','silence_ratio_mean')
POLICY={'version':'v8_recording_sanity_v1','duration':'source frames / sample rate, before trimming or normalization',
        'short_fraction':'fraction of raw recordings under 0.5 seconds',
        'rms':'root mean square of mono waveform in released source',
        'peak':'maximum absolute amplitude of mono waveform in released source',
        'silence_ratio':'fraction of raw mono samples below 1% of utterance peak; all-zero recording=1',
        'aggregation':'one equally weighted speaker feature row, population SD',
        'classifier':'StandardScaler + LogisticRegression(C=1, class_weight=balanced, solver=lbfgs,max_iter=1000)',
        'selection':'fixed features and C, no hyperparameter search; same TORGO inner OOF sensitivity>=0.9 threshold',
        'external':'fit/freeze all TORGO deployment models before predicting official UASpeech M3'}

def features(ctx,meta,corpus,kind):
    directory=ctx.output_dir/'baselines/features'/corpus;directory.mkdir(parents=True,exist_ok=True)
    target=directory/f'{kind}_speaker_features.csv';manifest=directory/f'{kind}_manifest.json'
    signature=canonical_hash({'source':ctx.pipeline.source_signature(meta),'policy':POLICY,'kind':kind,'code':sha256(Path(__file__))})
    if manifest.exists():
        saved=read_json(manifest)
        if saved['signature']!=signature or saved['feature_sha256']!=sha256(target):raise RuntimeError('Baseline feature cache signature mismatch')
        return pd.read_csv(target)
    frame=meta[['source_id','physical_speaker_id','label','source_seconds']].copy()
    if kind=='artifact':
        utterance_path=directory/'artifact_utterance_features.csv';receipt=directory/'artifact_utterance_features.json'
        previous=pd.read_csv(utterance_path) if utterance_path.exists() else pd.DataFrame()
        if receipt.exists() and read_json(receipt)['signature']!=signature:raise RuntimeError('Partial artifact feature signature mismatch')
        write_json(receipt,{'signature':signature})
        cache=previous.set_index('source_id').to_dict('index') if len(previous) else {}
        for i,row in enumerate(meta.itertuples(index=False)):
            if row.source_id in cache:continue
            stat=Path(row.file_path).stat()
            if stat.st_size!=row.source_size or stat.st_mtime_ns!=row.source_mtime_ns:raise RuntimeError('Raw recording changed')
            audio,_=sf.read(row.file_path,dtype='float32',always_2d=True);audio=audio.mean(1)
            if len(audio)==0 or not np.isfinite(audio).all():raise RuntimeError(f'Invalid sanity audio: {row.file_path}')
            peak=float(np.abs(audio).max());rms=float(np.sqrt(np.mean(audio.astype(np.float64)**2)))
            silence=float(np.mean(np.abs(audio)<.01*peak)) if peak else 1.
            cache[row.source_id]={'rms':rms,'peak':peak,'silence_ratio':silence}
            if (i+1)%2000==0:
                write_csv(utterance_path,pd.DataFrame.from_dict(cache,orient='index').rename_axis('source_id').reset_index())
                print(f'{corpus} artifact features {i+1}/{len(meta)}',flush=True)
        utterance=pd.DataFrame.from_dict(cache,orient='index').rename_axis('source_id').reset_index()
        write_csv(utterance_path,utterance)
        frame=frame.merge(utterance,on='source_id',validate='one_to_one')
    rows=[]
    for speaker,part in frame.groupby('physical_speaker_id',sort=True):
        if part.label.nunique()!=1:raise RuntimeError('Speaker label inconsistency')
        d=part.source_seconds.to_numpy()
        item={'physical_speaker_id':speaker,'label':int(part.label.iloc[0]),'n_utterances':len(part),
              'duration_mean':d.mean(),'duration_median':np.median(d),'duration_sd':d.std(ddof=0),'short_fraction':float((d<.5).mean())}
        if kind=='artifact':item.update(rms_mean=part.rms.mean(),rms_sd=part.rms.std(ddof=0),peak_mean=part.peak.mean(),silence_ratio_mean=part.silence_ratio.mean())
        rows.append(item)
    table=pd.DataFrame(rows);write_csv(target,table)
    write_json(manifest,{'signature':signature,'feature_sha256':sha256(target),'source_signature':ctx.pipeline.source_signature(meta),'policy':POLICY})
    return table

def fit_classifier(table,train_ids,names,seed):
    train=table.set_index('physical_speaker_id').loc[sorted(train_ids)]
    if train.label.nunique()!=2:raise ValueError('Both diagnoses required in TORGO training partition')
    model=make_pipeline(StandardScaler(),LogisticRegression(C=1,class_weight='balanced',solver='lbfgs',max_iter=1000,random_state=seed))
    model.fit(train[list(names)].to_numpy(),train.label.to_numpy());return model

def predict(model,table,speakers,names):
    part=table.set_index('physical_speaker_id').loc[sorted(speakers)].reset_index()
    result=part[['physical_speaker_id','label','n_utterances']].copy()
    result['probability']=model.predict_proba(part[list(names)].to_numpy())[:,1]
    return result

def save_model(model,directory,names):
    directory.mkdir(parents=True,exist_ok=True);checkpoint=directory/'model.joblib';joblib.dump(model,checkpoint)
    scale=model.named_steps['standardscaler'];lr=model.named_steps['logisticregression']
    write_json(directory/'interpretable_model.json',{'features':names,'scaler_mean':scale.mean_.tolist(),'scaler_scale':scale.scale_.tolist(),
              'standardized_coefficients':lr.coef_[0].tolist(),'intercept':float(lr.intercept_[0]),'classes':lr.classes_.tolist(),
              'checkpoint_sha256':sha256(checkpoint)})
    return checkpoint

def run_kind(ctx,kind):
    if kind not in ('duration','artifact'):raise ValueError(kind)
    names=DURATION_FEATURES if kind=='duration' else ARTIFACT_FEATURES
    stage_status(ctx,kind,'running');directory=ctx.output_dir/'baselines'/kind;directory.mkdir(parents=True,exist_ok=True)
    protocol=dict(POLICY,feature_names=names,run_signature=ctx.run_signature,implementation_sha256=sha256(Path(__file__)))
    protocol_path=directory/'protocol.json'
    if protocol_path.exists() and canonical_hash(read_json(protocol_path))!=canonical_hash(protocol):raise RuntimeError('Baseline protocol changed')
    write_json(protocol_path,protocol)
    table=features(ctx,ctx.meta,'TORGO',kind);all_ids=set(table.physical_speaker_id);rows=[];locked=[]
    for seed in ctx.config.seeds:
        for outer,train_ids,test_ids in ctx.pipeline.grouped_speaker_splits(ctx.meta,seed,ctx.config.outer_folds):
            audit=read_json(ctx.run_dir/'folds/speech_only'/f'seed{seed}_outer{outer}'/'fold_audit.json')
            if train_ids!=set(audit['train_physical_speakers']) or test_ids!=set(audit['test_physical_speakers']):raise RuntimeError('Baseline split drift')
            validation=[]
            for inner in audit['inner_folds']:
                train=set(inner['train_physical_speakers']);val=set(inner['validation_physical_speakers'])
                if train&val or train&test_ids or val&test_ids:raise RuntimeError('Baseline nested speaker leakage')
                model=fit_classifier(table,train,names,seed)
                validation.append(predict(model,table,val,names))
            validation=pd.concat(validation,ignore_index=True)
            threshold=ctx.pipeline.select_threshold(validation)
            fold_dir=directory/f'seed{seed}_outer{outer}';fold_dir.mkdir(parents=True,exist_ok=True)
            write_csv(fold_dir/'inner_oof_predictions.csv',validation)
            model=fit_classifier(table,train_ids,names,seed);checkpoint=save_model(model,fold_dir,names)
            split=dict(audit,selected_threshold=threshold,model_arm=f'{kind.title()}-only logistic regression',run_signature=ctx.run_signature)
            split_path=fold_dir/'split_manifest.json';write_json(split_path,split)
            pred=predict(model,table,test_ids,names)
            pred['seed']=seed;pred['outer_fold']=outer;pred['corpus']='TORGO';pred['threshold']=threshold
            pred['checkpoint_path']=str(checkpoint);pred['checkpoint_sha256']=sha256(checkpoint);pred['split_manifest']=str(split_path)
            rows.append(pred)
        # Freeze each deployment model using TORGO grouped inner OOF only.
        validation=[];inner_rows=[]
        for inner,train,val in ctx.pipeline.grouped_speaker_splits(ctx.meta,seed+731,ctx.config.outer_folds):
            if train&val or train|val!=all_ids:raise RuntimeError('Deployment split invalid')
            model=fit_classifier(table,train,names,seed)
            validation.append(predict(model,table,val,names))
            inner_rows.append({'train_physical_speakers':['TORGO:'+s for s in sorted(train)],'validation_physical_speakers':['TORGO:'+s for s in sorted(val)]})
        validation=pd.concat(validation,ignore_index=True);threshold=ctx.pipeline.select_threshold(validation)
        deploy=directory/'deployment'/f'seed{seed}';deploy.mkdir(parents=True,exist_ok=True)
        write_csv(deploy/'torgo_inner_oof_predictions.csv',validation)
        model=fit_classifier(table,all_ids,names,seed);checkpoint=save_model(model,deploy,names)
        split_path=deploy/'split_manifest.json'
        split={'train_physical_speakers':['TORGO:'+s for s in sorted(all_ids)],
            'test_physical_speakers':['UASpeech:'+s for s in sorted(set.union(*ctx.pipeline.UASPEECH_ROSTER.values()))],
            'inner_folds':inner_rows,'evaluation_only':True,'training_corpus':'TORGO','evaluation_corpus':'UASpeech',
            'threshold_source':'TORGO grouped inner speaker OOF only','selected_threshold':threshold,'run_signature':ctx.run_signature}
        write_json(split_path,split)
        locked.append({'seed':seed,'threshold':threshold,'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'split_manifest':str(split_path)})
    lock_path=directory/'deployment_lock.json';write_json(lock_path,{'models':locked,'training_corpus':'TORGO','protocol':protocol,'external_fit':False})
    external_meta,_,_=external_features(ctx)
    external=features(ctx,external_meta,'UASpeech',kind)
    for record in locked:
        if sha256(record['checkpoint_path'])!=record['checkpoint_sha256']:raise RuntimeError('Baseline locked checkpoint changed')
        model=joblib.load(record['checkpoint_path']);pred=predict(model,external,set(external.physical_speaker_id),names)
        for key,value in record.items():pred[key]=value
        pred['corpus']='UASpeech';rows.append(pred)
    predictions=pd.concat(rows,ignore_index=True);predictions['model_arm']=f'{kind.title()}-only logistic regression'
    predictions['feature_manifest']=str(directory/'protocol.json')
    publish_result(ctx,kind,predictions,group_cols=['corpus','model_arm','seed'])
    print(f'{kind} baseline: TORGO OOF and frozen UASpeech complete',flush=True)
    return predictions

def run(ctx):
    return {kind:run_kind(ctx,kind) for kind in ('duration','artifact')}
