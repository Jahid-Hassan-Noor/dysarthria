"""Read-only V8 adapter, strict cached-feature checks, and artifact registration."""
from __future__ import annotations
import hashlib
import json
import os
import sys
import types
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch

V8_DIR = Path(__file__).resolve().parents[1]
PUBLICATION_ROOT = V8_DIR / 'v8_publication_completion'
REFERENCE_NAME = 'WavLM Last-4 Mix + Attention (V8 Reference)'
METRICS = ('Accuracy','Balanced_Accuracy','Sensitivity','Specificity','F1_dysarthric','ROC_AUC','PR_AUC','Brier','ECE')

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def canonical_hash(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':'),default=str).encode()).hexdigest()

def read_json(path): return json.loads(Path(path).read_text())

def write_json(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    tmp.write_text(json.dumps(data,indent=2,sort_keys=True,default=str));os.replace(tmp,path)

def write_csv(path,frame):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    frame.to_csv(tmp,index=False);os.replace(tmp,path)

def load_pipeline(notebook_path=None):
    """Execute definitions only: never run original fitting/plot/evaluation cells."""
    path=Path(notebook_path or V8_DIR/'dysarthria_v8.ipynb')
    notebook=read_json(path)
    module=types.ModuleType('v8_publication_reference_'+hashlib.sha256(str(path).encode()).hexdigest()[:12])
    sys.modules[module.__name__]=module
    for cell in notebook['cells']:
        if cell['cell_type']=='code' and 'v8-definitions' in cell.get('metadata',{}).get('tags',[]):
            source=cell['source'];source=''.join(source) if isinstance(source,list) else source
            exec(compile(source,str(path),'exec'),module.__dict__)
    module.V8_DIR=V8_DIR;module.ROOT=V8_DIR.parent
    module.OUT=V8_DIR/'outputs';module.CACHE_ROOT=module.OUT/'cache/dysarthria_detection_v8'
    return module

@dataclass
class Context:
    pipeline: object
    config: object
    run_dir: Path
    original_run_dir: Path
    output_dir: Path
    meta: pd.DataFrame
    speech: np.ndarray
    acoustic: np.ndarray
    run_signature: str
    manifest: dict
    lock: dict
    device: torch.device
    frozen_verified: bool = True


def validate_features(pipeline,config,feature_dir,expected_spec,expected_source=None,check_sources=True):
    """Mismatch is an error, never a reason to overwrite or silently rebuild a cache."""
    feature_dir=Path(feature_dir); manifest=read_json(feature_dir/'manifest.json')
    if not manifest.get('complete') or manifest.get('extraction_errors'):
        raise RuntimeError(f'Incomplete feature cache: {feature_dir}')
    if manifest.get('feature_spec')!=expected_spec or manifest.get('feature_signature')!=canonical_hash(expected_spec):
        raise RuntimeError(f'Cached representation signature mismatch: {feature_dir}')
    meta=pd.read_csv(feature_dir/'usable_metadata.csv')
    signature=pipeline.source_signature(meta)
    if signature!=manifest['source_signature'] or (expected_source is not None and signature!=expected_source):
        raise RuntimeError(f'Feature source signature mismatch: {feature_dir}')
    speech=np.load(feature_dir/'speech_embeddings.npy',mmap_mode='r')
    acoustic=np.load(feature_dir/'acoustic_features.npy',mmap_mode='r')
    if len(meta)!=len(speech) or len(meta)!=len(acoustic) or list(speech.shape)!=manifest['embedding_shape']:
        raise RuntimeError('Cache dimensions disagree with metadata')
    if speech.shape[1:]!=(config.selected_layers,1536): raise RuntimeError('Unexpected WavLM layer dimensions')
    if check_sources:
        for row in meta.itertuples(index=False):
            stat=Path(row.file_path).stat()
            if stat.st_size!=row.source_size or stat.st_mtime_ns!=row.source_mtime_ns:
                raise RuntimeError(f'Raw audio changed after extraction: {row.file_path}')
    return meta,speech,acoustic


def make_context(publication_root=PUBLICATION_ROOT,device=None,verify=True):
    root=Path(publication_root).resolve()
    frozen=root/'frozen_reference'
    # Snapshot layout is recorded by freeze.py, locate the unambiguous frozen run.
    manifests=list(frozen.rglob('run_manifest.json'))
    if len(manifests)!=1: raise RuntimeError('Freeze the successful V8 run before experiments')
    run_dir=manifests[0].parent
    notebooks=list(frozen.rglob('*.ipynb'))
    if len(notebooks)!=1: raise RuntimeError('Expected one frozen reference notebook')
    if verify:
        from .freeze import verify_reference
        verify_reference(frozen, compare_originals=True)
    pipeline=load_pipeline(notebooks[0]);manifest=read_json(run_dir/'run_manifest.json')
    cfg=dict(manifest['config']);cfg['seeds']=tuple(cfg['seeds']);config=pipeline.Config(**cfg)
    config.validate()
    if config.encoder_name!='WAVLM_BASE' or config.selected_layers!=4 or config.use_fusion:
        raise RuntimeError('Publication protocol requires the frozen speech-only last-four WavLM reference')
    if pipeline._feature_spec(config,manifest['pretrained_weight_sha256'])!=manifest['feature_spec']:
        raise RuntimeError('Active encoder/preprocessing versions differ from frozen feature specification')
    meta,speech,acoustic=validate_features(pipeline,config,run_dir/'features',manifest['feature_spec'],manifest['raw_source_signature'])
    lock=read_json(run_dir/'deployment/v8_torgo_model_lock.json')
    pipeline._verify_torgo_deployment_lock(lock,config,run_dir,manifest)
    original=Path(read_json(V8_DIR/'outputs/v8_latest_run.json')['output_dir'])
    ctx=Context(pipeline,config,run_dir,original,root,meta,speech,acoustic,manifest['run_signature'],manifest,lock,
                torch.device(device) if device else pipeline.choose_device())
    return ctx


def external_features(ctx,channel='M3'):
    spec=dict(ctx.manifest['feature_spec'],external_preprocessing='normalized_UASpeech_mic_unknown_trim25_no_gate_v1')
    directory=(ctx.run_dir/'external_uaspeech'/channel/'features' if channel=='M3' else ctx.output_dir/'external_cache'/channel/'features')
    return validate_features(ctx.pipeline,ctx.config,directory,spec)


def stage_status(ctx,name,state,reason=None,**extra):
    path=ctx.output_dir/'status.json';data=read_json(path) if path.exists() else {'stages':{}}
    data['stages'][name]={'state':state,'updated_utc':datetime.now(timezone.utc).isoformat(),**extra}
    if reason:data['stages'][name]['reason']=reason
    write_json(path,data)


def external_split_manifest(ctx,record):
    path=ctx.output_dir/'splits'/f'deployment_seed{record["seed"]}.json'
    source=ctx.run_dir/record['checkpoint'];seed_dir=source.parent.parent
    inner=[]
    for p in sorted(seed_dir.glob('inner*/complete.json')):
        data=read_json(p)
        inner.append({'train_physical_speakers':['TORGO:'+s for s in data['train_speakers']],
                      'validation_physical_speakers':['TORGO:'+s for s in data['validation_speakers']]})
    data={'training_corpus':'TORGO','evaluation_corpus':'UASpeech','evaluation_only':True,
          'run_signature':ctx.run_signature,'train_physical_speakers':['TORGO:'+s for s in sorted(ctx.meta.physical_speaker_id.unique())],
          'test_physical_speakers':['UASpeech:'+s for s in sorted(set.union(*ctx.pipeline.UASPEECH_ROSTER.values()))],
          'inner_folds':inner,'threshold_source':record['threshold_source'],
          'selected_threshold':record['threshold'],'checkpoint_sha256':record['checkpoint_sha256']}
    if path.exists() and read_json(path)!=data:raise RuntimeError('Existing deployment split provenance changed')
    write_json(path,data);return path


def add_reference_lineage(ctx,predictions,corpus):
    result=predictions.copy();result['corpus']=corpus;result['run_signature']=ctx.run_signature
    result['model_arm']=REFERENCE_NAME
    for ix,row in result.iterrows():
        if corpus=='TORGO':
            directory=ctx.run_dir/'folds/speech_only'/f'seed{int(row.seed)}_outer{int(row.outer_fold)}'
            checkpoint=directory/'final/best.pt';split=directory/'fold_audit.json'
        else:
            record=next(r for r in ctx.lock['models'] if r['seed']==int(row.seed) and r['model_arm']=='speech_only')
            checkpoint=ctx.run_dir/record['checkpoint'];split=external_split_manifest(ctx,record)
        result.loc[ix,'checkpoint_path']=str(checkpoint)
        result.loc[ix,'checkpoint_sha256']=sha256(checkpoint)
        result.loc[ix,'split_manifest']=str(split)
    return result


def publish_result(ctx,name,predictions,group_cols=None):
    if not ctx.frozen_verified: raise RuntimeError('Unverified frozen reference')
    frame=predictions.copy();frame['run_signature']=ctx.run_signature
    frame['analysis']=name
    if group_cols is None:
        group_cols=[c for c in ('corpus','model_arm','channel','budget','repetition','seed') if c in frame]
    if 'seed' not in group_cols:raise ValueError('Metrics must retain seed')
    if frame.empty:raise ValueError('Cannot publish empty predictions')
    required={'physical_speaker_id','label','probability','threshold','seed','checkpoint_path','checkpoint_sha256','split_manifest'}
    if not required.issubset(frame):raise ValueError(f'Missing lineage columns: {required-set(frame)}')
    if frame.duplicated(group_cols+['physical_speaker_id']).any():raise ValueError('Duplicate physical speaker predictions')
    if not np.isfinite(frame[['probability','threshold']].to_numpy()).all():raise ValueError('Nonfinite predictions')
    frame['predicted_label']=(frame.probability>=frame.threshold).astype(int)
    directory=ctx.output_dir/'results'/name;directory.mkdir(parents=True,exist_ok=True)
    path=directory/'predictions.csv';write_csv(path,frame)
    # Metrics are calculated from the round-tripped artifact that readers will use.
    frame=pd.read_csv(path);rows=[]
    for key,part in frame.groupby(group_cols,dropna=False,sort=True):
        key=key if isinstance(key,tuple) else (key,)
        rows.append(dict(zip(group_cols,key))|ctx.pipeline.speaker_metrics(part))
    metric_path=directory/'per_seed_metrics.csv';write_csv(metric_path,pd.DataFrame(rows))
    provenance={'name':name,'run_signature':ctx.run_signature,'pipeline_signature':ctx.manifest['pipeline_signature'],
                'prediction_sha256':sha256(path),'metric_sha256':sha256(metric_path),'group_columns':group_cols,
                'source_signatures':{'TORGO':ctx.manifest['raw_source_signature']},'external_model_selection':False,
                'created_utc':datetime.now(timezone.utc).isoformat()}
    prov_path=directory/'provenance.json';write_json(prov_path,provenance)
    registry_path=ctx.output_dir/'registry.json'
    registry=read_json(registry_path) if registry_path.exists() else {'schema_version':1,'run_signature':ctx.run_signature,'datasets':{},'frozen_reference':str(ctx.output_dir/'frozen_reference')}
    registry['datasets'][name]={'prediction_file':str(path),'metric_file':str(metric_path),'group_columns':group_cols,'provenance_file':str(prov_path)}
    write_json(registry_path,registry)
    stage_status(ctx,name,'complete',n_prediction_rows=len(frame),n_speakers=int(frame.physical_speaker_id.nunique()))
    return directory


def register_primary(ctx):
    for name,corpus,path in [('primary_torgo','TORGO',ctx.run_dir/'v8_speaker_oof_predictions.csv'),
                             ('primary_uaspeech_m3','UASpeech',ctx.run_dir/'external_uaspeech/M3/v8_uaspeech_speaker_predictions.csv')]:
        predictions=add_reference_lineage(ctx,pd.read_csv(path),corpus)
        if corpus=='UASpeech':predictions['channel']='M3'
        publish_result(ctx,name,predictions)


def frozen_model(ctx,seed,outer_fold=None,meta=None,speech=None,acoustic=None,device=None):
    device=device or ctx.device
    if outer_fold is None:
        record=next(r for r in ctx.lock['models'] if r['seed']==int(seed) and r['model_arm']=='speech_only')
        checkpoint=ctx.run_dir/record['checkpoint'];scaler_path=ctx.run_dir/record['acoustic_scaler']
        threshold=float(record['threshold']);split=external_split_manifest(ctx,record)
        if sha256(checkpoint)!=record['checkpoint_sha256']:raise RuntimeError('Locked checkpoint hash mismatch')
    else:
        directory=ctx.run_dir/'folds/speech_only'/f'seed{int(seed)}_outer{int(outer_fold)}'
        checkpoint=directory/'final/best.pt';scaler_path=directory/'final/acoustic_scaler.npz'
        split=directory/'fold_audit.json';audit=read_json(split);threshold=float(audit['selected_threshold'])
    model=ctx.pipeline._new_head(ctx.config,ctx.speech,ctx.acoustic,False,device)
    model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True));model.eval()
    with np.load(scaler_path) as data:scaler=(data['mean'].copy(),data['scale'].copy())
    store=ctx.pipeline.BagStore(ctx.meta if meta is None else meta,ctx.speech if speech is None else speech,
        ctx.acoustic if acoustic is None else acoustic,scaler,device)
    lineage={'checkpoint_path':str(checkpoint),'checkpoint_sha256':sha256(checkpoint),'split_manifest':str(split),
             'threshold':threshold,'run_signature':ctx.run_signature,'seed':int(seed),'model_arm':REFERENCE_NAME}
    if outer_fold is not None:lineage['outer_fold']=int(outer_fold)
    return model,store,lineage
