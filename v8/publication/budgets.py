"""Frozen-model utterance budgets, with identical eligible cohorts and saved draws."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from .common import (REFERENCE_NAME, add_reference_lineage, external_features, frozen_model,
                     publish_result, read_json, sha256, stage_status, write_csv, write_json)

BUDGETS=(1,5,10,20,50)
REPETITIONS=100
SAMPLING_VERSION='v8_budget_without_replacement_common_ge50_v1'

def sampling_seed(corpus,speaker,seed,repetition):
    text=f'{SAMPLING_VERSION}|{corpus}|{speaker}|{seed}|{repetition}'
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8],'big')

@torch.no_grad()
def project_utterances(model,speech,batch_size=512):
    """Cache the frozen eval-mode projection; no learned operation is changed."""
    if model.training or model.use_fusion: raise ValueError('Expected frozen speech-only eval head')
    device=next(model.parameters()).device;parts=[]
    mix=torch.softmax(model.layer_logits,dim=0)
    for start in range(0,len(speech),batch_size):
        value=torch.from_numpy(np.array(speech[start:start+batch_size],dtype=np.float32,copy=True)).to(device)
        parts.append(model.speech_proj((value*mix[None,:,None]).sum(1)).cpu())
    return torch.cat(parts)

@torch.no_grad()
def projected_probabilities(model,projected,selections):
    """Batched independent bags, mathematically identical to V8's eval forward."""
    device=next(model.parameters()).device
    utterance=projected[torch.as_tensor(selections,dtype=torch.long)].to(device)
    context=utterance.mean(1)
    expanded=context[:,None,:].expand_as(utterance)
    score=model.attention(torch.cat((utterance,expanded,utterance*expanded),-1)).squeeze(-1)
    weight=torch.softmax(score,dim=1)
    pooled=torch.cat(((utterance*weight[:,:,None]).sum(1),context),-1)
    return torch.sigmoid(model.classifier(pooled).squeeze(-1)).cpu().numpy()

def run_corpus(ctx,corpus,*,repetitions=REPETITIONS):
    if repetitions!=REPETITIONS:raise ValueError('Publication protocol requires 100 fixed repetitions')
    name='budgets_torgo' if corpus=='TORGO' else 'budgets_uaspeech'
    stage_status(ctx,name,'running')
    directory=ctx.output_dir/'budgets'/corpus;directory.mkdir(parents=True,exist_ok=True)
    if corpus=='TORGO':meta,speech,acoustic=ctx.meta,ctx.speech,ctx.acoustic
    else:meta,speech,acoustic=external_features(ctx)
    counts=meta.groupby('physical_speaker_id').size()
    eligible=set(counts[counts>=max(BUDGETS)].index)
    if meta.loc[meta.physical_speaker_id.isin(eligible),'label'].nunique()!=2:
        stage_status(ctx,name,'not_estimable',reason='Common >=50 utterance cohort lacks both diagnoses');return
    cohort=pd.DataFrame({'physical_speaker_id':counts.index,'n_available':counts.values})
    cohort['eligible_all_budgets']=cohort.physical_speaker_id.isin(eligible)
    write_csv(directory/'cohort.csv',cohort)
    policy={'version':SAMPLING_VERSION,'corpus':corpus,'budgets':[*BUDGETS,'all'],'repetitions':repetitions,
        'sampling':'Without replacement. Prefixes of same seeded permutation couple finite budgets.',
        'eligibility':'Common cohort >=50 usable utterances; no duplication. All uses same cohort.',
        'finite_bag_rule':'Budgets <=50 each form one bag, preserving frozen head forward.',
        'all_bag_rule':'Original deterministic 64-utterance bags and weighted-logit average.',
        'thresholds':'Original fold-specific TORGO / deployment TORGO thresholds, unchanged.',
        'run_signature':ctx.run_signature,'metadata_sha256':sha256((ctx.run_dir/'features' if corpus=='TORGO' else ctx.run_dir/'external_uaspeech/M3/features')/'usable_metadata.csv'),
        'n_total_speakers':len(counts),'n_eligible_speakers':len(eligible),'implementation_sha256':sha256(Path(__file__))}
    policy_path=directory/'protocol.json'
    if policy_path.exists() and read_json(policy_path)!=policy:raise RuntimeError('Budget protocol changed; refusing cached draws')
    write_json(policy_path,policy)
    original_path=ctx.run_dir/('v8_speaker_oof_predictions.csv' if corpus=='TORGO' else 'external_uaspeech/M3/v8_uaspeech_speaker_predictions.csv')
    original=add_reference_lineage(ctx,pd.read_csv(original_path),corpus)
    all_parts=[];torch.set_num_threads(2)
    for seed in ctx.config.seeds:
        seed_path=directory/f'seed{seed}_predictions.csv';draw_path=directory/f'seed{seed}_draws.csv';receipt=directory/f'seed{seed}_receipt.json'
        if receipt.exists():
            saved=read_json(receipt)
            if saved['protocol_sha256']!=sha256(policy_path) or saved['predictions_sha256']!=sha256(seed_path) or saved['draws_sha256']!=sha256(draw_path):
                raise RuntimeError('Budget cached results changed')
            all_parts.append(pd.read_csv(seed_path));continue
        rows=[];draws=[]
        groups=original.loc[original.seed.eq(seed)].groupby('outer_fold') if corpus=='TORGO' else [(None,original.loc[original.seed.eq(seed)])]
        for outer,part in groups:
            speakers=sorted(set(part.physical_speaker_id)&eligible)
            # Project only held-out speakers for TORGO; keeps compute proportional to evaluation data.
            mask=meta.physical_speaker_id.isin(speakers).to_numpy();subset=meta.loc[mask].reset_index(drop=True)
            sub_speech=speech[mask];sub_acoustic=acoustic[mask]
            model,store,lineage=frozen_model(ctx,seed,outer,subset,sub_speech,sub_acoustic,device=torch.device('cpu'))
            projected=project_utterances(model,sub_speech)
            for sp in speakers:
                indices=store.indices[sp]
                selections=np.stack([np.random.default_rng(sampling_seed(corpus,sp,seed,rep)).permutation(indices)[:50] for rep in range(repetitions)])
                # Check the batched shortcut against the unmodified reference forward on real inputs.
                for budget in BUDGETS:
                    selected=selections[:,:budget]
                    probs=projected_probabilities(model,projected,selected)
                    raw,raw_acoustic=store.bag(sp,explicit_indices=selected[0])
                    with torch.no_grad(): direct=float(torch.sigmoid(model(raw,raw_acoustic)).item())
                    if not np.isclose(direct,float(probs[0]),rtol=1e-5,atol=2e-6):raise RuntimeError('Frozen inference vectorization parity failed')
                    for rep,prob in enumerate(probs):
                        rows.append(dict(lineage,corpus=corpus,physical_speaker_id=sp,label=store.labels[sp],
                            probability=float(prob),budget=str(budget),repetition=rep,n_utterances=budget,n_available=len(indices),
                            n_eligible_speakers=len(eligible),n_total_speakers=len(counts),channel='M3' if corpus=='UASpeech' else 'TORGO',
                            sampling_seed=str(sampling_seed(corpus,sp,seed,rep)),sampling_manifest=str(draw_path)))
                for rep,selection in enumerate(selections):
                    draws.append({'seed':seed,'physical_speaker_id':sp,'repetition':rep,'sampling_seed':str(sampling_seed(corpus,sp,seed,rep)),
                                  'selected_source_ids_json':json.dumps(subset.iloc[selection].source_id.tolist()),'prefix_policy':'first k for budget k'})
            # The all-utterance anchor is the saved reference prediction, with identical model + threshold.
            anchor=part.loc[part.physical_speaker_id.isin(speakers)].copy()
            anchor['budget']='all';anchor['repetition']=0;anchor['n_eligible_speakers']=len(eligible);anchor['n_total_speakers']=len(counts)
            anchor['channel']='M3' if corpus=='UASpeech' else 'TORGO';anchor['n_available']=anchor.n_utterances
            rows.extend(anchor.to_dict('records'))
            del model,store,projected,sub_speech,sub_acoustic
        frame=pd.DataFrame(rows);write_csv(seed_path,frame);write_csv(draw_path,pd.DataFrame(draws))
        write_json(receipt,{'protocol_sha256':sha256(policy_path),'predictions_sha256':sha256(seed_path),'draws_sha256':sha256(draw_path)})
        all_parts.append(frame);print(f'Budget {corpus} seed {seed}: 100 repetitions complete',flush=True)
    predictions=pd.concat(all_parts,ignore_index=True)
    publish_result(ctx,name,predictions,group_cols=['corpus','model_arm','channel','budget','repetition','seed'])
    return predictions

def run(ctx):
    return {corpus:run_corpus(ctx,corpus) for corpus in ('TORGO','UASpeech')}
