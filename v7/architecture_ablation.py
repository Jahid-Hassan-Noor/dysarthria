"""Grouped neural architecture comparison called by the existing V7 notebook.

The four candidates share one cached-feature augmentation and TORGO-only nested
evaluation. These results are kept separate from the earlier waveform-augmented
V7 Stream-1 checkpoints. Official UASpeech is never loaded here.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import nn
from torch.nn import functional as F

from .methodology import (OUT, SEEDS, SpectroTemporalEncoder, _cache_paths,
    _speaker_balanced_indices, apply_temperature, assert_disjoint,
    audit_physical_speakers, calibrate_temperature, device, grouped_splits,
    load_torgo, threshold_table)
from .corrected_analysis import METRIC_COLUMNS, _variable_threshold_metrics

ARCH_NAMES=("Stream 1 Only (CNN-BiLSTM)","Stream 2 Only (Transformer)",
    "Dual-Stream (No Pretraining)","Dual-Stream (Full)")
ARCH_KEYS=("stream1","stream2","dual_no_pretrain","dual_full")
SUPERVISED_MAX_EPOCHS=20
SAMPLES_PER_EPOCH=512
PRETRAIN_EPOCHS=50
ABLATION_SIGNATURE="v7_grouped_cached_dual_feature_architecture_ablation_v1"


class PositionalEncoding(nn.Module):
    def __init__(self,width=256,max_len=128):
        super().__init__()
        position=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,width,2).float()*(-np.log(10000.0)/width))
        pe=torch.zeros(max_len,width)
        pe[:,0::2]=torch.sin(position*div)
        pe[:,1::2]=torch.cos(position*div)
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x):return x+self.pe[:,:x.size(1)]


class TrajectoryEncoder(nn.Module):
    """V6 4-layer, 4-head trajectory transformer, unchanged in capacity."""
    def __init__(self):
        super().__init__()
        self.input_proj=nn.Linear(64,256)
        self.pos_encoding=PositionalEncoding()
        layer=nn.TransformerEncoderLayer(d_model=256,nhead=4,
            dim_feedforward=512,dropout=.1,batch_first=True)
        self.transformer=nn.TransformerEncoder(layer,num_layers=4,
            enable_nested_tensor=False)
        self.norm=nn.LayerNorm(256)
    def forward(self,x,lengths):
        n=x.shape[1]
        mask=torch.arange(n,device=x.device)[None,:]>=lengths.clamp(max=n)[:,None]
        x=self.transformer(self.pos_encoding(self.input_proj(x)),
            src_key_padding_mask=mask)
        valid=(~mask).unsqueeze(-1).float()
        return self.norm((x*valid).sum(1)/valid.sum(1).clamp_min(1e-6))


class GatedFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_net=nn.Sequential(nn.Linear(512,256),nn.GELU(),
            nn.Dropout(.1),nn.Linear(256,256))
        nn.init.constant_(self.gate_net[-1].bias,float(np.log(.8/.2)))
        self.proj=nn.Sequential(nn.Linear(256,256),nn.GELU(),
            nn.Dropout(.15),nn.LayerNorm(256))
    def forward(self,s,t):
        gate=torch.sigmoid(self.gate_net(torch.cat([s,t],dim=-1)))
        return self.proj(gate*s+(1-gate)*t)


class _ReverseGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,scale):
        ctx.scale=scale
        return x.view_as(x)
    @staticmethod
    def backward(ctx,gradient):
        return -ctx.scale*gradient,None


class ArchitectureModel(nn.Module):
    def __init__(self,key,n_speakers):
        super().__init__()
        if key not in ARCH_KEYS:raise ValueError(key)
        self.key=key
        if key!="stream2":self.spectro_encoder=SpectroTemporalEncoder()
        if key!="stream1":self.traj_encoder=TrajectoryEncoder()
        if key.startswith("dual"):self.fusion=GatedFusion()
        if key.startswith("dual"):
            self.speaker_head=nn.Sequential(nn.Linear(256,128),nn.GELU(),
                nn.Dropout(.2),nn.Linear(128,n_speakers))
        self.dysarthria_head=nn.Sequential(nn.Linear(256,128),
            nn.BatchNorm1d(128),nn.GELU(),nn.Dropout(.3),nn.Linear(128,2))
    def embed(self,logmel,traj,lengths):
        if self.key=="stream1":return self.spectro_encoder(logmel,lengths)
        if self.key=="stream2":return self.traj_encoder(traj,lengths)
        return self.fusion(self.spectro_encoder(logmel,lengths),
            self.traj_encoder(traj,lengths))
    def forward(self,logmel,traj,lengths):
        return self.dysarthria_head(self.embed(logmel,traj,lengths))
    def speaker_logits(self,embedding):
        return self.speaker_head(_ReverseGradient.apply(embedding,.02))


def _augment(logmel,traj,rng):
    """Identical cached-feature augmentation policy for all four candidates."""
    lm=np.array(logmel,copy=True)
    tr=np.array(traj,copy=True)
    for i in range(len(lm)):
        if rng.random()<.5:
            t0=int(rng.integers(0,128));tw=int(rng.integers(0,21))
            lm[i,:,t0:min(128,t0+tw)]=lm[i].mean()
        if rng.random()<.5:
            f0=int(rng.integers(0,80));fw=int(rng.integers(0,16))
            lm[i,f0:min(80,f0+fw),:]=lm[i].mean()
        if rng.random()<.5:
            tr[i]+=rng.normal(0,.015,tr[i].shape).astype(np.float32)
    return lm,tr


def _batch(meta,logmels,trajs,indices,dev,augment=None):
    lm=logmels[indices]
    tr=trajs[indices]
    if augment is not None:lm,tr=_augment(lm,tr,augment)
    return (torch.as_tensor(lm,device=dev).unsqueeze(1),
        torch.as_tensor(tr,device=dev),
        torch.as_tensor(meta.iloc[indices].valid_frames.to_numpy(int).copy(),device=dev),
        torch.as_tensor(meta.iloc[indices].label.to_numpy(int).copy(),device=dev))


def _predict(model,meta,logmels,trajs,indices,batch_size=48):
    dev=device();model.eval();out=[]
    with torch.no_grad():
        for start in range(0,len(indices),batch_size):
            lm,tr,ln,_=_batch(meta,logmels,trajs,indices[start:start+batch_size],dev)
            out.append(torch.softmax(model(lm,tr,ln),1)[:,1].cpu().numpy())
    return np.concatenate(out)


def _supcon(z,labels,temperature=.07):
    z=F.normalize(z,dim=1)
    n=len(z)
    logits=z@z.T/temperature
    logits=logits-logits.max(dim=1,keepdim=True).values.detach()
    own=torch.eye(n,dtype=torch.bool,device=z.device)
    positive=labels[:,None].eq(labels[None,:]) & ~own
    exp=torch.exp(logits).masked_fill(own,0)
    logprob=logits-torch.log(exp.sum(1,keepdim=True)+1e-12)
    good=positive.any(1)
    return -(positive[good].float()*logprob[good]).sum(1).div(
        positive[good].sum(1).clamp_min(1)).mean()


def _pretrain(model,meta,logmels,trajs,indices,seed,history_path):
    """V6-style supervised contrastive pretraining on training speakers only."""
    dev=device();opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    rng=np.random.default_rng(seed+30000)
    history=[];start_epoch=1
    checkpoint=history_path.with_suffix(".pt")
    if checkpoint.exists():
        saved=torch.load(checkpoint,map_location="cpu",weights_only=False)
        if saved.get("signature")!=ABLATION_SIGNATURE or saved.get("seed")!=seed or\
                saved.get("train_speakers")!=sorted(set(meta.iloc[indices].physical_speaker_id)):
            raise RuntimeError(f"Stale architecture pretraining checkpoint: {checkpoint}")
        model.load_state_dict(saved["model_state_dict"])
        opt.load_state_dict(saved["optimizer_state_dict"])
        rng.bit_generator.state=saved["numpy_rng_state"]
        torch.random.set_rng_state(saved["torch_rng_state"])
        if dev.type=="mps" and saved.get("mps_rng_state") is not None:
            torch.mps.set_rng_state(saved["mps_rng_state"])
        history=saved["history"]
        start_epoch=int(saved["last_epoch"])+1
    for epoch in range(start_epoch,PRETRAIN_EPOCHS+1):
        model.train();order=_speaker_balanced_indices(meta,indices,rng,SAMPLES_PER_EPOCH)
        losses=[]
        for start in range(0,len(order),32):
            ix=order[start:start+32]
            lm1,tr1,ln,y=_batch(meta,logmels,trajs,ix,dev,augment=rng)
            lm2,tr2,_,_=_batch(meta,logmels,trajs,ix,dev,augment=rng)
            z=torch.cat([model.embed(lm1,tr1,ln),model.embed(lm2,tr2,ln)])
            loss=_supcon(z,torch.cat([y,y]))
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5)
            opt.step();losses.append(float(loss.detach().cpu()))
        history.append(dict(epoch=epoch,train_supcon_loss=np.mean(losses)))
        pd.DataFrame(history).to_csv(history_path,index=False)
        torch.save(dict(signature=ABLATION_SIGNATURE,seed=seed,
            train_speakers=sorted(set(meta.iloc[indices].physical_speaker_id)),
            last_epoch=epoch,history=history,
            model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},
            optimizer_state_dict=opt.state_dict(),
            numpy_rng_state=rng.bit_generator.state,
            torch_rng_state=torch.random.get_rng_state(),
            mps_rng_state=torch.mps.get_rng_state() if dev.type=="mps" else None),
            checkpoint)


def _train(model,meta,logmels,trajs,train,val,seed,epochs,history_path=None):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    dev=device();model=model.to(dev)
    opt=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=1e-4)
    rng=np.random.default_rng(seed);best=float("inf");best_epoch=epochs
    state=None;stale=0;history=[]
    speaker_map={speaker:i for i,speaker in enumerate(sorted(
        set(meta.iloc[train].physical_speaker_id)))}
    for epoch in range(1,epochs+1):
        model.train();order=_speaker_balanced_indices(meta,train,rng,SAMPLES_PER_EPOCH)
        losses=[]
        for start in range(0,len(order),32):
            lm,tr,ln,y=_batch(meta,logmels,trajs,order[start:start+32],dev,augment=rng)
            emb=model.embed(lm,tr,ln)
            loss=F.cross_entropy(model.dysarthria_head(emb),y)
            if model.key.startswith("dual"):
                speaker_labels=torch.as_tensor([speaker_map[s] for s in
                    meta.iloc[order[start:start+32]].physical_speaker_id],device=dev)
                loss=loss+.02*F.cross_entropy(model.speaker_logits(emb),speaker_labels)
            opt.zero_grad();loss.backward();opt.step()
            losses.append(float(loss.detach().cpu()))
        if val is not None:
            p=_predict(model,meta,logmels,trajs,val)
            y=meta.iloc[val].label.to_numpy(int)
            sp=meta.iloc[val].physical_speaker_id.to_numpy()
            counts=pd.Series(sp).value_counts()
            w=np.array([1/counts[s] for s in sp]);w/=w.sum()
            ll=float(-np.sum(w*(y*np.log(np.clip(p,1e-7,1))+
                (1-y)*np.log(np.clip(1-p,1e-7,1)))))
            if ll<best-1e-5:
                best=ll;best_epoch=epoch
                state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                stale=0
            else:stale+=1
            history.append(dict(epoch=epoch,train_loss=np.mean(losses),val_loss=ll,
                val_roc_auc=roc_auc_score(y,p) if len(np.unique(y))==2 else np.nan,
                val_f1_at_0_5=f1_score(y,p>=.5,zero_division=0),
                best_epoch_so_far=best_epoch,stale_epochs=stale))
            if history_path is not None:pd.DataFrame(history).to_csv(history_path,index=False)
            if stale>=4:break
    if state is not None:model.load_state_dict(state)
    return model,best_epoch


def _fit_one(meta,logmels,trajs,train,val,seed,key,name,epochs):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    model=ArchitectureModel(key,meta.iloc[train].physical_speaker_id.nunique()).to(device())
    if key=="dual_full":
        _pretrain(model,meta,logmels,trajs,train,seed,
            OUT/f"{name}_pretraining_history.csv")
    return _train(model,meta,logmels,trajs,train,val,seed,epochs,
        OUT/f"{name}_finetune_history.csv" if val is not None else None)


def run_architecture_ablation(seeds=SEEDS,keys=ARCH_KEYS):
    """Resumable five-seed/five-fold V7 architecture ablation."""
    budget=json.loads((OUT/"v7_training_budget_decision.json").read_text())
    if budget["selected_max_epochs"]!=SUPERVISED_MAX_EPOCHS:
        raise RuntimeError("Ablation budget must match TORGO convergence decision")
    meta,logmels,cache=load_torgo()
    meta,_=audit_physical_speakers(meta,save=False)
    z=np.load(cache/"dual_stream_features.npz",allow_pickle=True)
    trajs=z["trajs"]
    assert len(trajs)==len(meta)
    all_predictions=[]
    for seed in seeds:
        for fold,outer_train,outer_test in grouped_splits(meta,seed):
            for key in keys:
                base=OUT/f"v7_arch_{key}_seed{seed}_fold{fold}"
                path=base.with_suffix(".csv")
                if path.exists():
                    pred=pd.read_csv(path)
                    assert set(pred.file_path)==set(meta.iloc[outer_test].file_path)
                    all_predictions.append(pred)
                    continue
                local=meta.iloc[outer_train].reset_index(drop=True)
                oof=[];selected=[]
                for inner,tri,vai in grouped_splits(local,seed+731,n_splits=3):
                    train,val=outer_train[tri],outer_train[vai]
                    assert_disjoint(meta,train,val,outer_test)
                    model,epoch=_fit_one(meta,logmels,trajs,train,val,seed+inner,key,
                        f"v7_arch_{key}_seed{seed}_fold{fold}_inner{inner}",
                        SUPERVISED_MAX_EPOCHS)
                    raw=_predict(model,meta,logmels,trajs,val)
                    oof.append(pd.DataFrame(dict(index=val,label=meta.iloc[val].label.to_numpy(int),
                        physical_speaker_id=meta.iloc[val].physical_speaker_id.to_numpy(),
                        probability_raw=raw)))
                    selected.append(epoch);del model
                inner_oof=pd.concat(oof,ignore_index=True)
                assert len(inner_oof)==len(outer_train) and inner_oof["index"].is_unique
                temp=calibrate_temperature(inner_oof.label,inner_oof.probability_raw,
                    inner_oof.physical_speaker_id)
                inner_oof["probability"]=apply_temperature(inner_oof.probability_raw,temp)
                policy=threshold_table(inner_oof.label,inner_oof.probability,
                    inner_oof.physical_speaker_id)
                threshold=float(policy.set_index("policy").loc[
                    "Specificity_at_sensitivity_0.90","threshold"])
                final_epoch=int(np.median(selected))
                model,_=_fit_one(meta,logmels,trajs,outer_train,None,seed+10000,key,
                    f"v7_arch_{key}_seed{seed}_fold{fold}_final",final_epoch)
                raw=_predict(model,meta,logmels,trajs,outer_test)
                pred=meta.iloc[outer_test][["file_path","physical_speaker_id",
                    "label","gender","mic_type"]].copy()
                pred["probability_raw"]=raw
                pred["probability"]=apply_temperature(raw,temp)
                pred["threshold"]=threshold
                pred["temperature"]=temp
                pred["seed"]=seed
                pred["outer_fold"]=fold
                pred["model"]=ARCH_NAMES[ARCH_KEYS.index(key)]
                pred["ablation_signature"]=ABLATION_SIGNATURE
                pred.to_csv(path,index=False)
                inner_oof.to_csv(base.with_suffix(".inner_oof.csv"),index=False)
                policy.to_csv(base.with_suffix(".threshold_policies.csv"),index=False)
                torch.save(dict(signature=ABLATION_SIGNATURE,seed=seed,fold=fold,key=key,
                    train_physical_speakers=sorted(set(meta.iloc[outer_train].physical_speaker_id)),
                    test_physical_speakers=sorted(set(meta.iloc[outer_test].physical_speaker_id)),
                    selected_epochs=final_epoch,temperature=temp,threshold=threshold,
                    model_state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()}),
                    base.with_suffix(".pt"))
                all_predictions.append(pred)
                print("Architecture",key,seed,fold,flush=True)
    predictions=pd.concat(all_predictions,ignore_index=True)
    predictions.to_csv(OUT/"v7_corrected_architecture_predictions.csv",index=False)
    rows=[]
    for (name,seed),part in predictions.groupby(["model","seed"]):
        assert part.file_path.is_unique and part.physical_speaker_id.nunique()==15
        speaker=part.groupby("physical_speaker_id",as_index=False).agg(
            label=("label","first"),probability=("probability","mean"),
            threshold=("threshold","first"))
        for level,frame in (("utterance",part),("physical_speaker",speaker)):
            rows.append(dict(model=name,seed=seed,level=level,
                **_variable_threshold_metrics(frame)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"v7_corrected_architecture_ablation.csv",index=False)
    return result
