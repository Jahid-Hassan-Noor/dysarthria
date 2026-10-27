"""Publication figures computed only from V7 result tables, never constants."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score, roc_curve)

from .methodology import OUT, SEEDS, ece, speaker_aggregate

FIG = OUT / "publication_figures"
FIG.mkdir(exist_ok=True)
BLUE="#245A81"; RED="#C45C47"; GOLD="#C49A36"; GREY="#65717C"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.titlesize":12,
                     "axes.labelsize":10,"figure.figsize":(7,5),"axes.spines.top":False,
                     "axes.spines.right":False,"savefig.facecolor":"white",
                     "pdf.fonttype":42,"ps.fonttype":42})


def _save(fig, filename, experiment, level, metric, purpose, section, index):
    fig.tight_layout()
    fig.savefig(FIG/(filename+".png"),dpi=300,bbox_inches="tight")
    fig.savefig(FIG/(filename+".pdf"),bbox_inches="tight")
    plt.close(fig)
    index.append({"Figure":len(index)+1,"Filename":filename+".png; "+filename+".pdf","Experiment":experiment,
                  "Level":level,"Metric(s)":metric,"Purpose":purpose,"Placement":section})


def _mean_sd(df,metric):
    x=df.groupby("seed")[metric].mean()
    return float(x.mean()),float(x.std(ddof=1))


def _reliability(ax,y,p,title):
    observed,predicted=calibration_curve(y,p,n_bins=10,strategy="uniform")
    ax.plot([0,1],[0,1],"--",color=GREY,label="Perfect calibration")
    ax.plot(predicted,observed,"o-",color=BLUE,label="Observed")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Mean predicted dysarthria probability",
           ylabel="Observed dysarthria rate",title=title)
    ax.text(.03,.97,f"ECE {ece(y,p):.3f}\nBrier {np.mean((np.asarray(p)-np.asarray(y))**2):.3f}",
            transform=ax.transAxes,va="top",bbox=dict(facecolor="white",alpha=.85,edgecolor="none"))
    ax.legend(fontsize=8,loc="lower right")


def make_publication_figures():
    required=["torgo_nested_fold_metrics.csv","torgo_nested_predictions.csv",
              "official_per_seed_metrics.csv","official_speaker_cluster_ci.csv",
              "official_all_channel_predictions.csv","official_matched_channel_metrics.csv",
              "paired_identical_checkpoint_metrics.csv","paired_outer_official_predictions.csv"]
    missing=[n for n in required if not (OUT/n).is_file()]
    if missing:raise RuntimeError(f"V7 results missing; figures withheld: {missing}")
    tor=pd.read_csv(OUT/required[0]);tor_pred=pd.read_csv(OUT/required[1]);
    ext=pd.read_csv(OUT/required[2]);ci=pd.read_csv(OUT/required[3]);
    allpred=pd.read_csv(OUT/required[4],low_memory=False);channels=pd.read_csv(OUT/required[5])
    paired=pd.read_csv(OUT/required[6]);paired_pred=pd.read_csv(OUT/required[7],low_memory=False)
    if set(ext.seed)!=set(SEEDS) or set(tor.seed)!=set(SEEDS):
        raise RuntimeError("Five matched V7 seeds are required for publication figures")
    primary=allpred[allpred.microphone.eq("M3")]
    toru=tor[tor.level.eq("utterance")];extu=ext[ext.level.eq("utterance") & ext.channel.eq("M3")]
    if set(paired.seed)!=set(SEEDS):raise RuntimeError("Five identical-checkpoint comparison rows required")
    torpair=paired[["seed"]+[c for c in paired if c.startswith("TORGO_")]].rename(
        columns={c:c.removeprefix("TORGO_") for c in paired if c.startswith("TORGO_")})
    extpair=paired[["seed"]+[c for c in paired if c.startswith("Official_UASpeech_")]].rename(
        columns={c:c.removeprefix("Official_UASpeech_") for c in paired if c.startswith("Official_UASpeech_")})
    index=[]

    # 1: identical fold-zero checkpoint on both domains; fixed before external outcomes.
    values=[("Accuracy","Accuracy"),("Balanced_Accuracy","Balanced Accuracy"),
            ("F1_dysarthric","F1"),("ROC_AUC","ROC-AUC"),("PR_AUC","PR-AUC")]
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(len(values));width=.37
    for offset,df,label,color in ((-width/2,torpair,"TORGO held-out speakers",BLUE),(width/2,extpair,"Official UASpeech zero-shot",RED)):
        means,sds=zip(*[_mean_sd(df,m) for m,_ in values])
        ax.bar(x+offset,means,width,yerr=sds,capsize=3,label=label,color=color)
    ax.set(xticks=x,xticklabels=[n for _,n in values],ylim=(0,1),ylabel="Metric value",
           title="Identical frozen models: TORGO vs official UASpeech (outer fold 0)")
    ax.legend();_save(fig,"fig_external_torgo_vs_uaspeech","TORGO vs official UASpeech; identical outer-fold-0 checkpoints","utterance",
                       "Accuracy; Balanced Accuracy; F1; ROC-AUC; PR-AUC","Paired checkpoint external performance", "Main",index)

    # 2: seed curves plus interpolated mean on common FPR grid.
    fig,ax=plt.subplots(figsize=(6,5.5));grid=np.linspace(0,1,201);curves=[];aucs=[]
    for seed in SEEDS:
        p=primary[f"prob_seed{seed}"].to_numpy();y=primary.label.to_numpy();fpr,tpr,_=roc_curve(y,p)
        auc=roc_auc_score(y,p);aucs.append(auc);curves.append(np.interp(grid,fpr,tpr))
        ax.plot(fpr,tpr,color=BLUE,alpha=.18,lw=1,label="Individual seeds" if seed==SEEDS[0] else None)
    ax.plot(grid,np.mean(curves,axis=0),color=BLUE,lw=2.8,label=f"Mean across seeds (ROC-AUC {np.mean(aucs):.3f})")
    ax.plot([0,1],[0,1],"--",color=GREY,label="Chance")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="False positive rate",ylabel="True positive rate",
           title="Official UASpeech zero-shot ROC — M3");ax.legend()
    _save(fig,"fig_external_roc","Official UASpeech zero-shot","utterance","ROC-AUC","Discrimination","Main",index)

    # 3: PR curves; mean precision interpolated at a common recall grid.
    fig,ax=plt.subplots(figsize=(6,5.5));grid=np.linspace(0,1,201);curves=[];aps=[]
    for seed in SEEDS:
        p=primary[f"prob_seed{seed}"].to_numpy();y=primary.label.to_numpy();precision,recall,_=precision_recall_curve(y,p)
        aps.append(average_precision_score(y,p));curves.append(np.interp(grid,recall[::-1],precision[::-1]))
        ax.plot(recall,precision,color=RED,alpha=.18,lw=1,label="Individual seeds" if seed==SEEDS[0] else None)
    ax.plot(grid,np.mean(curves,axis=0),color=RED,lw=2.8,label=f"Mean across seeds (AP {np.mean(aps):.3f})")
    ax.axhline(primary.label.mean(),ls="--",color=GREY,label="Prevalence / no skill")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="Recall",ylabel="Precision",title="Official UASpeech zero-shot PR — M3")
    ax.legend();_save(fig,"fig_external_pr","Official UASpeech zero-shot","utterance","PR-AUC / Average Precision",
                      "Positive-class precision-recall","Supplementary",index)

    # Figures with a single operating point use seed 42, declared before results.
    seed=42;col=f"prob_seed{seed}";thr=float(extu.set_index("seed").loc[seed,"threshold"])
    y=primary.label.to_numpy(int);p=primary[col].to_numpy(float)
    cm=confusion_matrix(y,p>=thr,labels=[0,1]);pct=cm/cm.sum(axis=1,keepdims=True)
    fig,ax=plt.subplots(figsize=(5.5,5));ax.imshow(pct,cmap="Blues",vmin=0,vmax=1)
    for i in range(2):
        for j in range(2):ax.text(j,i,f"{cm[i,j]:,}\n{pct[i,j]:.1%}",ha="center",va="center",color="black")
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Control","Dysarthric"],yticklabels=["Control","Dysarthric"],
           xlabel="Predicted",ylabel="Actual",title="Official UASpeech Zero-Shot — Locked TORGO Threshold\nSeed 42, M3")
    _save(fig,"fig_external_confusion_matrix","Official UASpeech zero-shot","utterance","Confusion matrix",
          "Locked threshold operating point","Main",index)

    # 5 sensitivity/specificity/BA.
    vals=[("Sensitivity","Sensitivity"),("Specificity","Specificity"),("Balanced_Accuracy","Balanced Accuracy")]
    fig,ax=plt.subplots(figsize=(7,5));x=np.arange(3);w=.37
    for off,df,label,color in ((-w/2,torpair,"TORGO",BLUE),(w/2,extpair,"Official UASpeech",RED)):
        mean,sd=zip(*[_mean_sd(df,m) for m,_ in vals]);ax.bar(x+off,mean,w,yerr=sd,capsize=3,label=label,color=color)
    ax.set(xticks=x,xticklabels=[v for _,v in vals],ylim=(0,1),ylabel="Metric value",
           title="Operating-point shift in identical frozen models (outer fold 0)");ax.legend()
    _save(fig,"fig_external_sensitivity_specificity","TORGO vs official UASpeech; identical outer-fold-0 checkpoints","utterance",
          "Sensitivity; Specificity; Balanced Accuracy","Operating-point shift","Main",index)

    # 6 calibration, using predictions as evaluated. No UASpeech recalibration.
    tor42=tor_pred[tor_pred.seed.eq(42) & tor_pred.outer_fold.eq(0)]
    pair_y=paired_pred.label.to_numpy(int);pair_p=paired_pred.prob_seed42.to_numpy(float)
    fig,axs=plt.subplots(1,2,figsize=(11,4.8))
    _reliability(axs[0],tor42.label,tor42.probability,"TORGO outer-test speakers, seed 42, fold 0")
    _reliability(axs[1],pair_y,pair_p,"Official UASpeech M3, same frozen model")
    _save(fig,"fig_external_calibration","TORGO vs official UASpeech; identical outer-fold-0 checkpoint","utterance","Brier; ECE",
          "Calibration transfer without external refit","Main",index)

    # 7 probability distributions plus diagnostic logarithmic view.
    fig,ax=plt.subplots(figsize=(7,5));bins=np.linspace(0,1,41)
    for label,name,color in ((0,"Control",BLUE),(1,"Dysarthric",RED)):
        ax.hist(p[y==label],bins=bins,alpha=.55,density=True,label=name,color=color)
    ax.axvline(thr,ls="--",color="black",label=f"Locked TORGO threshold {thr:.3g}")
    ax.set(xlim=(0,1),xlabel="Predicted dysarthria probability",ylabel="Density",
           title="Official UASpeech probability by diagnosis — M3, seed 42");ax.legend()
    _save(fig,"fig_external_probability_distribution","Official UASpeech zero-shot","utterance","Probability",
          "Class-score overlap","Supplementary",index)
    fig,ax=plt.subplots(figsize=(7,5));eps=1e-7;bins=np.geomspace(eps,1,50)
    for label,name,color in ((0,"Control",BLUE),(1,"Dysarthric",RED)):
        ax.hist(np.clip(p[y==label],eps,1),bins=bins,alpha=.55,density=False,label=name,color=color)
    ax.axvline(max(thr,eps),ls="--",color="black",label=f"Locked TORGO threshold {thr:.3g}")
    ax.set_xscale("log");ax.set(xlabel="Predicted dysarthria probability (log scale)",ylabel="Utterances",
                              title="Diagnostic log-scale probability view — M3, seed 42");ax.legend()
    _save(fig,"fig_external_probability_distribution_log","Official UASpeech zero-shot","utterance","Probability",
          "Near-zero threshold visibility","Diagnostic",index)

    # 8/9 speaker probabilities and speaker ROC/CI.
    sp=speaker_aggregate(primary[["physical_speaker_id","label",col]].rename(columns={col:"probability"})).sort_values("probability")
    fig,ax=plt.subplots(figsize=(11,5.5));x=np.arange(len(sp))
    ax.bar(x,sp.probability,color=np.where(sp.label.eq(1),RED,BLUE))
    ax.axhline(thr,ls="--",color="black",label="Locked TORGO threshold")
    ax.set(xticks=x,xticklabels=sp.physical_speaker_id,ylim=(0,1),ylabel="Mean predicted dysarthria probability",
           title="Official UASpeech physical-speaker probabilities — M3, seed 42")
    ax.tick_params(axis="x",rotation=65)
    ax.legend(handles=[Patch(facecolor=BLUE,label="Control"),
                       Patch(facecolor=RED,label="Dysarthric"),
                       plt.Line2D([0],[0],color="black",ls="--",label="Locked TORGO threshold")])
    _save(fig,"fig_external_speaker_probabilities","Official UASpeech zero-shot","physical speaker","Mean probability",
          "Patient-level score distribution","Main",index)
    mean_ci=pd.read_csv(OUT/"official_five_seed_mean_speaker_cluster_ci.csv")
    ci_row=mean_ci[mean_ci.metric.eq("ROC_AUC")].iloc[0]
    fig,ax=plt.subplots(figsize=(6,5.5));grid=np.linspace(0,1,201);speaker_curves=[]
    for seed in SEEDS:
        seed_sp=speaker_aggregate(primary[["physical_speaker_id","label",f"prob_seed{seed}"]].rename(
            columns={f"prob_seed{seed}":"probability"})).sort_values("physical_speaker_id")
        fpr,tpr,_=roc_curve(seed_sp.label,seed_sp.probability)
        speaker_curves.append(np.interp(grid,fpr,tpr))
        ax.plot(fpr,tpr,color=BLUE,alpha=.16,lw=1,
                label="Individual frozen seeds" if seed==SEEDS[0] else None)
    ax.plot(grid,np.mean(speaker_curves,axis=0),lw=2.5,color=BLUE,
        label=f"Mean of five seeds: ROC-AUC {ci_row.point_five_seed_mean:.3f}\n95% speaker CI {ci_row.ci_low:.3f}–{ci_row.ci_high:.3f}")
    ax.plot([0,1],[0,1],"--",color=GREY,label="Chance")
    ax.set(xlim=(0,1),ylim=(0,1),xlabel="False positive rate",ylabel="True positive rate",
           title="Official UASpeech physical-speaker ROC — M3, five seeds");ax.legend(fontsize=8)
    _save(fig,"fig_external_speaker_roc","Official UASpeech zero-shot","physical speaker","ROC-AUC; 95% speaker bootstrap CI",
          "Patient-level discrimination","Main",index)

    # 10 severity from official category names, equal weight per speaker.
    sev=pd.read_csv(OUT/"official_severity_descriptive.csv")
    sev=sev[sev.seed.eq(42)].sort_values("mean_speaker_probability")
    fig,ax=plt.subplots(figsize=(7,5));x=np.arange(len(sev))
    ax.bar(x,sev.mean_speaker_probability,yerr=sev.sd_speaker_probability.fillna(0),capsize=4,color=RED)
    ax.set(xticks=x,xticklabels=sev.severity,ylim=(0,1),xlabel="Official severity/intelligibility category",
           ylabel="Mean physical-speaker probability",title="Official UASpeech severity analysis — M3, seed 42")
    _save(fig,"fig_external_severity","Official UASpeech zero-shot","physical speaker","Mean probability by official category",
          "Descriptive severity pattern","Supplementary",index)

    # 11/12 equivalent speaker/content subset across microphones.
    ch=channels[channels.level.eq("physical_speaker")]
    names=[f"M{i}" for i in range(2,9)];mets=[("ROC_AUC","ROC-AUC"),("PR_AUC","PR-AUC"),("Balanced_Accuracy","Balanced Accuracy")]
    fig,ax=plt.subplots(figsize=(9,5));x=np.arange(7);w=.25
    for j,(m,label) in enumerate(mets):
        means=[ch[ch.channel.eq(c)][m].mean() for c in names]
        stds=[ch[ch.channel.eq(c)][m].std(ddof=1) for c in names]
        ax.bar(x+(j-1)*w,means,w,yerr=stds,capsize=2,label=label,color=[BLUE,RED,GOLD][j])
    ax.set(xticks=x,xticklabels=names,ylim=(0,1),xlabel="Microphone channel (matched speaker/content)",
           ylabel="Metric value, mean ± seed SD",title="Official UASpeech channel robustness")
    ax.legend();_save(fig,"fig_external_microphone_robustness","Official UASpeech zero-shot","physical speaker",
                       "ROC-AUC; PR-AUC; Balanced Accuracy","Channel robustness","Supplementary",index)
    means=ch.groupby("channel").ROC_AUC.mean().sort_values(ascending=False)
    fig,ax=plt.subplots(figsize=(7,5));ax.bar(means.index,means.values,color=[BLUE if c=="M3" else GREY for c in means.index])
    ax.set(ylim=(0,1),xlabel="Microphone channel (diagnostic rank)",ylabel="ROC-AUC, five-seed mean",
           title="Official UASpeech per-channel generalization gap")
    _save(fig,"fig_external_channel_gap","Official UASpeech zero-shot","physical speaker","ROC-AUC",
          "Channel rank; M3 remains prespecified primary","Supplementary",index)

    # 13/14 seed stability and paired seed gaps.
    tt=torpair.set_index("seed").reindex(SEEDS)
    ee=extpair.set_index("seed").reindex(SEEDS)
    fig,ax=plt.subplots(figsize=(7,5))
    for seed in SEEDS:ax.plot([0,1],[tt.loc[seed,"ROC_AUC"],ee.loc[seed,"ROC_AUC"]],"o-",alpha=.65,label=f"Seed {seed}")
    ax.set(xticks=[0,1],xticklabels=["TORGO grouped internal","Official UASpeech zero-shot"],ylim=(0,1),
           ylabel="ROC-AUC",title="Identical-checkpoint external stability (outer fold 0)");ax.legend(ncol=2)
    _save(fig,"fig_external_seed_stability","TORGO vs official UASpeech; identical outer-fold-0 checkpoints","utterance","ROC-AUC",
          "Seedwise external shift","Supplementary",index)
    fig,ax=plt.subplots(figsize=(8,5));x=np.arange(len(SEEDS));w=.25
    for j,(m,name) in enumerate([("ROC_AUC","ROC-AUC"),("F1_dysarthric","F1"),("Balanced_Accuracy","Balanced Accuracy")]):
        gap=tt[m]-ee[m];ax.bar(x+(j-1)*w,gap,w,label=name,color=[BLUE,RED,GOLD][j])
    ax.axhline(0,color=GREY,lw=1);ax.set(xticks=x,xticklabels=SEEDS,xlabel="Seed",
        ylabel="TORGO minus official UASpeech",title="Paired checkpoint gaps by seed (outer fold 0)");ax.legend()
    _save(fig,"fig_external_generalization_gap","TORGO vs official UASpeech; identical outer-fold-0 checkpoints","utterance",
          "ROC-AUC; F1; Balanced Accuracy","Paired descriptive gaps","Supplementary",index)

    # 15/16 exist only after the corresponding V7 experiments complete.
    noise_path=OUT/"torgo_noise_results.csv"
    if noise_path.exists():
        noise=pd.read_csv(noise_path)
        fig,ax=plt.subplots(figsize=(7,5))
        for m,color in (("ROC_AUC",BLUE),("Balanced_Accuracy",GOLD),("F1_dysarthric",RED)):
            line=noise.groupby("SNR_dB")[m].mean().reindex([30,20,10,5,0])
            ax.plot(line.index,line.values,"o-",label=m.replace("_"," "),color=color)
        ax.invert_xaxis();ax.set(ylim=(0,1),xlabel="SNR (dB)",ylabel="Metric value",
             title="TORGO held-out physical-speaker noise robustness");ax.legend()
        _save(fig,"fig_external_noise_robustness","TORGO grouped outer test","utterance","ROC-AUC; Balanced Accuracy; F1",
              "Noise robustness","Supplementary",index)
    mic_path=OUT/"torgo_cross_microphone_results.csv"
    if mic_path.exists():
        mic=pd.read_csv(mic_path)
        matrix=mic.pivot_table(index="train_mic",columns="test_mic",values="Balanced_Accuracy",aggfunc="mean").reindex(index=["head","array"],columns=["head","array"])
        fig,ax=plt.subplots(figsize=(5.5,4.5));ax.imshow(matrix,vmin=0,vmax=1,cmap="Blues")
        for i in range(2):
            for j in range(2):ax.text(j,i,f"{matrix.iloc[i,j]:.3f}",ha="center",va="center")
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=["Head","Array"],yticklabels=["Head","Array"],
               xlabel="Test microphone",ylabel="Train microphone",title="TORGO cross-microphone Balanced Accuracy")
        _save(fig,"fig_external_torgo_cross_microphone","TORGO grouped outer test","utterance","Balanced Accuracy",
              "Microphone transfer","Supplementary",index)
    # The extra diagnostic zoom is Figure 7b; requested figure numbers continue.
    if len(index)==17:
        for row,label in zip(index,[*range(1,8),"7b",*range(8,17)]):row["Figure"]=label
    idx=pd.DataFrame(index);idx.to_csv(OUT/"publication_figure_index.csv",index=False)
    print(idx.to_string(index=False));print("FIGURES GENERATED:",len(idx))
    return idx
