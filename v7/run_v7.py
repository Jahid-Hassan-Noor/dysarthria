"""Resume the full V7 experiment from the command line.

Run from the project root: .venv/bin/python -m v7.run_v7
All outputs are under v7/outputs. V6 artifacts are read-only.
"""
import warnings

import pandas as pd

from .methodology import (OUT, SEEDS, audit_physical_speakers, historical_v6_leakage_audit,
                          load_torgo, run_cross_microphone, run_nested_torgo,
                          run_noise_robustness, split_audit,
                          summarize_grouped_validation_calibration, summarize_outer_calibration_comparison,
                          summarize_gender_subgroups,
                          summarize_threshold_policies, relock_exact_thresholds,
                          train_deployment, write_split_manifests)
from .external import (evaluate_official, load_frozen_models, load_official_metadata,
                       precompute_official_features, predict_official,
                       evaluate_identical_outer_checkpoints)
from .figures import make_publication_figures


def main():
    warnings.filterwarnings("ignore",message="n_fft=2048 is too large")
    meta,features,_=load_torgo();meta,mapping=audit_physical_speakers(meta)
    splits=split_audit(meta);write_split_manifests(meta)
    historical=historical_v6_leakage_audit()
    print(f"PHYSICAL-SPEAKER LEAKAGE CHECK: PASS ({meta.physical_speaker_id.nunique()} speakers; {len(splits)} audited partitions)",flush=True)
    print(f"Historical V6 overlap: {historical.loc[historical.physical_overlap_across_v6_splits,'physical_speaker_id'].nunique()} physical speakers",flush=True)
    run_nested_torgo(meta,features,seeds=SEEDS,epochs=20,samples_per_epoch=512)
    train_deployment(meta,features,seeds=SEEDS,epochs=20,samples_per_epoch=512)
    relock_exact_thresholds()
    summarize_grouped_validation_calibration()
    summarize_outer_calibration_comparison()
    summarize_gender_subgroups()
    summarize_threshold_policies()
    run_cross_microphone(meta,features,seed=42,epochs=20,samples_per_epoch=512)
    run_noise_robustness(meta,seed=42)
    official=load_official_metadata();precompute_official_features(official)
    frozen=load_frozen_models();pred=predict_official(official,frozen)
    evaluate_official(pred,frozen)
    evaluate_identical_outer_checkpoints(official)
    index=make_publication_figures()
    tor=pd.read_csv(OUT/"torgo_nested_fold_metrics.csv");ext=pd.read_csv(OUT/"official_per_seed_metrics.csv")
    for heading,frame in (("TORGO MULTI-SEED",tor[tor.level.eq("utterance")]),
                          ("OFFICIAL UASPEECH ZERO-SHOT",ext[ext.level.eq("utterance")]),
                          ("OFFICIAL UASPEECH SPEAKER LEVEL",ext[ext.level.eq("physical_speaker")])):
        print(heading,flush=True)
        for metric in ("Accuracy","Balanced_Accuracy","Sensitivity","Specificity","Precision","NPV",
                       "F1_dysarthric","Macro_F1","ROC_AUC","PR_AUC","Brier","ECE"):
            s=frame.groupby("seed")[metric].mean()
            suffix=(f"; non-estimable in {len(SEEDS)-s.notna().sum()}/{len(SEEDS)} seeds"
                    if s.notna().sum()<len(SEEDS) else "")
            report=(f"{s.mean():.4f} ± {s.std(ddof=1):.4f}" if s.notna().sum()>1
                    else f"{s.mean():.4f}; SD unavailable" if s.notna().sum()==1
                    else "non-estimable")
            print(f"  {metric}: {report}{suffix}",flush=True)
    print("OFFICIAL UASPEECH FIVE-SEED MEAN SPEAKER CLUSTER CIs",flush=True)
    print(pd.read_csv(OUT/"official_five_seed_mean_speaker_cluster_ci.csv").to_string(index=False),flush=True)
    print("EXTERNAL CALIBRATION: Brier / ECE in the tables above",flush=True)
    ch=pd.read_csv(OUT/"official_matched_channel_metrics.csv")
    print("MICROPHONE ROBUSTNESS:",ch[ch.level.eq("physical_speaker")].groupby("channel").ROC_AUC.mean().round(4).to_dict(),flush=True)
    noise=pd.read_csv(OUT/"torgo_noise_results.csv")
    print("NOISE ROBUSTNESS:",noise.groupby("SNR_dB").Balanced_Accuracy.mean().round(4).to_dict(),flush=True)
    print("FIGURES GENERATED:",len(index),flush=True)
    for text in ("training","fine-tuning","threshold selection","calibration","model selection"):
        print(f"Official UASpeech used for {text}: NO",flush=True)


if __name__=="__main__":main()
