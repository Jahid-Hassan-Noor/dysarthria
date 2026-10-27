# Dysarthria V7 methodology repair

Open `dysarthria_v7_methodology_fixed.ipynb` from the project root. Its first 112 cells are the original V6 notebook, including the historical V6 results. Run only the appended V7 section in a fresh project environment. The appendix imports `v7.methodology`, `v7.external`, and `v7.figures`; all new outputs are under `v7/outputs`.

For a resumable command-line run from the project root:

```sh
.venv/bin/python -m v7.run_v7
```

The TORGO mapping audit is `outputs/torgo_physical_speaker_mapping_audit.csv`; exact grouped speaker membership is saved in each seed/fold `split_manifest.json`. Five all-TORGO deployment checkpoints are in `outputs/deployment_seed42` through `outputs/deployment_seed46`. Their thresholds and calibration parameters are TORGO-derived and locked before official UASpeech inference.

The earlier V7 external result is the five deployment-checkpoint zero-shot evaluation on prespecified UASpeech M3. `outputs/official_per_seed_metrics.csv` and `outputs/official_five_seed_mean_speaker_cluster_ci.csv` contain its per-seed and physical-speaker summaries. These results predate the full corrected comparison and are historical until final TORGO model selection. `outputs/official_zero_shot_audit.json` records source hashes and the absence of UASpeech training, fine-tuning, threshold selection, calibration, or model selection.

TORGO calibration diagnostics are split into `outputs/torgo_grouped_validation_calibration_comparison.csv` for the inner OOF scores used to fit temperature and `outputs/torgo_outer_calibration_comparison.csv` for untouched outer-test speakers. Use the latter to assess calibration transfer without fitting-data optimism.

The direct TORGO-versus-UASpeech figures instead use the same prespecified outer-fold-0 TORGO checkpoint on both corpora for each seed. The exact pairs and checkpoint hashes are in `outputs/paired_identical_checkpoint_metrics.csv`. These paired-fold figures are separate from the primary all-TORGO deployment-model evaluation; do not combine their estimates as if they were one model fit. An exploratory seed-only comparison with nonidentical fits is retained under `outputs/historical_mismatched_seed_comparison` for audit history and is excluded from publication figures.

The historical V7 figure index is `outputs/publication_figure_index.csv`. Its 17 rows include 16 topics plus a diagnostic log-scale probability view labeled Figure 7b. Every figure is in `outputs/publication_figures` as a 300-DPI PNG and vector PDF. These are pre-correction figures and are excluded from the corrected figure gate. The earlier printed summary is in `outputs/final_master_report.txt`.

## Corrected research extension (in the same V7 notebook)

The final cells of `dysarthria_v7_methodology_fixed.ipynb` contain the current corrected research work. Earlier V6 and V7 outputs remain historical evidence. `v7.corrected_analysis` writes additive `outputs/v7_*` files and never replaces existing checkpoints or results.

`outputs/v7_pooled_oof_torgo_predictions.csv` contains the original V7 physical-speaker-safe outer predictions, checked against every split manifest and checkpoint. `outputs/v7_pooled_oof_torgo_speakers.csv`, `outputs/v7_pooled_oof_speaker_metrics.csv`, `outputs/v7_pooled_oof_mean_sd.csv`, and `outputs/v7_pooled_oof_speaker_cluster_ci.csv` derive from those predictions. Each of 15 TORGO speakers appears in exactly one outer-test fold per seed.

`outputs/v7_corrected_threshold_policy_validation_summary.csv` compares the four prespecified policies using grouped TORGO validation. `outputs/v7_corrected_threshold_policy_outer_metrics.csv` and its summary report their held-out sensitivity analysis. The sensitivity-constrained policy remains primary throughout; these test-set comparisons never choose an external threshold.
`outputs/v7_corrected_fold_subgroup_metrics.csv` keeps one-class fold ROC-AUC as missing, and `outputs/v7_corrected_valid_fold_subgroup_summary.csv` averages only folds containing both diagnoses. The four gender-by-diagnosis physical-speaker counts are in the pooled subgroup file.

The original V7 checkpoint bundles did not save per-epoch inner-fold histories. The completed replay recorded all 75 grouped inner histories in new files. `outputs/v7_convergence_summary.csv` and `outputs/v7_training_budget_decision.json` show that the 20-epoch maximum was not reached, though many validation curves remain unstable. The controlled session-versus-physical split diagnostic is complete across five seeds; `outputs/v7_split_protocol_per_seed_metrics.csv`, `outputs/v7_split_protocol_comparison.csv`, and `outputs/v7_split_protocol_paper_table.csv` retain its results and signed protocol difference. The session arm is a diagnostic, not a primary estimate. `outputs/v7_final_research_audit.json` is evidence gated and reports `NO` for any requested experiment or final figure set that has not actually finished.

The completed SVM/RF/stacking rerun is saved in `outputs/v7_corrected_classical_predictions.csv` and `outputs/v7_corrected_classical_metrics.csv`. All 75 classical training-only grouped inner OOF files have been backfilled and indexed in `outputs/v7_corrected_classical_inner_oof_index.csv`; each is checked against its untouched outer-test fold. `outputs/v7_interim_physical_speaker_baseline_results.csv` adds the existing V7 Stream-1 pooled estimate for a clearly marked four-model interim comparison. It is not the requested final seven-model table. `outputs/v7_corrected_classical_calibration_comparison.csv` preserves both raw and temperature-scaled outer-test results. The `outputs/v7_corrected_publication_figures` directory contains only figures traceable to the corrected files listed in its index; the audit withholds a final-figure PASS until all requested figure topics exist.

The notebook's corrected architecture cell calls `v7.architecture_ablation` on the same V7 physical-speaker folds. Its four candidates use one matched cached-feature augmentation and the same supervised epoch/sample cap. This is a fresh, TORGO-only matched ablation; it does not relabel the older waveform-augmented Stream-1 fits. The full dual-stream candidate additionally receives training-speaker-only contrastive pretraining, so its total optimization steps are higher by design and this difference is disclosed in the audit. Dual-stream pretraining checkpoints and histories are resumable and stay within `v7/outputs`. The architecture comparison remains incomplete until all four candidates and five seeds have actual prediction files.

The V7 Stream-1 deployment model is prespecified for external evaluation; the architecture ablation describes alternatives without selecting an external model from UASpeech outcomes. `outputs/v7_final_model_lock.json` records all five TORGO-only checkpoint hashes, temperatures, and thresholds. `v7.final_official` performs fresh inference with those frozen checkpoints and existing official UASpeech metadata/channel protocol, writing new `v7_final_official_*` prediction, metric, CI, matched-channel, and zero-shot audit files while preserving every historical external file.

The official corpus was already evaluated in historical V7. The new final files document a reproducibility rerun of the same frozen deployment models, not a newly acquired prospective external cohort. No UASpeech labels are used for training, fine-tuning, calibration, threshold selection, architecture choice, or budget changes.
`outputs/v7_final_official_historical_reproduction.csv` compares every one of the 35 channel-by-seed fresh prediction sets with the preserved historical files and records the maximum absolute difference.

`outputs/v7_result_traceability_audit.json` is regenerated from saved prediction, metric, table, and figure files. A `PASS` means the numerical metric rows were recomputed from their prediction files and each figure has the source files named in its index; `INCOMPLETE` means the corresponding experiment has not yet finished.

After the architecture run is complete, `python -m v7.materialize_corrected_notebook_outputs` executes only the appended corrected cells and writes their outputs back into this same notebook. It checks that the first 123 historical cells are unchanged as notebook data before writing. The saved corrected CSVs and checkpoint files are the sources for the displayed tables and figures.
