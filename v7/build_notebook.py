"""Append V7 methods to V6 without editing or deleting a V6 cell."""
import json
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"v6"/"dysarthria_v6.ipynb"
TARGET=ROOT/"v7"/"dysarthria_v7_methodology_fixed.ipynb"


def cell(kind, source):
    d={"cell_type":kind,"id":uuid.uuid4().hex[:8],"metadata":{},"source":source.splitlines(keepends=True)}
    if kind=="code":d.update(execution_count=None,outputs=[])
    return d


notebook=json.loads(SOURCE.read_text())
original=list(notebook["cells"])
notebook["cells"].extend([
    cell("markdown", """# V7 Methodology Repair — Physical-Speaker-Safe Analysis

This appendix is the V7 research workflow. The 112 preceding V6 cells and their outputs
are retained verbatim as historical experiments. Their session-level split and the V6
seed-42 official UASpeech result are **historical baselines**, not V7 estimates.

Run the V7 cells below from this point in a fresh kernel. They import independent V7
modules from the sibling `v7` directory and never write into `v6/outputs`.
The primary official UASpeech channel remains M3, as specified in V6 before these
new experiments. V7 training, epoch selection, calibration, and threshold selection
use TORGO only. Five TORGO-only seeds (42–46) must exist before official UASpeech
evaluation or publication figures run. Missing results produce an error, never a
fabricated mean or placeholder figure.

Physical-speaker mapping follows the [TORGO corpus directory documentation](https://www.cs.toronto.edu/~complingweb/data/TORGO/torgo.html): speaker codes own separate Session directories. The local flattened mirror is audited against folder names, WAV basenames, labels, gender, and that documented roster. All V7 model fitting is grouped by physical speaker.
"""),
    cell("code", """# V7.1 — audited TORGO identity and grouped split plan (no model fitting)
import sys
from pathlib import Path
_v7_root = Path('/Users/Noor/Documents/projects/dysarthia')
if str(_v7_root) not in sys.path:
    sys.path.insert(0, str(_v7_root))
from v7.methodology import (OUT as V7_OUT, SEEDS as V7_SEEDS, load_torgo,
    audit_physical_speakers, historical_v6_leakage_audit, split_audit,
    write_split_manifests, grouped_splits, assert_disjoint)
v7_meta, v7_logmels, v7_source_cache = load_torgo()
v7_meta, v7_mapping = audit_physical_speakers(v7_meta)
v7_split_audit = split_audit(v7_meta)
v7_historical_leakage = historical_v6_leakage_audit()
write_split_manifests(v7_meta)
print('TORGO physical-speaker mapping: 30 observed session IDs ->', v7_meta.physical_speaker_id.nunique(), 'documented speakers')
print('Audited nested train/validation/test partitions:', len(v7_split_audit))
print('PHYSICAL-SPEAKER LEAKAGE CHECK: PASS')
print('Historical V6 session split physical-speaker leakage: FAIL (',
      v7_historical_leakage.loc[v7_historical_leakage.physical_overlap_across_v6_splits,
                                'physical_speaker_id'].nunique(), 'of',
      v7_historical_leakage.physical_speaker_id.nunique(), 'speakers overlap)')
display(v7_mapping[['session_id','physical_speaker_id','label','gender','n_utterances','microphones']])
display(v7_split_audit)
"""),
    cell("markdown", """## Nested TORGO evaluation and five frozen Stream-1 seeds

Five outer stratified physical-speaker folds provide final internal evaluation.
Within each outer development set, three physical-speaker folds supply out-of-fold
predictions for epoch selection, temperature scaling, and four threshold policies:
F1, Youden J, Balanced Accuracy, and maximum specificity subject to sensitivity
at least 0.90. The last is the prespecified deployment policy. Temperature and
threshold are fitted with equal physical-speaker weights; the final threshold
search evaluates every distinct TORGO validation score boundary exactly. Final outer models train
for the median inner-selected epoch count without viewing outer-test speakers.
All outer models, OOF tables, threshold comparisons, calibration parameters,
speaker memberships, signatures, and predictions are saved separately.
The inner OOF calibration table describes the fitting data; a separate
raw-versus-calibrated Brier/ECE/ROC-AUC/PR-AUC table and reliability plot use
untouched outer-test speakers to assess calibration transfer without refitting.

Each training batch first samples diagnosis, then a physical speaker, then an
utterance. Training-only waveform augmentation uses random gain, additive noise
with the stated 40/15/15/15/10/5 SNR mix, mild time shift, and random microphone
frequency response; a mild gain is also applied after V6's per-utterance
normalization so it remains effective. Log-mel SpecAugment masks time and frequency. Test and
validation audio receive no stochastic augmentation.

Final deployment models are trained on all TORGO speakers for each seed, using
fresh five-fold TORGO-only OOF predictions for their locked thresholds and
temperatures. Their grouped TORGO estimates come from held-out outer models;
the final deployment checkpoints differ from those fold models. To make the
TORGO-vs-UASpeech figures a comparison of **identical model weights**, a separate
predesignated outer-fold-0 checkpoint from each seed is also evaluated unchanged
on official UASpeech. Figures 1, 5, 6, 13, and 14 use these exact checkpoint
pairs. The primary five-seed zero-shot table and remaining external figures
use the all-TORGO deployment checkpoints. Fold 0 was fixed before external
outcomes; it is not chosen for favorable external performance.
"""),
    cell("code", """# V7.2 — long-running, resumable TORGO-only training
from v7.methodology import (run_nested_torgo, train_deployment,
    summarize_grouped_validation_calibration, summarize_outer_calibration_comparison,
    summarize_gender_subgroups,
    summarize_threshold_policies, relock_exact_thresholds)
V7_MAX_EPOCHS = 20
V7_SAMPLES_PER_EPOCH = 512
v7_torgo_folds = run_nested_torgo(v7_meta, v7_logmels, seeds=V7_SEEDS,
    epochs=V7_MAX_EPOCHS, samples_per_epoch=V7_SAMPLES_PER_EPOCH)
v7_deployment = train_deployment(v7_meta, v7_logmels, seeds=V7_SEEDS,
    epochs=V7_MAX_EPOCHS, samples_per_epoch=V7_SAMPLES_PER_EPOCH)
v7_torgo_folds, v7_deployment = relock_exact_thresholds()
v7_calibration_comparison = summarize_grouped_validation_calibration()
v7_outer_calibration_comparison = summarize_outer_calibration_comparison()
v7_gender_subgroups = summarize_gender_subgroups()
v7_threshold_comparison = summarize_threshold_policies()
display(v7_torgo_folds)
display(v7_deployment)
display(v7_calibration_comparison)
display(v7_outer_calibration_comparison)
display(v7_gender_subgroups)
display(v7_threshold_comparison)
"""),
    cell("markdown", """## TORGO microphone and noise robustness

The microphone experiment applies the same physical-speaker outer folds.
Head-trained models use head-only inner training and head-only inner validation;
array-trained models likewise use array-only inner training and validation.
Each locked model is evaluated on both microphone types of outer-test speakers.
The SNR experiment adds predetermined Gaussian noise to outer-test audio at
30, 20, 10, 5, and 0 dB, using the locked TORGO parameters. These are diagnostic
TORGO analyses and do not consult official UASpeech labels for training choices.
"""),
    cell("code", """# V7.3 — TORGO robustness (seed 42, all outer speaker folds)
from v7.methodology import run_cross_microphone, run_noise_robustness
v7_cross_mic = run_cross_microphone(v7_meta, v7_logmels, seed=42,
    epochs=V7_MAX_EPOCHS, samples_per_epoch=V7_SAMPLES_PER_EPOCH)
v7_noise = run_noise_robustness(v7_meta, seed=42)
display(v7_cross_mic)
display(v7_noise)
"""),
    cell("markdown", """## Official UASpeech five-seed zero-shot external validation

This section loads only frozen TORGO deployment checkpoints. M3 is the
prespecified primary channel. M2–M8 are evaluated separately and their
comparison uses the intersection of speaker/content keys across channels;
simultaneous recordings are never counted as independent physical subjects.
Speaker metrics aggregate one mean probability per physical speaker. The 95%
interval resamples physical speakers, not utterances. Official severity names
are read unchanged from the verified release metadata. UASpeech labels are used
only for final measurement after all five models and TORGO settings are frozen.
"""),
    cell("code", """# V7.4 — frozen external inference; errors if any of seeds 42–46 is missing
from v7.external import (load_official_metadata, load_frozen_models,
    precompute_official_features, predict_official, evaluate_official,
    evaluate_identical_outer_checkpoints)
v7_official_meta = load_official_metadata()
v7_frozen = load_frozen_models()
v7_official_feature_cache = precompute_official_features(v7_official_meta)
v7_official_predictions = predict_official(v7_official_meta, v7_frozen)
v7_official_metrics, v7_official_ci, v7_channels = evaluate_official(
    v7_official_predictions, v7_frozen)
v7_identical_checkpoint_pairs, v7_paired_predictions = evaluate_identical_outer_checkpoints(
    v7_official_meta)
display(v7_official_metrics)
display(v7_official_ci)
display(v7_channels)
display(v7_identical_checkpoint_pairs)
"""),
    cell("markdown", """# Publication Figures — Official UASpeech External Validation

These figures are generated from V7 computed predictions and result tables only.
Each saved figure has a 300-DPI PNG and vector PDF. The index distinguishes
main-paper, supplementary, and diagnostic visuals. The seed-42 confusion matrix,
speaker probability display, and calibration diagram are descriptive prespecified
single-seed views; the speaker ROC and five-seed plots/summaries use all five
frozen seeds. The optional radar
chart is omitted because conventional plots carry the manuscript evidence.
"""),
    cell("code", """# V7.5 — generate publication figures and figure index
from v7.figures import make_publication_figures
v7_figure_index = make_publication_figures()
display(v7_figure_index)
"""),
    cell("code", """# V7.6 — final master report, only from completed V7 outputs
import pandas as pd
import numpy as np
_v7_tor = pd.read_csv(V7_OUT / 'torgo_nested_fold_metrics.csv')
_v7_ext = pd.read_csv(V7_OUT / 'official_per_seed_metrics.csv')
_v7_ci = pd.read_csv(V7_OUT / 'official_speaker_cluster_ci.csv')
_v7_mean_ci = pd.read_csv(V7_OUT / 'official_five_seed_mean_speaker_cluster_ci.csv')
_v7_ch = pd.read_csv(V7_OUT / 'official_matched_channel_metrics.csv')
_v7_noise = pd.read_csv(V7_OUT / 'torgo_noise_results.csv')
_v7_mic = pd.read_csv(V7_OUT / 'torgo_cross_microphone_results.csv')
_v7_outer_cal = pd.read_csv(V7_OUT / 'torgo_outer_calibration_comparison.csv')
def _v7_mean_sd(frame, metric):
    per_seed = frame.groupby('seed')[metric].mean()
    n = per_seed.notna().sum()
    if n < len(V7_SEEDS):
        if n == 0:
            detail = 'no estimable-seed mean'
        elif n == 1:
            detail = f'one estimable seed {per_seed.mean():.3f}; SD unavailable'
        else:
            detail = f'estimable-seed mean {per_seed.mean():.3f} ± {per_seed.std(ddof=1):.3f}'
        return f'non-estimable in {len(V7_SEEDS)-n}/{len(V7_SEEDS)} seeds; {detail}'
    return f'{per_seed.mean():.3f} ± {per_seed.std(ddof=1):.3f}'
_v7_metrics = ['Accuracy','Balanced_Accuracy','Sensitivity','Specificity','Precision','NPV',
               'F1_dysarthric','Macro_F1','ROC_AUC','PR_AUC','Brier','ECE']
print('PHYSICAL-SPEAKER LEAKAGE CHECK: PASS')
_v7_hist_overlap = v7_historical_leakage.loc[v7_historical_leakage.physical_overlap_across_v6_splits,
                                             'physical_speaker_id'].nunique()
print('Historical V6 session split physical-speaker leakage: FAIL',
      f'({_v7_hist_overlap} of {v7_historical_leakage.physical_speaker_id.nunique()} speakers overlap)')
print('TORGO MULTI-SEED (grouped outer evaluation):')
for k in _v7_metrics: print(' ',k, _v7_mean_sd(_v7_tor[_v7_tor.level.eq('utterance')],k))
print('OFFICIAL UASPEECH ZERO-SHOT (M3):')
for k in _v7_metrics: print(' ',k, _v7_mean_sd(_v7_ext[_v7_ext.level.eq('utterance')],k))
print('OFFICIAL UASPEECH SPEAKER LEVEL:')
for k in _v7_metrics: print(' ',k, _v7_mean_sd(_v7_ext[_v7_ext.level.eq('physical_speaker')],k))
print(' Physical-speaker cluster bootstrap 95% CI:')
print(_v7_ci.to_string(index=False))
print(' Five-seed mean physical-speaker cluster bootstrap 95% CI:')
print(_v7_mean_ci.to_string(index=False))
print('EXTERNAL CALIBRATION (Brier / ECE):',
      _v7_mean_sd(_v7_ext[_v7_ext.level.eq('utterance')],'Brier'), '/',
      _v7_mean_sd(_v7_ext[_v7_ext.level.eq('utterance')],'ECE'))
print('TORGO HELD-OUT CALIBRATION (uncalibrated vs TORGO temperature scaled):')
print(_v7_outer_cal.groupby('variant')[['Brier','ECE','ROC_AUC','PR_AUC']].mean().round(3).to_string())
print('MICROPHONE ROBUSTNESS (matched-content ROC-AUC):',
      _v7_ch[_v7_ch.level.eq('physical_speaker')].groupby('channel').ROC_AUC.mean().round(3).to_dict())
print('NOISE ROBUSTNESS (SNR / Balanced Accuracy):',
      _v7_noise.groupby('SNR_dB').Balanced_Accuracy.mean().round(3).to_dict())
print('TORGO CROSS-MICROPHONE:',
      _v7_mic.groupby(['train_mic','test_mic']).Balanced_Accuracy.mean().round(3).to_dict())
print('TORGO subgroup counts and non-estimable discrimination flags:',
      V7_OUT / 'torgo_outer_subgroup_counts.csv',
      V7_OUT / 'torgo_outer_subgroup_metrics.csv',
      V7_OUT / 'torgo_gender_subgroup_metrics.csv')
print('FIGURES GENERATED:',len(v7_figure_index))
print('Official UASpeech used for training: NO')
print('Official UASpeech used for fine-tuning: NO')
print('Official UASpeech used for threshold selection: NO')
print('Official UASpeech used for calibration: NO')
print('Official UASpeech used for model selection: NO')
"""),
])
assert notebook["cells"][:len(original)]==original
if TARGET.exists():
    existing=json.loads(TARGET.read_text())
    if len(existing.get("cells",[]))>len(original):
        # The checked-in V7 notebook contains executed results and later
        # corrected analyses. Rebuilding from V6 would erase that evidence.
        print(f"Preserving existing V7 notebook and its {len(existing['cells'])} cells: {TARGET}")
        raise SystemExit(0)
TARGET.write_text(json.dumps(notebook,ensure_ascii=False,indent=1)+"\n")
print(TARGET)
print(f"Preserved V6 cells: {len(original)}; added V7 cells: {len(notebook['cells'])-len(original)}")
