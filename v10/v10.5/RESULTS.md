# Dysarthria V10.5: exploratory follow-up findings

These analyses follow the frozen V10.4 study and were not original pre-specified analyses. The speech-only WavLM weights, saved held-out TORGO heads, TORGO-trained UASpeech M3 deployment heads, and decision thresholds were unchanged. Five seeds reuse the same physical speakers; bootstrap intervals resample people within diagnosis, not seeds.

## A. TORGO prompt-matched duration

**Status: not estimable with the local audio-only TORGO release.** The frozen 17,630-recording metadata contains speaker, session-bearing path, microphone, and raw duration, but no verified prompt IDs or transcripts. The available local TORGO files are WAVs only. Treating a shared recording number as a shared prompt would be unsupported. The implementation writes `outputs/prompt_match/coverage.json` and waits for a verified `source_id` or `relative_path` to `prompt_id`/`transcript` map.

| Full-cohort context | Mean five-seed speaker-level ROC-AUC | Prompt-matched result |
| --- | ---: | --- |
| Original four-statistic raw-duration logistic baseline | 0.964 | Not estimable |
| Frozen WavLM speech-only detector | 0.850 | Not estimable |

The matched implementation is complete: it requires the same prompt set for all 15 physical speakers; chooses one recording per speaker/prompt with a fixed microphone preference; compares source and frozen-preprocessed duration; retrains only the fixed duration baseline within the saved nested speaker folds; rescores cached matched WavLM features with the corresponding held-out heads; and generates paired speaker bootstrap intervals, calibration metrics, coverage tables, and figures when a verified map is supplied. Matched WavLM inner-fold predictions are unavailable, so matched-duration incremental stacking is withheld even if prompt annotations are later supplied. The dominant duration-confound conclusion therefore remains **unresolved by prompt matching**.

## B. Speech-preserving background interventions

Three predeclared, hash-ranked recordings per speaker were taken from the V8 objectively selected quiet-edge set, subject to at least 100 ms of quiet frames remaining after the frozen trim. A different-speaker, same-microphone donor was selected without diagnosis labels. Interventions were applied to the exact post-trim waveform encoded by WavLM, with 5 ms fades at quiet-frame boundaries. Original and sham were identical by construction. The complements of edited frames were bit-identical, durations were unchanged, and no clipping or non-finite samples were observed.

| Corpus and condition | ROC-AUC | Balanced accuracy | Sensitivity | Specificity | Brier | ECE10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TORGO original/sham | 0.586 | 0.489 | 0.750 | 0.229 | 0.361 | 0.396 |
| TORGO neutralized | 0.596 | 0.493 | 0.900 | 0.086 | 0.331 | 0.351 |
| TORGO swapped | 0.521 | 0.434 | 0.725 | 0.143 | 0.395 | 0.433 |
| TORGO half intensity | 0.579 | 0.498 | 0.825 | 0.171 | 0.374 | 0.394 |
| TORGO double intensity | 0.614 | 0.518 | 0.750 | 0.286 | 0.341 | 0.373 |
| UA-Speech M3 original/sham | 0.915 | 0.826 | 0.760 | 0.892 | 0.147 | 0.165 |
| UA-Speech M3 neutralized | 0.951 | 0.747 | 0.987 | 0.508 | 0.151 | 0.206 |
| UA-Speech M3 swapped | 0.943 | 0.875 | 0.827 | 0.923 | 0.106 | 0.128 |
| UA-Speech M3 half intensity | 0.870 | 0.763 | 0.787 | 0.738 | 0.180 | 0.193 |
| UA-Speech M3 double intensity | 0.923 | 0.862 | 0.800 | 0.923 | 0.122 | 0.140 |

Values are means of five seed-pooled speaker metrics on this **three-recording subset**, not replacements for the original full-cohort performance estimates. TORGO's subset baseline AUC of 0.586 is much lower than the published full-cohort 0.850, so its intervention effects have limited generality.

Paired, diagnosis-stratified physical-speaker bootstrap comparisons against sham (2,000 draws):

| Corpus, intervention | Mean logit change [95% CI] | ROC-AUC change [95% CI] | Locked decisions changed |
| --- | ---: | ---: | ---: |
| TORGO neutralized | +0.226 [-0.381, +1.003] | +0.011 [-0.111, +0.104] | 15/75 repeated speaker-seed evaluations |
| TORGO swapped | -0.118 [-0.843, +0.881] | -0.064 [-0.271, +0.089] | 12/75 |
| TORGO half intensity | +0.370 [+0.056, +0.732] | -0.007 [-0.086, +0.064] | 5/75 |
| TORGO double intensity | -0.313 [-0.588, -0.066] | +0.029 [-0.025, +0.086] | 2/75 |
| UA-Speech neutralized | +2.384 [+1.673, +3.114] | +0.036 [-0.023, +0.121] | 42/140 repeated speaker-seed evaluations |
| UA-Speech swapped | +0.130 [-0.325, +0.643] | +0.028 [-0.016, +0.093] | 9/140 |
| UA-Speech half intensity | +1.132 [+0.763, +1.584] | -0.045 [-0.114, -0.003] | 14/140 |
| UA-Speech double intensity | +0.003 [-0.346, +0.445] | +0.008 [-0.028, +0.051] | 9/140 |

UA-Speech neutralization increased sensitivity but reduced specificity from 0.892 to 0.508; the paired specificity difference was -0.385 [95% CI -0.600, -0.185]. The detector clearly responds to low-energy waveform changes while speech is present, beyond the exact sham copy. The cross-source swap's mean logit and AUC intervals include zero in both corpora. These data therefore support **sensitivity to manipulated quiet-frame content and intensity**, especially on UA-Speech, but do not isolate a recording-background mechanism: low-energy frames and donor edges may contain weak speech, and some altered acoustic cues may arise from replacement despite the fades.

Quiet frames covered 41.7% of TORGO and 43.4% of UA-Speech samples on average, but only 0.32% and 1.33% of waveform energy respectively. Maximum edited-frame energy fractions were 2.22% and 7.18%. This quantifies a small energy footprint, not proof that every edited sample was speech-free. SNR values in `waveform_diagnostics.csv` are speech-frame/quiet-frame RMS proxies, not reference-separated SNR. Three recordings per speaker substantially reduce power and, for TORGO, change the evaluated content distribution.

## Manuscript implications

- Keep the original duration finding, but explicitly state that it has not survived a verified prompt match because such a match could not be executed with the available transcript-free TORGO release.
- Add the intervention as an exploratory follow-up: UA-Speech locked predictions move when low-energy frames are modified after trimming, including large neutralization-related decision changes. Describe the strong sensitivity/specificity tradeoff, not only AUC.
- Avoid claiming a pure background-artifact cause. State that quiet-frame speech contamination and the small per-speaker subset leave mechanism identification inconclusive. A larger, independently speech-verified segment set and verified TORGO prompts are needed before a causal claim.

No manuscript or supplementary file was changed. Full per-seed metrics, paired confidence intervals, speaker scores, waveform checks, source data tables, PNG/PDF figures, and source hashes are under `v10/v10.5/outputs/` and `provenance.json`.
