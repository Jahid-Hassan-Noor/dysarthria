# Official UASpeech M3 locked zero-shot result

| model | corpus | channel | status | n_seeds | n_speakers_min | n_recordings | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | F1_dysarthric | ROC_AUC | PR_AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | M3 | COMPLETE | 5 | 28 | 21126.0 | 0.850 ± 0.030 | 0.860 ± 0.028 | 0.720 ± 0.056 | 1.000 ± 0.000 | 0.836 ± 0.037 | 0.966 ± 0.013 | 0.976 ± 0.008 | 0.130 ± 0.020 | 0.163 ± 0.024 |

Metrics are mean ± sample SD across seeds; repetitions are averaged inside each seed. PR-AUC denotes average precision. Missing analyses contain no estimated values. External results are locked zero-shot cross-corpus evaluations; M3 is prespecified primary.
