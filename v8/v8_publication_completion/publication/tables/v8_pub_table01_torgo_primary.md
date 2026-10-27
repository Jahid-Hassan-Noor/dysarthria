# TORGO physical-speaker-independent V8 primary result

| model | corpus | status | n_seeds | n_speakers_min | n_recordings | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | F1_dysarthric | ROC_AUC | PR_AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | COMPLETE | 5 | 15 | 17630.0 | 0.853 ± 0.087 | 0.854 ± 0.081 | 0.850 ± 0.163 | 0.857 ± 0.000 | 0.854 ± 0.098 | 0.850 ± 0.136 | 0.884 ± 0.089 | 0.160 ± 0.090 | 0.230 ± 0.116 |

Metrics are mean ± sample SD across seeds; repetitions are averaged inside each seed. PR-AUC denotes average precision. Missing analyses contain no estimated values. External results are locked zero-shot cross-corpus evaluations; M3 is prespecified primary.
