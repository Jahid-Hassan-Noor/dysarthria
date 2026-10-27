# V8 component ablation

| model | status | n_seeds | n_speakers_min | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | F1_dysarthric | ROC_AUC | PR_AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WavLM Last Layer + Mean Pooling | COMPLETE | 5 | 15 | 0.827 ± 0.101 | 0.829 ± 0.101 | 0.800 ± 0.112 | 0.857 ± 0.101 | 0.830 ± 0.103 | 0.850 ± 0.109 | 0.914 ± 0.055 | 0.158 ± 0.093 | 0.217 ± 0.074 |
| WavLM Last Layer + Attention | COMPLETE | 5 | 15 | 0.827 ± 0.060 | 0.825 ± 0.056 | 0.850 ± 0.137 | 0.800 ± 0.078 | 0.835 ± 0.072 | 0.821 ± 0.119 | 0.855 ± 0.081 | 0.181 ± 0.075 | 0.282 ± 0.068 |
| WavLM Last-4 Mix + Mean Pooling | COMPLETE | 5 | 15 | 0.827 ± 0.101 | 0.827 ± 0.101 | 0.825 ± 0.112 | 0.829 ± 0.120 | 0.835 ± 0.102 | 0.889 ± 0.029 | 0.937 ± 0.028 | 0.140 ± 0.088 | 0.197 ± 0.086 |
| WavLM Last-4 Mix + Attention (V8 Reference) | COMPLETE | 5 | 15 | 0.853 ± 0.087 | 0.854 ± 0.081 | 0.850 ± 0.163 | 0.857 ± 0.000 | 0.854 ± 0.098 | 0.850 ± 0.136 | 0.884 ± 0.089 | 0.160 ± 0.090 | 0.230 ± 0.116 |

Metrics are mean ± sample SD across seeds; repetitions are averaged inside each seed. PR-AUC denotes average precision. Missing analyses contain no estimated values. External results are locked zero-shot cross-corpus evaluations; M3 is prespecified primary.
