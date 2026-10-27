# Utterance-budget sensitivity

| model | corpus | channel | budget | status | n_seeds | n_speakers_min | n_recordings | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | F1_dysarthric | ROC_AUC | PR_AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | TORGO | 1 | COMPLETE | 5 | 15 | 15.0 | 0.655 ± 0.034 | 0.638 ± 0.035 | 0.881 ± 0.034 | 0.395 ± 0.055 | 0.732 ± 0.026 | 0.731 ± 0.084 | 0.783 ± 0.076 | 0.261 ± 0.041 | 0.294 ± 0.040 |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | TORGO | 10 | COMPLETE | 5 | 15 | 150.0 | 0.797 ± 0.071 | 0.793 ± 0.070 | 0.854 ± 0.101 | 0.732 ± 0.076 | 0.816 ± 0.070 | 0.830 ± 0.108 | 0.864 ± 0.077 | 0.172 ± 0.074 | 0.233 ± 0.075 |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | TORGO | 20 | COMPLETE | 5 | 15 | 300.0 | 0.813 ± 0.080 | 0.811 ± 0.078 | 0.846 ± 0.121 | 0.776 ± 0.076 | 0.826 ± 0.083 | 0.839 ± 0.123 | 0.873 ± 0.086 | 0.165 ± 0.085 | 0.227 ± 0.087 |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | TORGO | 5 | COMPLETE | 5 | 15 | 75.0 | 0.764 ± 0.058 | 0.757 ± 0.058 | 0.854 ± 0.072 | 0.661 ± 0.067 | 0.794 ± 0.055 | 0.808 ± 0.097 | 0.842 ± 0.074 | 0.187 ± 0.064 | 0.242 ± 0.065 |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | TORGO | 50 | COMPLETE | 5 | 15 | 750.0 | 0.829 ± 0.080 | 0.829 ± 0.077 | 0.842 ± 0.137 | 0.815 ± 0.073 | 0.837 ± 0.087 | 0.846 ± 0.131 | 0.879 ± 0.088 | 0.162 ± 0.089 | 0.227 ± 0.094 |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | TORGO | all | COMPLETE | 5 | 15 | 17630.0 | 0.853 ± 0.087 | 0.854 ± 0.081 | 0.850 ± 0.163 | 0.857 ± 0.000 | 0.854 ± 0.098 | 0.850 ± 0.136 | 0.884 ± 0.089 | 0.160 ± 0.090 | 0.230 ± 0.116 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | M3 | 1 | COMPLETE | 5 | 28 | 28.0 | 0.818 ± 0.014 | 0.824 ± 0.014 | 0.743 ± 0.039 | 0.905 ± 0.037 | 0.811 ± 0.018 | 0.881 ± 0.008 | 0.919 ± 0.007 | 0.154 ± 0.018 | 0.174 ± 0.013 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | M3 | 10 | COMPLETE | 5 | 28 | 280.0 | 0.843 ± 0.025 | 0.854 ± 0.023 | 0.710 ± 0.048 | 0.997 ± 0.005 | 0.826 ± 0.034 | 0.947 ± 0.011 | 0.967 ± 0.006 | 0.136 ± 0.017 | 0.166 ± 0.021 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | M3 | 20 | COMPLETE | 5 | 28 | 560.0 | 0.849 ± 0.027 | 0.859 ± 0.025 | 0.719 ± 0.051 | 0.999 ± 0.002 | 0.833 ± 0.036 | 0.954 ± 0.012 | 0.971 ± 0.007 | 0.132 ± 0.018 | 0.163 ± 0.022 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | M3 | 5 | COMPLETE | 5 | 28 | 140.0 | 0.835 ± 0.025 | 0.845 ± 0.023 | 0.702 ± 0.048 | 0.989 ± 0.007 | 0.817 ± 0.034 | 0.936 ± 0.011 | 0.960 ± 0.007 | 0.141 ± 0.017 | 0.169 ± 0.021 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | M3 | 50 | COMPLETE | 5 | 28 | 1400.0 | 0.849 ± 0.026 | 0.859 ± 0.024 | 0.719 ± 0.048 | 1.000 ± 0.000 | 0.835 ± 0.033 | 0.960 ± 0.012 | 0.974 ± 0.006 | 0.130 ± 0.017 | 0.163 ± 0.022 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | M3 | all | COMPLETE | 5 | 28 | 21126.0 | 0.850 ± 0.030 | 0.860 ± 0.028 | 0.720 ± 0.056 | 1.000 ± 0.000 | 0.836 ± 0.037 | 0.966 ± 0.013 | 0.976 ± 0.008 | 0.130 ± 0.020 | 0.163 ± 0.024 |

Metrics are mean ± sample SD across seeds; repetitions are averaged inside each seed. PR-AUC denotes average precision. Missing analyses contain no estimated values. External results are locked zero-shot cross-corpus evaluations; M3 is prespecified primary.
