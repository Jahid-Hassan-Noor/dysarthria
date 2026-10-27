# Artifact/content sanity controls

| model | corpus | analysis | channel | status | n_seeds | n_speakers_min | n_recordings | n_matched_prompts | Accuracy | Balanced_Accuracy | Sensitivity | Specificity | F1_dysarthric | ROC_AUC | PR_AUC | Brier | ECE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | primary_reference |  | COMPLETE | 5 | 15 | 17630.0 |  | 0.853 ± 0.087 | 0.854 ± 0.081 | 0.850 ± 0.163 | 0.857 ± 0.000 | 0.854 ± 0.098 | 0.850 ± 0.136 | 0.884 ± 0.089 | 0.160 ± 0.090 | 0.230 ± 0.116 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | primary_reference | M3 | COMPLETE | 5 | 28 | 21126.0 |  | 0.850 ± 0.030 | 0.860 ± 0.028 | 0.720 ± 0.056 | 1.000 ± 0.000 | 0.836 ± 0.037 | 0.966 ± 0.013 | 0.976 ± 0.008 | 0.130 ± 0.020 | 0.163 ± 0.024 |
| WavLM Last-4 Mix + Attention (V8 Reference) | TORGO | silence | original_TORGO | COMPLETE | 5 | 15 | 750.0 |  | 0.480 ± 0.056 | 0.452 ± 0.055 | 0.875 ± 0.088 | 0.029 ± 0.064 | 0.641 ± 0.046 | 0.461 ± 0.099 | 0.557 ± 0.059 | 0.391 ± 0.024 | 0.397 ± 0.007 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | silence | M3 | COMPLETE | 5 | 28 | 1259.0 |  | 0.571 ± 0.062 | 0.538 ± 0.067 | 1.000 ± 0.000 | 0.077 ± 0.133 | 0.715 ± 0.031 | 0.927 ± 0.019 | 0.939 ± 0.019 | 0.313 ± 0.083 | 0.361 ± 0.070 |
| Duration-only logistic regression | TORGO | duration |  | COMPLETE | 5 | 15 | 17630.0 |  | 0.707 ± 0.076 | 0.689 ± 0.081 | 0.950 ± 0.068 | 0.429 ± 0.175 | 0.777 ± 0.047 | 0.964 ± 0.028 | 0.973 ± 0.021 | 0.109 ± 0.010 | 0.201 ± 0.051 |
| Duration-only logistic regression | UASpeech | duration |  | COMPLETE | 5 | 28 | 21126.0 |  | 0.643 ± 0.000 | 0.667 ± 0.000 | 0.333 ± 0.000 | 1.000 ± 0.000 | 0.500 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.275 ± 0.000 | 0.337 ± 0.000 |
| Artifact-only logistic regression | TORGO | artifact |  | COMPLETE | 5 | 15 | 17630.0 |  | 0.600 ± 0.000 | 0.575 ± 0.005 | 0.950 ± 0.068 | 0.200 ± 0.078 | 0.716 ± 0.015 | 0.921 ± 0.020 | 0.947 ± 0.013 | 0.123 ± 0.009 | 0.185 ± 0.053 |
| Artifact-only logistic regression | UASpeech | artifact |  | COMPLETE | 5 | 28 | 21126.0 |  | 0.779 ± 0.089 | 0.793 ± 0.083 | 0.587 ± 0.166 | 1.000 ± 0.000 | 0.729 ± 0.121 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.311 ± 0.000 | 0.381 ± 0.000 |
| WavLM Last-4 Mix + Attention (V8 Reference) | UASpeech | matched_content | M3 | COMPLETE | 5 | 25 | 7650.0 | 306.0 | 0.856 ± 0.046 | 0.871 ± 0.041 | 0.743 ± 0.081 | 1.000 ± 0.000 | 0.850 ± 0.053 | 0.982 ± 0.010 | 0.988 ± 0.006 | 0.127 ± 0.020 | 0.177 ± 0.035 |

Metrics are mean ± sample SD across seeds; repetitions are averaged inside each seed. PR-AUC denotes average precision. Missing analyses contain no estimated values. External results are locked zero-shot cross-corpus evaluations; M3 is prespecified primary.
