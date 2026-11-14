# Simple baselines

| Model or condition | ROC-AUC [95% interval if reported] | Balanced accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| MFCC Logistic Regression | 0.857 [0.668, 0.986] | 0.688 | 0.775 | 0.600 |
| MFCC RBF SVM | 0.532 [0.382, 0.679] | 0.479 | 0.900 | 0.057 |
| WavLM Final Layer Logistic Regression | 0.907 [0.757, 1.000] | 0.713 | 0.825 | 0.600 |

TORGO: 15 physical speakers across five repeated seeds. Intervals resample speakers; seeds are not independent cohorts.

Full-precision values, additional metrics, and seed variation: [table03_simple_baselines.csv](table03_simple_baselines.csv).
