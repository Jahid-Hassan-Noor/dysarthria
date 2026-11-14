# Controlled ablations

| Model or condition | ROC-AUC [95% interval if reported] | Balanced accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| WavLM Last Layer + Mean Pooling | 0.850 [0.664, 1.000] | 0.829 | 0.800 | 0.857 |
| WavLM Last Layer + Attention | 0.821 [0.611, 0.979] | 0.825 | 0.850 | 0.800 |
| WavLM Last-4 Mix + Mean Pooling | 0.889 [0.693, 1.000] | 0.827 | 0.825 | 0.829 |
| WavLM Last-4 Mix + Attention (V8 Reference) | 0.850 [0.679, 0.975] | 0.854 | 0.850 | 0.857 |
| V10 mean only | 0.882 [0.768, 0.979] | 0.827 | 0.825 | 0.829 |

TORGO: five repeated seeds. Intervals follow the original ablation analysis. These comparisons do not isolate a clinical mechanism.

Full-precision values, additional metrics, and seed variation: [table04_controlled_ablations.csv](table04_controlled_ablations.csv).
