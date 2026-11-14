# Confound diagnostics

| Model or condition | ROC-AUC [95% interval if reported] | Balanced accuracy | Sensitivity | Specificity |
| --- | ---: | ---: | ---: | ---: |
| Combined measured confounds | 0.836 | 0.591 | 0.925 | 0.257 |
| Duration only | 0.964 | 0.689 | 0.950 | 0.429 |
| RMS energy only | 0.668 | 0.500 | 1.000 | 0.000 |
| Silence ratio only | 0.479 | 0.487 | 0.975 | 0.000 |
| Spectral proxies only | 0.625 | 0.454 | 0.850 | 0.057 |
| Confounds + V10 probability | 0.857 | 0.636 | 0.900 | 0.371 |
| V10 probability only | 0.850 | 0.854 | 0.850 | 0.857 |
| Inherited V8 background-only diagnostic | 0.461 | 0.452 | 0.875 | 0.029 |

TORGO: 15 physical speakers across five repeated seeds. A missing interval means the source table did not report one; cue-only discrimination does not establish cue use by the locked detector.

Full-precision values, additional metrics, and seed variation: [table10_confound_diagnostics.csv](table10_confound_diagnostics.csv).
