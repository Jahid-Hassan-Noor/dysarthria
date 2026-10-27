# V10.3 export copy

This folder contains copies of the V10.3 rate-intervention files produced by `v10/dysarthria_10.3.ipynb`. The originals remain in `v10/outputs/v10_3_rate` and `v10/publication`.

- `outputs/v10_3_rate/`: frozen analysis plan, 750-recording subset manifest, seven-condition manifest, 525 speaker scores, result summary, and six saved feature archives.
- `publication/figure_sources/`: Figure 22 source CSV (15 data rows).
- `publication/figures/`: Figure 22 PNG (3034 × 1234) and vector PDF.

Audit: the analysis-plan SHA-256 matches the summary; all seven conditions and five seeds are present; speaker scores have unique condition/seed/speaker keys; recorded AUCs match values recalculated from the scores; feature archives contain 750 finite records in the subset order and their crop counts match the summary. Every copy matches the original SHA-256 in `SHA256SUMS.txt`.

The notebook's optional UASpeech M3 extension has `RUN_UASPEECH = False`, so `uaspeech_m3_scores_by_condition.csv` was not produced or included. The primary TORGO V10.3 experiment is complete.
