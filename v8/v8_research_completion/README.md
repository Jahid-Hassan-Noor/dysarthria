# Additional V8 paper figures

These four figures use the completed, frozen V8 results. They do not run feature extraction, train a model, or change the existing V8 publication figures. Each figure is saved as a 300 DPI PNG and a vector PDF in [`figures/`](figures/). The exact plotted values are in [`figures/source_data/`](figures/source_data/), and [`figure_manifest.json`](figures/figure_manifest.json) records source-file hashes and counts.

The final two cells of [`../dysarthria_v8.ipynb`](../dysarthria_v8.ipynb) generate and display these figures when you select **Run All** in VS Code using the project's `.venv` kernel.

Regenerate them with:

```bash
.venv/bin/python v8/v8_research_completion/make_paper_figures.py
```

| Figure | Suggested placement | What it supports |
| --- | --- | --- |
| [Fig. 10: duration vs. V8 score](figures/fig10_duration_vs_v8_score.pdf) | Confounds / limitations | Shows the association between each speaker's median **recording** duration and mean V8 score. The five-seed score range is shown for each speaker. UASpeech M3 has a strong pooled rank association (ρ = +0.77); TORGO does not (ρ = −0.22). These descriptive associations mix diagnoses and do not establish whether V8 uses duration or whether duration causes a prediction. |
| [Fig. 11: matched words, paired scores](figures/fig11_matched_words_paired_scores.pdf) | External validation / lexical control | Compares the same 25 UASpeech M3 speakers under the full and verified common-word subsets using the locked model and thresholds. Seven of 125 speaker-seed decisions differ. Three of 28 M3 speakers lack the verified matched-word subset, so this does not describe the full M3 cohort. |
| [Fig. 12: paired ablation effects](figures/fig12_paired_ablation_deltas.pdf) | Model ablation | Shows speaker-bootstrap 95% intervals for each alternative minus the V8 reference. All balanced-accuracy intervals cross zero; the existing results do not establish a clear gain from the reference architecture. The five seeds reuse the same 15 TORGO speakers. |
| [Fig. 13: epoch selection](figures/fig13_epoch_selection.pdf) | Methods supplement / stability | Shows the 75 inner-fold best epochs and the 25 selected final-fit epoch counts. The inner-fold range is 1–40 and the final-fit range is 5–39. This is a descriptive stability diagnostic, not a fixed-epoch experiment. |

Figures 10 and 11 are the strongest additions to the main paper because they directly address the current confound and lexical-control concerns. Figure 12 belongs with the ablation results. Figure 13 fits best in the supplement unless training stability becomes a central claim. None of these figures establishes that V8 retains dysarthria-specific signal after active-speech and duration matching; that requires the separate controlled analyses described in the research brief.
