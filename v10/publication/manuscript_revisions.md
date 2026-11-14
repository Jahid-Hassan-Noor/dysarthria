# Manuscript revision text

Replacement text for the abstract and Section 5 of the supplied preprint, plus
shorter versions of its Tables 7 and 10. These values refer to the original
speaker-level analyses; full precision results remain in the publication CSVs
and the V10.4 supplement outputs. The supplied PDFs are read-only, and no
editable source for this preprint is in the repository.

## Abstract

**Background and objective:** High dysarthria detection scores on TORGO and
UA-Speech may reflect recording duration or acquisition conditions. We separate
whether a cue is available, whether it can reproduce discrimination, and
whether the locked detector uses it.

**Methods:** A frozen WavLM Base encoder and speaker-attention head were assessed
on 15 TORGO speakers using nested speaker-grouped cross-validation across five
seeds. Models and thresholds were locked for evaluation on 28 UA-Speech
speakers. Cue-only classifiers, paired speaker-bootstrap comparisons, and
test-time interventions with processing controls examined candidate shortcuts.

**Results:** Detector ROC-AUC was 0.850 on TORGO and 0.966 on UA-Speech; at
locked thresholds, sensitivity was 0.850 and 0.720, respectively. A
duration-only classifier reached ROC-AUC 0.964 and 1.000, leaving little
ranking headroom. In the original TORGO folds, adding detector scores to a
duration stack changed ROC-AUC by 0.025 (95% speaker-bootstrap interval
-0.104 to 0.196). A classifier trained on TORGO low-energy edges reached
0.804, but its paired advantage over locked speech heads on the same recordings
was uncertain (0.082, interval -0.125 to 0.311); the edges may contain speech.
The locked detector scored UA-Speech silence at 0.927 but TORGO silence at
0.461. Tempo and duration-spread interventions changed scores, yet their
processing-only controls were larger, leaving mechanism tests inconclusive.

**Conclusions:** Duration can reproduce much of the observed ranking, but these
experiments do not establish how much the detector uses it or whether its
scores reflect a disease-specific speech marker. Cue availability, predictive
sufficiency, and detector use require separate tests in each corpus.

## Section 5. Conclusions

Cue availability, predictive sufficiency, and use by a trained detector are
separate questions. The frozen WavLM detector achieved ROC-AUC 0.850 on held-out
TORGO speakers and 0.966 on locked UA-Speech evaluation. A duration-only model
achieved 0.964 and 1.000, respectively. The small, uncertain increment over
duration in the original TORGO folds does not establish additional
disease-specific information. On UA-Speech, the locked detector also separated
groups using silence-only input (ROC-AUC 0.927). These findings limit what
speaker-level ranking on the two corpora can show about a clinical speech
mechanism.

The mechanism probes were inconclusive. A separately trained classifier found
signal in TORGO low-energy edges, but the paired comparison with locked speech
heads included zero; the edges may contain speech. The locked detector scored
TORGO silence near chance, illustrating that cue availability does not imply
use by that detector. Test-time tempo and within-speaker duration-spread
manipulations moved scores, but the processing-only controls moved them as much
or more. The observed responses therefore cannot be assigned to duration or
tempo alone. The five seeds repeatedly partitioned the same 15 TORGO speakers,
so they do not expand the clinical cohort.

Future work should use larger speaker cohorts recorded with comparable
equipment and conditions, report duration and background controls alongside
locked operating points, and design interventions whose processing artifacts
are smaller than the cue change being tested. This would allow an increment
beyond shortcuts, severity-stratified errors, and mechanism claims to be
estimated with useful precision.

## Replacement Table 7: confound battery

| Test | Corpus (speakers) | ROC-AUC | Balanced accuracy |
| --- | ---: | ---: | ---: |
| Reference detector | TORGO (15) | 0.850 | 0.854 |
| Reference detector | UA-Speech M3 (28) | 0.966 | 0.860 |
| Locked detector, silence | TORGO (15) | 0.461 | 0.452 |
| Locked detector, silence | UA-Speech M3 (28) | 0.927 | 0.538 |
| Locked detector, duration-overlap subset | TORGO (15) | 0.846 | 0.843 |
| Locked detector, matched prompts | UA-Speech M3 (25) | 0.982 | 0.871 |
| Duration-only classifier | TORGO (15) | 0.964 | 0.689 |
| Duration-only classifier | UA-Speech M3 (28) | 1.000 | 0.667 |
| Duration plus level classifier | TORGO (15) | 0.921 | 0.575 |
| Duration plus level classifier | UA-Speech M3 (28) | 1.000 | 0.793 |
| Other cue-only classifiers | TORGO (15) | 0.479–0.836 | 0.454–0.591 |
| All descriptors plus detector | TORGO (15) | 0.857 | 0.636 |

Values are five-seed means; the five seeds reuse the same speakers. The range
in “other cue-only classifiers” covers silence ratio, RMS/peak, spectral
proxies, and all 16 descriptors, rather than a pooled estimate. Keep the
original full Table 7 in supplementary material for accuracy, Brier score,
seed variation, and individual cue-only rows.

## Replacement Table 10: evidence by cue

| Cue and corpus | Availability or sufficiency | Locked-detector use |
| --- | --- | --- |
| Duration, TORGO | Duration-only ROC-AUC 0.964; original-fold increment +0.025 [-0.104, 0.196] | Tempo and spread interventions inconclusive after processing controls. |
| Duration, UA-Speech | Duration-only ROC-AUC 1.000; no ROC-AUC headroom | Not directly tested. |
| Low-energy edges, TORGO | Trained-head ROC-AUC 0.804; paired advantage over locked speech heads +0.082 [-0.125, 0.311] | Locked detector on silence ROC-AUC 0.461; edges may contain speech. |
| Silence, UA-Speech | Available; independent contribution beyond duration unresolved | Locked detector on silence ROC-AUC 0.927. |
| Matched lexical content, UA-Speech | Matched subset retained ROC-AUC 0.982 | No clear evidence of dependence on prompt mix. |
| Vocoder processing, TORGO | Processing-only control shifted scores | Mechanism of duration response unresolved. |

Intervals are 95% speaker-bootstrap intervals. “Not directly tested” and
“unresolved” preserve the distinction between a missing or failed mechanism
test and evidence that a cue was not used. The original numerical details
belong in supplementary material.
