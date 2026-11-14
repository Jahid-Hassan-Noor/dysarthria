# Manuscript revision suggestions

The supplied manuscript is a PDF with no matching editable source in the
workspace. The text below is an exact proposed replacement for the named
sections and tables; it has **not** been inserted into the PDF. The supplied
supplement remains unchanged. Original wording excerpts and reasons are
recorded here so each proposed change can be reviewed.

| Location | Original wording (manuscript PDF) | Reason for proposed replacement |
| --- | --- | --- |
| Abstract, background | “We developed a framework measuring all three.” | The study measures availability and predictive performance, but processing artifacts prevent a clean determination of use for duration or tempo. |
| Abstract, methods | “Locked models scored silence-only and time-stretched audio, with a processing-only control, to test use.” | A restricted-input or processed-audio response does not by itself establish reliance during ordinary speech classification. |
| Abstract, results | “Slowing raised scores, but processing alone moved them more, leaving tempo’s share inconclusive.” | This finding is retained in the replacement; the duration-spread control is also named. |
| Conclusion | “One such counterexample is enough to show that a cue’s presence does not imply its use” | Preserve the distinction, while avoiding a claim that near-chance silence-only scoring rules out all background reliance during ordinary speech. |
| Conclusion, final sentence | “A dysarthria corpus in which patients and controls share recording equipment and setting, with about 40–120 speakers per class for a ±0.05 interval, would remove the duration ceiling” | Matched equipment alone cannot guarantee matched duration distributions or a particular precision; the replacement states design goals rather than a guaranteed effect. |
| Discussion, future data paragraph | “A dysarthria corpus in which patient and control recordings share microphone, room and session protocol ... would remove the duration ceiling” | The same guarantee appears before the Limitations section; the replacement makes the study-design recommendation conditional. |
| Table 10 | “Yes: duration SD alone separates all 15 speakers (1.000; descriptive, in-sample)” and “Partly: scores rise with slowing” occur in the same “Duration, TORGO” row alongside several other estimates. | Separate availability, sufficiency, response, and ordinary-speech reliance; retain numbers below the table. |

The abstract replacement is under 250 words. All quantitative values cited in
it were checked against the manuscript, supplement, and saved V10/V10.4 result
files. The compact tables are proposals for the manuscript source, with the
full numerical originals to be retained in supplementary material.

Replacement text for the abstract, selected Results/Discussion/Limitations
passages, and Section 5 of the supplied preprint, plus shorter versions of its
Tables 7 and 10. These values refer to the original
speaker-level analyses; full precision results remain in the publication CSVs
and the V10.4 supplement outputs. The supplied PDFs are read-only, and no
editable source for this preprint is in the repository.

## Abstract

**Background and objective:** Detection on TORGO and UA-Speech may reflect
duration or acquisition conditions. We separate cue availability, predictive
sufficiency, and locked-detector response.

**Methods:** A frozen WavLM Base encoder and speaker-attention head were assessed
on 15 TORGO speakers using nested speaker-grouped cross-validation across five
seeds. Models and thresholds were locked for evaluation on 28 UA-Speech
speakers. Cue-only classifiers, speaker-bootstrap comparisons, and
test-time interventions with processing controls probed shortcuts
without assuming reliance during ordinary speech classification.

**Results:** Detector ROC-AUC was 0.850 on TORGO and 0.966 on UA-Speech; at
locked thresholds, sensitivity was 0.850 and 0.720, respectively. A
duration-only classifier reached ROC-AUC 0.964 and 1.000, leaving little
ranking headroom. In TORGO folds, adding detector scores to a
duration stack changed ROC-AUC by 0.025 (95% speaker-bootstrap interval
-0.104 to 0.196). A classifier trained on TORGO low-energy edges reached
0.804, but its paired advantage over locked speech heads on the same recordings
was uncertain (0.082, interval -0.125 to 0.311); the edges may contain speech.
Silence-only detector AUC was 0.927 on UA-Speech and 0.461 on TORGO;
ordinary-speech background reliance remains untested. Tempo and duration-spread
interventions changed scores, yet their
processing-only controls were larger, leaving mechanism tests inconclusive.

**Conclusions:** Duration can reproduce much of the observed ranking, but these
experiments do not establish how much the detector uses it or whether its
scores reflect a disease-specific speech marker. Cue availability, predictive
sufficiency, and detector use require separate tests in each corpus.

## Results interpretation replacement: TORGO low-energy and silence controls

A classifier trained on the selected TORGO low-energy edges reached ROC-AUC
0.804. Its paired advantage over locked speech heads on the same source
recordings was +0.082 (95% speaker-bootstrap interval -0.125 to 0.311), and
the selected edges may contain speech. The locked detector scored the tested
TORGO silence-only input near chance (ROC-AUC 0.461), indicating no
demonstrated discrimination from that restricted input. This does not rule out
reliance on recording-background information during ordinary speech
classification. On UA-Speech the locked detector's silence-only ROC-AUC was
0.927, a restricted-input response whose contribution during ordinary speech
classification was not isolated from duration.

## Discussion replacement: future data paragraph before Section 4.6

A future dysarthria corpus should use comparable recording equipment and
sessions for patients and controls, record duration and speaking rate
explicitly, and include enough speakers per class for useful interval
precision. Those design choices would reduce identifiable recording confounds;
they would not by themselves guarantee that duration ceases to be predictive.
An increment beyond duration should be tested directly. Multiple independent
rate-manipulation methods, listening checks, and processing-only controls
could help separate tempo from artifacts before a mechanism claim is made.
Likewise, near-chance scoring on tested TORGO silence-only clips cannot exclude
background reliance when speech and background are present together.

## Section 4.6 Limitations insertion

Add after the existing caveat that selected silence segments may contain
residual speech: “Near-chance scoring on the tested TORGO silence-only input
does not exclude use of background information when speech and background
co-occur. The restricted-input analysis does not isolate ordinary-speech
reliance.” Retain the existing small-cohort, threshold, processing-artifact,
prior UA-Speech exposure, dataset-shift, and repeated-seed limitations.

## Section 5. Conclusions

Cue availability, predictive sufficiency, and use by a trained detector are
separate questions. The frozen WavLM detector achieved ROC-AUC 0.850 on held-out
TORGO speakers and 0.966 on locked UA-Speech evaluation of a previously studied
corpus. A duration-only model
achieved 0.964 and 1.000, respectively. The small, uncertain increment over
duration in the original TORGO folds does not establish additional
disease-specific information. On UA-Speech, the locked detector also separated
groups using silence-only input (ROC-AUC 0.927). These findings limit what
speaker-level ranking on the two corpora can show about a clinical speech
mechanism.

The mechanism probes were inconclusive. A separately trained classifier found
signal in TORGO low-energy edges, but the paired comparison with locked speech
heads included zero; the edges may contain speech. The locked detector scored
the tested TORGO silence-only input near chance, showing no discrimination
from that restricted input; this does not exclude background reliance during
ordinary speech classification. Test-time tempo and within-speaker duration-spread
manipulations moved scores, but the processing-only controls moved them as much
or more. The observed responses therefore cannot be assigned to duration or
tempo alone. The five seeds repeatedly partitioned the same 15 TORGO speakers,
so they do not expand the clinical cohort.

Future work should use larger speaker cohorts recorded with comparable
equipment and conditions, report duration and background controls alongside
locked operating points, and design interventions whose processing artifacts
are smaller than the cue change being tested. Threshold uncertainty, clinical
dataset shift, and severity-stratified errors must also be assessed before
generalizing to real-world recordings. Such studies could estimate any
increment beyond shortcuts with useful precision.

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
The TORGO silence-only row is a restricted-input result and does not exclude
background reliance during ordinary speech classification.

## Replacement Table 10: evidence by cue

| Cue and corpus | Availability | Predictive sufficiency | Locked-detector response; ordinary-speech reliance |
| --- | --- | --- | --- |
| Duration, TORGO | Yes | Yes by point estimate | Tempo and spread responses; reliance inconclusive after processing controls. |
| Duration, UA-Speech | Yes | Yes, at the ROC-AUC ceiling | Direct reliance test unavailable. |
| Low-energy edges, TORGO | Yes | Uncertain against matched speech heads | No discrimination shown on tested silence-only input; ordinary-speech background reliance unresolved. |
| Silence, UA-Speech | Yes | Independent value beyond duration unresolved | Silence-only discrimination observed; ordinary-speech reliance unresolved. |
| Matched lexical content, UA-Speech | Tested on a matched subset | Not a cue-only sufficiency test | No clear evidence of dependence on prompt mix. |
| Vocoder processing, TORGO | Not a candidate disease cue | Not applicable | Processing sensitivity observed; tempo mechanism unresolved. |

Quantitative evidence to retain beside the compact table: duration-only
ROC-AUC 0.964 on TORGO and 1.000 on UA-Speech; original-fold increment +0.025
[-0.104, 0.196]; TORGO edge-trained head 0.804 and paired advantage over
matched locked speech heads +0.082 [-0.125, 0.311]; locked silence-only
ROC-AUC 0.461 on TORGO and 0.927 on UA-Speech; matched prompts 0.982.
Intervals are 95% speaker-bootstrap intervals. The selected low-energy edges
may contain speech. “Unresolved” is not evidence that a cue was absent or
unused. Keep the complete numerical evidence matrix in supplementary material.

## Supplementary material note for Table S8

Add after the silence-trained-head comparison: “The edge-trained head's ROC-AUC
of 0.804 shows predictive information in the selected low-energy clips, which
may contain residual speech. The locked detector's TORGO silence-only ROC-AUC
of 0.461 shows no demonstrated discrimination from that restricted input; it
does not rule out background reliance during ordinary speech classification.
The paired comparison with locked speech heads on matched source recordings
has a 95% interval spanning zero.”
