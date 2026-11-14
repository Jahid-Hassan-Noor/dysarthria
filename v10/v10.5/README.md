# Dysarthria V10.5 exploratory confound follow-ups

V10.5 adds prompt-matched duration analysis and speech-preserving background interventions to the frozen V10 speech-only detector. These are exploratory follow-ups, not original pre-specified analyses. V10.4 and earlier inputs remain read-only. The WavLM encoder, outer-fold heads, deployment heads, and thresholds are unchanged.

## Run

From the repository root:

```sh
.venv/bin/python -m unittest discover -s tests -p 'test_v10_5_followups.py' -v
.venv/bin/python v10/v10.5/run_v10_5.py all
```

Stages are `prompt_match`, `background_torgo`, and `background_uaspeech`. Individual modules can also be run directly. Outputs are under `v10/v10.5/outputs/`, with CSV source data and PNG/PDF figures. `analysis_plan.json` fixes the intervention and statistical recipe.

Prompt matching needs a local `v10/v10.5/torgo_prompts.csv` or `--prompt-map PATH`. Required columns are `source_id` (preferred) or `relative_path`, plus `prompt_id` or `transcript`; `session_id` is optional and otherwise parsed from the TORGO folder name. If both prompt ID and transcript are supplied, each ID must have one consistent normalized transcript. The local audio-only TORGO release has no verified prompt mapping. In that case the stage writes a coverage/status JSON and does not claim a matched result. An utterance sequence number is not evidence that speakers said the same prompt.

The matched analysis keeps only prompts recorded by all 15 TORGO physical speakers, selects one head-mic recording per speaker/prompt when present, compares raw and frozen-preprocessed duration, retrains the original four-statistic logistic baseline within the saved speaker-disjoint nested folds, and scores cached matched WavLM features with corresponding locked outer-fold heads. Incremental stacking is withheld because matched-subset inner-fold WavLM predictions do not exist.

For the intervention, three recordings per speaker are chosen from the existing V8 low-energy-edge selections, subject to a post-trim quiet-frame gate. Original, sham, neutralized, cross-source swapped, half-intensity, and double-intensity waveforms are encoded. Only low-energy samples in the actual WavLM-bound waveform change, with 5 ms fades to limit boundary clicks. The sham performs the same mask-based copy while restoring the original samples exactly. Each condition is scored by the saved TORGO held-out heads or the TORGO-trained UASpeech M3 deployment heads, with their locked thresholds. UASpeech labels are used only for final evaluation and stratified uncertainty.

`snr_proxy_db` is the foreground-frame RMS divided by quiet-frame RMS, not a source-separated SNR. Low-energy edges and frames may contain speech; the waveform integrity test proves preservation only outside the edited frames. Three utterances per speaker limit power. Five seeds repeat the same speakers, so intervals resample physical speakers rather than treating seeds as independent observations.
