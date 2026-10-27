# Dysarthria V8 research notebook

Additional figures for the current manuscript are documented in [v8_research_completion/README.md](v8_research_completion/README.md). The notebook's final cell generates and displays them during **Run All** from frozen saved results.

Open [dysarthria_v8.ipynb](dysarthria_v8.ipynb) in the project's `.venv` kernel and select **Run All**. The notebook contains the TORGO experiment and a locked, zero-shot UASpeech external validation section. Both raw corpora and the Python dependencies are required for the full run. Install [requirements-mac.txt](../requirements-mac.txt) in a compatible Python environment.

The notebook works when Jupyter starts in the project root or `v8/`. A copied standalone notebook writes a local `outputs/` folder. Set dataset paths before starting the kernel when the defaults do not apply.

| Environment variable | Default | Purpose |
| --- | --- | --- |
| `TORGO_DATASET_PATH` | Local KaggleHub TORGO copy | Raw TORGO directory with `F_Con`, `F_Dys`, `M_Con`, and `M_Dys` |
| `UASPEECH_DATASET_PATH` | Project `UASpeech/` directory | Raw directory with `UASpeech_normalized_C` and `UASpeech_normalized_FM` |
| `DYSARTHRIA_V8_ENCODER` | `WAVLM_BASE` | `WAVLM_BASE` or `HUBERT_BASE` |
| `DYSARTHRIA_V8_FUSION` | `0` | Set `1` to run both the speech-only and acoustic-fusion arms |
| `DYSARTHRIA_V8_UASPEECH_CHANNELS` | `M3` | Include the prespecified primary M3 channel; optionally add `M2,M4,M5,M6,M7,M8` |

## TORGO experiment

V8 verifies raw TORGO WAVs, freezes pretrained speech-encoder weights, caches final-layer time statistics per utterance, trains a speaker-aware attention head, and evaluates with nested physical-speaker-held-out folds. Its optional fusion arm adds engineered acoustic features scaled using training speakers only. All features and models use V8's own `outputs/cache/dysarthria_detection_v8` namespace; no V4–V7 derived feature cache is read.

The saved default TORGO run is complete. It used WavLM Base, the speech-only arm, 17,630 usable utterances, 15 physical speakers and five seeds. The current summary and per-seed speaker metrics are under `outputs/cache/dysarthria_detection_v8/run_2058b417224f6290/`. The five seeds repeat partitions of the same 15 speakers, so they are not independent cohorts. The inner-fold threshold targets 0.90 validation sensitivity but does not guarantee that sensitivity on held-out speakers.

## UASpeech external validation

The final notebook section first trains one deployment head per configured seed and arm using **TORGO only**. Five grouped TORGO validation folds determine each head's epoch count and sensitivity-constrained threshold; all TORGO speakers train its final fit. The section saves checkpoint and scaler hashes in `deployment/v8_torgo_model_lock.json` before reading any UASpeech data. Subsequent runs verify and reuse that lock rather than selecting a new model from UASpeech outcomes.

UASpeech inventory and features come directly from the normalized control and dysarthric WAV releases. The prespecified primary evaluation is microphone **M3**, which contains 21,126 decodable recordings from 28 physical speakers in this release. Optional M2–M8 channel results are secondary. The external section uses UASpeech labels for source-label checks and final metrics, while applying every locked TORGO model and threshold without UASpeech training, fine-tuning, calibration, threshold selection, or model selection. It reports speaker-level predictions, per-seed metrics, physical-speaker bootstrap intervals, plots, and a zero-shot audit under the same run's `external_uaspeech/` directory. Feature extraction is resumable and may take substantial time.

The saved run contains completed **TORGO** and official **UASpeech M3** artifacts. The TORGO `v8_summary.json` describes the internal run; `external_uaspeech/v8_uaspeech_summary.json` describes the locked external evaluation.

## V8 publication completion

The notebook's final section calls `v8.publication.workflow` to resume the publication experiments and print the generated audit. The original successful run is separately preserved in `v8_publication_completion/frozen_reference/`. Its `snapshot_manifest.json` lists SHA256 hashes for the notebook, WavLM weights, all saved run files and five locked TORGO deployment checkpoints. The original run directory is not used for new results. Verify the snapshot with:

```bash
.venv/bin/python -m v8.publication.freeze verify --compare-originals
```

From the project root, the full resumable workflow is:

```bash
.venv/bin/python -m v8.publication.workflow primary ablations budgets baselines matched_content silence channels report
```

`v8_publication_completion/results/` holds speaker predictions, per-seed metrics and provenance for each analysis. `registry.json` points to those files. The four-arm ablation uses the same WavLM four-layer cache, original physical-speaker folds, training settings and threshold rule. Its B2 arm references the saved V8 predictions and is never retrained. The reference model uses attention pooling concatenated with mean speaker context; the controlled mean arms concatenate mean context with itself to retain classifier width. `v8_publication_completion/ablations/` also contains saved split membership, paired speaker-bootstrap differences and factorial effects.

Budget sensitivity uses 100 reproducible without-replacement draws for each finite budget and a common cohort of speakers with at least 50 recordings. Every inference keeps the original fold or deployment threshold. Duration and simple recording-artifact logistic controls are fit on TORGO physical-speaker folds and transferred to UASpeech without external fitting or tuning. Matched-content M3 uses verified official word transcripts and documents any excluded speakers. Background-only and M2–M8 feature extraction is resumable and can take substantial time; new cache shards include raw-audio content, WavLM weight and preprocessing signatures, and mismatches stop the workflow.

To prepare one secondary-channel cache independently, run `.venv/bin/python -m v8.publication.prefetch M2` from the project root (and likewise for M4–M8). Each channel uses a separate cache directory. The `channels` workflow stage later verifies and reuses these representations, then scores every available recording with the locked models.

`v8_publication_completion/publication/` contains six tables, nine figures as 300-DPI PNG and vector PDF, source CSVs, a traceability manifest and a final audit. Its limitations are generated from the saved results. In particular, strong duration or recording-artifact controls are reported as potential confounds; external M3 remains a locked zero-shot cross-corpus evaluation and was not historically unseen to this project.
The calibration figure uses count-scaled bin markers without joining sparse bins; its source CSV records the speaker count behind every point.

The last notebook cell also creates four additional diagnostic paper figures under `v8_research_completion/figures/` and displays them in the notebook. They use the frozen saved results and are quick to regenerate. A full **Run All** still executes the upstream TORGO, UASpeech, and publication stages, so its runtime depends on the local audio corpora and cache state.
