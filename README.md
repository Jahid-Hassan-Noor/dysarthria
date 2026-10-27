# Dysarthria detection research

This repository contains the code, executed notebooks, publication figures, and compact result tables for a sequence of dysarthria detection experiments. The main studies use speech recordings from **TORGO** for training and internal evaluation, and the official **UASpeech** corpus for locked external evaluation where the protocol permits it. The later versions focus on speaker-independent evaluation and on whether duration, recording conditions, or background audio can explain apparent discrimination.

**Start here:** Read the [V10.4 results](v10.4/RESULTS.md) for the latest follow-up analyses. The [V10 publication tables and figures](v10/publication/) summarize the frozen reference model and its comparisons. The notebooks retain executed outputs, so the saved findings can be inspected without downloading the datasets.

## Repository guide

| Path | Contents |
| --- | --- |
| [`v4/`](v4/)–[`v6/`](v6/) | Historical notebook experiments and their small saved figures/results. |
| [`v7/`](v7/) | Methodology repair, grouped speaker evaluation, external validation code, and selected publication outputs. |
| [`v8/`](v8/) | Main WavLM speech-only notebook, publication workflow code, figures, and tables. |
| [`v9/`](v9/) and [`v9_publication/`](v9_publication/) | Active-speech experiment, generated publication products, and feasibility audit. |
| [`v10/`](v10/) | Frozen V8-preserving reference study, diagnostics, and publication products. |
| [`v10.4/`](v10.4/) | Latest secondary analyses, executable Python modules, result summaries, and figures. |
| [`tests/`](tests/) | Repository-level publication checks; V8-specific tests are under [`v8/tests/`](v8/tests/). |

The version folders are retained because later analyses refer to frozen inputs and earlier protocols. The large local artifacts were left in place on the research machine but are excluded from Git: raw corpora, downloaded dataset shards, caches, extracted embeddings, pretrained weights, checkpoints, archives, and most recording-level predictions. The committed outputs are intended for reviewing results, not for resuming a full training run from a fresh clone.

## View the results

1. Open [`v10.4/RESULTS.md`](v10.4/RESULTS.md) for the latest result table and interpretation.
2. View the [V10 figures](v10/publication/figures/) and [V10 tables](v10/publication/tables/) for the main reference study.
3. Open [`v10/dysarthria_10.ipynb`](v10/dysarthria_10.ipynb), [`v9/dysarthria_v9.ipynb`](v9/dysarthria_v9.ipynb), or [`v8/dysarthria_v8.ipynb`](v8/dysarthria_v8.ipynb) in Jupyter to inspect the saved code, charts, and tables.
4. For earlier results, see the version READMEs and the [V8 publication products](v8/v8_publication_completion/publication/) and [V9 publication products](v9_publication/).

No dataset is needed to read these saved outputs. For a rerun, use your own authorized copies of TORGO and UASpeech and restore the frozen artifacts named in the version documentation.

## Set up a local environment

Run commands from the repository root. The dependency list was developed for a local Apple Silicon environment; PyTorch device selection in the notebooks falls back to CPU when MPS is unavailable.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements-mac.txt
.venv/bin/python -m jupyter lab
```

Set the dataset locations **before** starting a notebook kernel:

```bash
export TORGO_DATASET_PATH="/absolute/path/to/TORGO"
export UASPEECH_DATASET_PATH="/absolute/path/to/UASpeech"
```

`TORGO_DATASET_PATH` should point to the corpus folder containing `F_Con`, `F_Dys`, `M_Con`, and `M_Dys`. `UASPEECH_DATASET_PATH` should point to the official UASpeech directory with the normalized control and dysarthric channel folders. The V8 notebook documents the path defaults and other runtime options in [`v8/README.md`](v8/README.md). Set environment variables in the shell that launches Jupyter, or set them before the dataset-loading cell and restart the kernel.

For the primary historical workflow, open [`v8/dysarthria_v8.ipynb`](v8/dysarthria_v8.ipynb) and run cells in order. Its publication workflow and external evaluation require the raw corpora, the pretrained WavLM Base weights, and the local frozen-reference artifacts described in [`v8/README.md`](v8/README.md). Follow the version-specific README before attempting another version; these experiments are not interchangeable one-command runs.

The V10.4 command is:

```bash
.venv/bin/python v10.4/run_v10_4.py all
```

Its stages are `audit`, `reanalyses`, `background`, `spread`, and `report`. A **fresh GitHub clone cannot complete this command by itself**: [`v10.4/source_manifest.json`](v10.4/source_manifest.json) verifies hashes of frozen V8/V10 caches and checkpoints that are deliberately excluded from Git, and the spread analysis also needs local TORGO WAV files and WavLM weights. See [`v10.4/README.md`](v10.4/README.md) for the required inputs and resumable stages. The saved V10.4 tables and figures are available without rerunning it.

## Main findings

| Analysis | Saved result | What it supports |
| --- | --- | --- |
| V10 frozen TORGO reference | Mean five-seed speaker-level ROC-AUC **0.850** and balanced accuracy **0.854** across 15 physical speakers. | The reference separates speakers in this small, repeated speaker-held-out experiment. [Source](v10/publication/tables/table01_torgo_primary_v10.md) |
| Locked UASpeech M3 evaluation | Mean ROC-AUC **0.966** and balanced accuracy **0.860** across 28 speakers; sensitivity **0.720** at the locked threshold. | Cross-corpus discrimination is strong under the V8/V10 protocol, while thresholded sensitivity is lower. This is a reproducibility rerun of a previously evaluated corpus, not a newly acquired prospective cohort. [Source](v10/publication/tables/table05_uaspeech_m3.md) |
| V9 active-speech, fixed-evidence protocol | The TORGO experiment completed, but the locked UASpeech M3 benchmark was **not estimable** because 15 of 28 speakers lacked the required 20 eligible recordings. | No V9 UASpeech performance claim is made. [Source](v9_publication/README.md) |
| V10.4 duration increment | AUC **0.896** for a duration stack and **0.921** for duration plus V10; paired Δ **0.025**, 95% speaker-bootstrap interval **[-0.104, 0.196]**. | Incremental value beyond duration is not established. The original duration-only AUC was 0.964. [Source](v10.4/RESULTS.md) |
| V10.4 background and intervention controls | Background-edge classifier AUC **0.804** with a wide paired interval against speech heads; the duration-spread intervention was marked **inconclusive** by its artifact control. | Recording/background signal is a material concern; these tests do not identify a disease-specific mechanism. [Source](v10.4/RESULTS.md) |

The five seeds reuse the same 15 TORGO physical speakers, so they are repeated partitions of one cohort. Confidence intervals in V10.4 resample physical speakers, not seeds as independent people. A high AUC alone does not show that a model has learned a clinical dysarthria mechanism. The detailed caveats and exact comparisons are in the linked result documents.

## What GitHub contains

The `.gitignore` keeps datasets and machine-specific working files local. In particular, this repository does **not** distribute TORGO or UASpeech audio, the downloaded EasyCall data, model weights, training checkpoints, feature arrays, cache trees, or large per-recording prediction exports. Publication PNG/PDF figures, their source tables, compact metrics and audits, notebooks, and code are included. Some provenance manifests retain hashes and paths for excluded files so that a full local rerun can verify the original frozen inputs after they are restored.

The project is research software and its saved results are not a clinical diagnostic tool.
