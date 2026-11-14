# Dysarthria V10.4

V10.4 completes five secondary analyses of the frozen V10 speech-only detector:
mean-preserving duration-spread intervention, a classifier trained on objective
background-edge segments, a Youden J threshold sensitivity check, per-fold
ROC-AUC, and incremental ROC-AUC beyond the duration model. The results and
their limits are in [RESULTS.md](RESULTS.md). The executable notebook is
[dysarthria_v10.4.ipynb](dysarthria_v10.4.ipynb).

## Run

From the project root, using this project's environment:

```sh
.venv/bin/python v10/v10.4/preflight.py
.venv/bin/python v10/v10.4/run_v10_4.py all
```

The preflight checks pinned Python packages, the analysis plan, 159 manifest
entries and hashes, 25 saved speaker-disjoint folds, the 750 selected TORGO
source IDs and audio files, output permissions, and saved summary schemas. It
returns a nonzero exit code for missing licensed inputs or a mismatch. Use
`--no-hash` only for a quick presence check; the full runner always checks
hashes. No UA-Speech audio is needed for these secondary V10.4 stages.

Individual resumable stages are `audit`, `reanalyses`, `background`, `spread`,
and `report`. The full run needs the frozen V8 feature caches and notebook,
the saved V10 reference folds, the local TORGO WAV files, and the WavLM Base
weight file. The first spread run performs new feature extraction. Subsequent
runs reuse per-speaker feature shards and completed background checkpoints.

The runner verifies SHA-256 hashes in `source_manifest.json` before and after
each run. All writes are under `v10/v10.4/`; V8, V9, V10, and V10.3 are read-only
inputs. `analysis_plan.json` specifies the comparisons and decision rules.

The runner resolves paths from its location and verifies the 159 manifest
entries before running. A fresh GitHub clone does not contain the excluded
frozen inputs; restore them at the paths in `source_manifest.json` before a
full rerun. The committed tables, figures, and results can be read without
those inputs.

A GitHub Actions workflow for synthetic speaker-split and statistic tests is
prepared locally but has not been published because the available GitHub token
lacks the `workflow` scope. Remote CI has not run. It will not download licensed
data or model weights when published. The local research machine passed the
full `all` command; this does not make a fresh public clone self-contained.

## Files

- `reanalyses.py`: threshold, fold AUC, and duration incremental analysis.
- `background.py`: nested training on 750 cached background-edge features and
  matched-recording speech scoring.
- `spread.py`: label-blind audio selection, new WavLM extraction, locked-model
  scoring, and artifact controls.
- `outputs/`: source predictions, metrics, model checkpoints, feature shards,
  and PNG/PDF figures.
- `provenance.json`: hashes of analysis code, the plan, source manifest, and
  result summaries.

The background-edge segments may contain speech. The spread result is
mechanistically inconclusive because the same-length audio-processing control
had a larger effect than the spread manipulation. These are findings to report,
not failures to be hidden or overwritten.
