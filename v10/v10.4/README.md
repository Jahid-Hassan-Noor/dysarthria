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
.venv/bin/python v10/v10.4/run_v10_4.py all
```

Individual resumable stages are `audit`, `reanalyses`, `background`, `spread`,
and `report`. The full run needs the frozen V8 feature caches and notebook,
the saved V10 reference folds, the local TORGO WAV files, and the WavLM Base
weight file. The first spread run performs new feature extraction. Subsequent
runs reuse per-speaker feature shards and completed background checkpoints.

The runner verifies SHA-256 hashes in `source_manifest.json` before and after
each run. All writes are under `v10/v10.4/`; V8, V9, V10, and V10.3 are read-only
inputs. `analysis_plan.json` specifies the comparisons and decision rules.

**Current rerun limitation:** Moving this folder changed its location, but
`common.py` still treats its parent as the project root, and the saved source
manifest records the earlier paths. The command above is the intended entry
point after those path assumptions are updated. The committed tables, figures,
and this results document can be read without a rerun.

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
