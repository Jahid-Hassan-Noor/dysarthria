# Reproducibility and manuscript audit

Audit date: 2026-11-14. The code and data checks below distinguish the local
research machine from a fresh public GitHub clone. No licensed dataset, model
weight, frozen output, seed, or partition was changed by this audit.

## Issue audit

| Issue | Current status | Evidence | Required action |
| --- | --- | --- | --- |
| V10.4 repository root and analysis-plan path | PASS | `common.py` uses `HERE.parents[1]`; `source_manifest.json` points to `v10/v10.4/analysis_plan.json`; all 159 entries pass locally. | None for this path repair. |
| Full V10.4 rerun on the research machine | PASS | `.venv/bin/python v10/v10.4/run_v10_4.py all` completed; frozen feature and 75 locked-probability checks reproduced saved values. | Preserve the frozen inputs. |
| Full V10.4 rerun from a fresh clone | BLOCKED | 82 of 159 manifest paths are excluded from Git, including V8 feature caches, duration-fold inputs, 25 final checkpoints, and WavLM weights. | Obtain authorized local copies and restore them at the manifest paths. No dataset download occurs in CI. |
| V10 reference training and fresh UA-Speech evaluation | NOT TESTED | The V10 notebook retains executed results, but the full V10 sequence was not rerun in this audit. | Run only with authorized TORGO/UA-Speech and frozen reference artifacts; record a separate validation log. |
| Dependency versions | PASS for this local environment | 16 exact pins are installed under Python 3.12.14; `pip check` and import smoke test pass; all 16 releases exist on PyPI. | Keep the local environment record distinct from a complete historical lockfile. |
| Portable full dependency installation | NOT TESTED | The complete Mac list was not freshly installed on Linux/Windows or in an empty Mac environment. | Test the target platform before promising cross-platform reproduction. |
| Lightweight synthetic tests | PASS locally | Full local environment: 16 tests. Newly created Python 3.12 environment with only `requirements-ci.txt`: 10 synthetic tests and `pip check` passed. | Observe the first GitHub Actions run after publishing the workflow. |
| Frozen source and split integrity | PASS locally | Preflight checked SHA-256/size for 159 entries, 25 disjoint outer/inner folds, 15 physical speakers, 17,630 frozen metadata records, and 750 selected source IDs/audio files. | Preserve source versions and fold audits. |
| Manuscript source | BLOCKED for direct PDF revision | The supplied manuscript and supplement are PDFs; no matching editable source was found. | Apply `MANUSCRIPT_REVISION_SUGGESTIONS.md` to the actual LaTeX/Word source, then regenerate and visually review PDFs. |
| Private data in Git | PASS for raw datasets; unresolved for path strings | No raw audio, dataset archive, or model-weight file appears in the tracked tree or reachable Git object names. Existing tracked notebooks/manifests contain local absolute path strings in 27 files. | Review a separate public-release sanitization plan before changing frozen provenance or rewriting published history. |

## Reproduction stages

| Stage | Command | Required inputs | Verified status |
| --- | --- | --- | --- |
| Environment | `python3 -m venv .venv` then `.venv/bin/python -m pip install -r requirements-mac.txt` | Python 3.12, package index, supported platform | NOT TESTED as a fresh full install; existing environment PASS |
| Lightweight tests | `.venv/bin/python -m unittest -q tests.test_v10_4_reproducibility tests.test_publication_ablations` | Existing local environment | PASS (16 tests) |
| Clean synthetic-test install | Create a Python 3.12 venv; `python -m pip install -r requirements-ci.txt`; `python -m unittest -q tests.test_v10_4_reproducibility` | PyPI; no audio or weights | PASS (10 tests, `pip check` clean) |
| V10.4 preflight | `.venv/bin/python v10/v10.4/preflight.py` | Pinned environment; saved inputs; TORGO audio | PASS locally; BLOCKED in a fresh clone |
| Manifest validation | `.venv/bin/python v10/v10.4/run_v10_4.py audit` | 159 manifest paths | PASS locally; BLOCKED in a fresh clone |
| V8/V10 preprocessing and feature extraction | Historical executed notebooks and frozen caches | Authorized TORGO, WavLM Base weights | NOT TESTED as a fresh extraction in this audit |
| V10 training and locked external evaluation | `v10/dysarthria_10.ipynb` | Authorized TORGO and UA-Speech, V8 frozen reference | NOT TESTED in this audit |
| V10.4 reanalyses, background, spread, tables, figures | `.venv/bin/python v10/v10.4/run_v10_4.py all` | Frozen V8/V10 inputs and TORGO audio | PASS locally; BLOCKED in a fresh clone |

`v10/v10.4/preflight.py` performs read-only scientific checks and creates only
a temporary output-permission probe that is removed immediately. It does not
regenerate a missing source manifest. `run_v10_4.py audit` checks the original
manifest against current inputs and refuses a mismatch. The runner writes new
results only under `v10/v10.4/`.

## Environment and integrity evidence

- Local interpreter: Python 3.12.14 on Apple Silicon. The 16 exact packages in
  `requirements-mac.txt` match the installed distributions. `pip check` found
  no broken requirements; imports of NumPy, pandas, SciPy, scikit-learn,
  PyTorch, TorchAudio, librosa, soundfile, and Matplotlib passed.
- All exact releases in `requirements-mac.txt` were found on the [PyPI project
  index](https://pypi.org/). The key frozen preprocessing versions in the V8
  run manifest are librosa 1.0.0 and TorchAudio 2.11.0. [TorchAudio's PyPI
  release notes](https://pypi.org/project/torchaudio/) state that 2.11 supports
  later `torch` releases, including the locally installed 2.14.0. The rest of
  the original research environment is incompletely recorded, so the pinned
  file describes a validated modern local execution environment.
- Local preflight: 159/159 manifest entries present with matching sizes and
  SHA-256 hashes; 25 outer folds validated; five seeds reuse the same 15
  physical speakers. The spread subset has 50 source IDs per speaker and
  matches frozen metadata by source ID, physical speaker, label, and mic type.
- The V10 notebook describes UA-Speech as locked evaluation and records no
  UA-Speech use for training, fine-tuning, threshold selection, calibration,
  or model selection. The V10.4 code reads frozen TORGO fold assignments and
  resamples physical speakers in its paired bootstrap; this audit did not
  retrain the V10 reference model or independently reconstruct its freeze.
- The full V10.4 rerun reproduced frozen WavLM embeddings (relative maximum
  difference 0) and all 75 locked V10 probabilities (maximum difference
  1.11e-16). The saved spread verdict remains `inconclusive_artifact_control`.
  Background-edge segments may contain speech, and the paired AUC interval
  includes zero. No new clinical mechanism claim follows from either result.
- The V10.4 `provenance.json` records hashes of the analysis plan, source
  manifest, six analysis files, and six result summaries. Preflight verifies
  source hashes; the existing full rerun regenerated the report and provenance.

## Manuscript and supplement consistency

The supplied PDFs were read as evidence. The replacement text is in
`MANUSCRIPT_REVISION_SUGGESTIONS.md`; the manuscript PDF itself was not edited.
The main abstract's TORGO/UA-Speech ROC-AUC values (0.850/0.966) agree with
the saved primary tables. The TORGO duration-only ROC-AUC (0.964), the
original-fold increment (+0.025, interval [-0.104, 0.196]), the background
comparison (+0.082, interval [-0.125, 0.311]), and the inconclusive spread
control agree with V10.4 summaries and supplementary Table S8. The main
Table 10 mixes availability, sufficiency, restricted-input response, and
ordinary-speech reliance; the proposed compact table separates them. The
abstract and conclusion revisions preserve the finding that tempo reliance
is inconclusive under processing artifacts and describe UA-Speech as locked
cross-corpus testing, not a new prospective cohort.

No numerical discrepancy requiring an experiment rerun was identified in
these checked values. This was a targeted consistency check, not a complete
line-by-line verification of every table and figure in the 33-page manuscript
and 10-page supplement.

## Privacy and remaining limits

The local licensed `UASpeech/` directory and downloaded `easycall-dysarthria/`
directory are ignored by Git, as are raw audio formats, feature arrays, model
weights, caches, and checkpoints. A reachable-object path scan found no raw
dataset or model-weight file in repository history. Compact derived metrics,
speaker-level outputs, figures, and manifests are already tracked. Their
redistribution terms require a separate rights review; this audit does not
interpret the dataset license.

Twenty-seven previously tracked files contain local absolute path strings.
These are path/provenance text, not the underlying audio. Removing them from
the current tree would not remove copies already in Git history, and editing
frozen manifests could invalidate hashes. No history rewrite or frozen-data
sanitization was performed. A fresh-clone full rerun remains blocked until the
82 excluded manifest artifacts and licensed audio are restored locally.
