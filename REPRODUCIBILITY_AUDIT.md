# Reproducibility and manuscript audit

Audit date: 2026-11-14. The code and data checks below distinguish the local
research machine from a fresh public GitHub clone. No licensed dataset, model
weight, frozen output, seed, or partition was changed by this audit.

## A. Environment

| Item | Declared | Verified on this machine |
| --- | --- | --- |
| Operating system and device | Local Apple Silicon setup | macOS 27.0.1, arm64; PyTorch MPS available |
| Python and package manager | Python 3.12; pinned `requirements-mac.txt` | Python 3.12.14; pip 26.2.1 |
| PyTorch and TorchAudio | `torch==2.14.0`; `torchaudio==2.11.0` | 2.14.0; 2.11.0; both imported successfully |
| NumPy, pandas, scikit-learn | 2.5.3; 3.0.5; 1.9.1 | 2.5.3; 3.0.5; 1.9.1 |
| Installation outcome | 16 exact pins in `requirements-mac.txt` | Existing local environment matches all 16; `pip check` passed. A fresh environment with only `requirements-ci.txt` installed and passed 10 synthetic tests. A fresh full install was NOT TESTED. |

The V8 frozen run manifest records librosa 1.0.0 and TorchAudio 2.11.0. It does
not record a complete historical package lock. This is a validated current
Apple Silicon execution environment, not proof of identical historical
dependency resolution or Linux CPU/CUDA support.

## B. Execution results

Commands were run from the repository root unless stated otherwise. `PASS`
means the command exited 0 on this machine; saved notebook output alone is
never counted as a newly executed stage.

| Stage | Command | Status | Evidence | Blocker |
| --- | --- | --- | --- | --- |
| Syntax and imports | `.venv/bin/python -m compileall -q v10/v10.4 tests/test_v10_4_reproducibility.py`; import V10.4 modules | PASS | Exit 0; `V10.4 imports OK`; `common.ROOT` resolved to this repository. | None locally. |
| Dependency validation | `.venv/bin/python -m pip check`; compare installed versions with `requirements-mac.txt` | PASS | Exit 0; no broken requirements; 16 exact pins installed. | Full clean install and non-Mac platforms NOT TESTED. |
| Root-path and configuration | `.venv/bin/python v10/v10.4/preflight.py` | PASS | Exit 0; Python, package pins, analysis-plan seeds and paths passed. | Fresh public clone lacks frozen inputs. |
| Manifest validation | `.venv/bin/python v10/v10.4/run_v10_4.py audit` | PASS | Exit 0; 159 read-only source artifacts verified before and after. | 82 excluded manifest paths in a fresh clone. |
| Frozen artifact validation | `.venv/bin/python v10/v10.4/preflight.py` | PASS | Exit 0; 159 sizes/hashes, 25 disjoint folds, 750 selected source IDs/audio files, and five summary schemas passed. | Authorized audio and excluded caches/checkpoints needed elsewhere. |
| V10 saved-reference validation | Read-only hash/CSV check against `source_manifest.json` and `v10_reference_reproduction.json` | PASS | Exit 0; 75 speaker rows, frozen splits/seeds and saved equivalence record checked. | Fresh V10 retraining and external evaluation NOT TESTED. |
| V10 reference retraining | No command run | NOT TESTED | Original trained heads and metrics left intact. | New training was outside this audit and would require a separate authorized run. |
| V10.4 analyses | `.venv/bin/python v10/v10.4/run_v10_4.py all` | PASS | Exit 0; 75 locked probabilities reproduced to maximum absolute difference 1.11e-16; spread verdict remained inconclusive. | Fresh public clone remains BLOCKED. |
| Result generation | `all` runs `report.py` after its analysis stages | PASS | Exit 0; RESULTS, notebook, figures and provenance generated; five stage-summary schemas passed. | None locally. |
| Tests | `.venv/bin/python -m unittest -q tests.test_v10_4_reproducibility tests.test_publication_ablations` | PASS | Exit 0; 16 tests. Clean lightweight environment: 10 tests passed. | None for the synthetic suite. |
| Remote CI | GitHub Actions `Reproducibility checks`, run 43566370131 | PASS | Published via SSH; synthetic-tests job and all steps passed. | Licensed-data reproduction remains BLOCKED in CI. |

## C. Artifact integrity

The 159-entry source manifest was checked in full. Representative values below
are copied from that manifest or the committed V10.4 provenance, then
recomputed from accessible local files in this pass.

| Artifact | Expected SHA-256 | Computed SHA-256 | Outcome |
| --- | --- | --- | --- |
| `v10/v10.4/analysis_plan.json` | `0a1c550800eb4ed842a202e7115fbdd9e15c039def835c04b257ac2e9bf42223` | `0a1c550800eb4ed842a202e7115fbdd9e15c039def835c04b257ac2e9bf42223` | PASS |
| `v10/v10.4/source_manifest.json` | `94995b4c163e78e2a3ebc2b084d0c33ec4b0a5d37d7ff1eb248daaea6b6995dc` | `94995b4c163e78e2a3ebc2b084d0c33ec4b0a5d37d7ff1eb248daaea6b6995dc` | PASS |
| Frozen WavLM Base weight (`v8/.../pretrained/wavlm_base.pth`) | `ee69baa3f5f2dd37493516acd5a5018797dd9fa25fedafc6c7f7ce53e7bd6ce3` | `ee69baa3f5f2dd37493516acd5a5018797dd9fa25fedafc6c7f7ce53e7bd6ce3` | PASS |
| V10 seed42 outer0 final checkpoint | `7f6523cd3a4c7d47fe031915c0fdd24c7a9ebc7ba04b6afd391105e956107b89` | `7f6523cd3a4c7d47fe031915c0fdd24c7a9ebc7ba04b6afd391105e956107b89` | PASS |
| V10 reference speaker OOF CSV | `eb829bf2910097c910c7f3e302a5798d44708ec741813513cf4a22a9db063cbe` | `eb829bf2910097c910c7f3e302a5798d44708ec741813513cf4a22a9db063cbe` | PASS |

The V10.4 runner's post-run audit again verified all 159 inputs. The Git diff
contains no changed original frozen weights, V8/V10 fold manifests, or V10
reference predictions. Generated PDF figures can acquire metadata-only binary
differences on a rerun; these were restored to their committed versions after
checking that the numeric summaries and provenance remained unchanged.

## D. Remaining limitations

- A fresh public clone is BLOCKED for the full experiment: 82 of the 159
  manifest entries are excluded, including features, duration-fold inputs,
  checkpoints and WavLM weights. Licensed audio also must be supplied locally.
- V10 reference retraining, fresh V8/V10 feature extraction, and a new
  prospective UA-Speech cohort were NOT TESTED. The 75-row saved V10 reference
  validation is a read-only integrity check, not new model training.
- The matching editable manuscript/supplement source is unavailable. Proposed
  text and tables are complete in `MANUSCRIPT_REVISION_SUGGESTIONS.md`, but PDF
  integration and visual verification remain BLOCKED.
- Remote CI passed its dataset-free synthetic tests on run 43566370131. It
  cannot execute the licensed-data or frozen-artifact experiments.
- Existing tracked provenance contains 27 files with local absolute path
  strings. Raw datasets and model weights are not tracked; path sanitization
  or history rewriting was outside this focused scientific audit.

## Issue audit

| Issue | Current status | Evidence | Required action |
| --- | --- | --- | --- |
| V10.4 repository root and analysis-plan path | PASS | `common.py` uses `HERE.parents[1]`; `source_manifest.json` points to `v10/v10.4/analysis_plan.json`; all 159 entries pass locally. | None for this path repair. |
| Full V10.4 rerun on the research machine | PASS | `.venv/bin/python v10/v10.4/run_v10_4.py all` completed; frozen feature and 75 locked-probability checks reproduced saved values. | Preserve the frozen inputs. |
| Full V10.4 rerun from a fresh clone | BLOCKED | 82 of 159 manifest paths are excluded from Git, including V8 feature caches, duration-fold inputs, 25 final checkpoints, and WavLM weights. | Obtain authorized local copies and restore them at the manifest paths. No dataset download occurs in CI. |
| V10 reference training and fresh UA-Speech evaluation | NOT TESTED | The V10 notebook retains executed results, but the full V10 sequence was not rerun in this audit. | Run only with authorized TORGO/UA-Speech and frozen reference artifacts; record a separate validation log. |
| Dependency versions | PASS for this local environment | 16 exact pins are installed under Python 3.12.14; `pip check` and import smoke test pass; all 16 releases exist on PyPI. | Keep the local environment record distinct from a complete historical lockfile. |
| Portable full dependency installation | NOT TESTED | The complete Mac list was not freshly installed on Linux/Windows or in an empty Mac environment. | Test the target platform before promising cross-platform reproduction. |
| Lightweight synthetic tests | PASS locally and on GitHub Actions | Full local environment: 16 tests. Newly created Python 3.12 environment with only `requirements-ci.txt`: 10 synthetic tests and `pip check` passed. GitHub Actions run 43566370131 passed. | CI does not cover licensed-data experiments. |
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

| Result | Manuscript / supplement | Saved output checked | Reconciliation |
| --- | --- | --- | --- |
| TORGO detector ROC-AUC | 0.850; abstract interval [0.679, 0.979] | V10.2 canonical primary 0.850 [0.678571, 0.978571], 15 speakers | Matches after rounding. |
| UA-Speech M3 detector ROC-AUC | 0.966 | V10 table05 0.966154, 28 speakers | Matches after rounding; locked cross-corpus evaluation. |
| Duration-only ROC-AUC | TORGO 0.964; UA-Speech 1.000 | V8 duration per-seed means 0.964286 and 1.000, respectively | Matches after rounding. |
| Locked silence-only ROC-AUC | TORGO 0.461; UA-Speech 0.927 | V8 silence per-seed means 0.460714 and 0.927179 | Matches after rounding; restricted input does not establish ordinary-speech reliance. |
| TORGO edge-trained classifier ROC-AUC | 0.804 | V10.4 background summary 0.803571 [0.564, 1.000] | Matches after rounding; low-energy edges may contain speech. |
| Original-fold increment over duration | +0.025 [-0.104, 0.196] | V10.4 duration summary +0.025 [-0.103661, 0.196429] | Matches after rounding; exploratory and imprecise. |
| TORGO background vs matched speech | +0.082 [-0.125, 0.311] | V10.4 background summary +0.082143 [-0.125089, 0.310804] | Matches after rounding; interval includes zero. |
| Duration-spread intervention | Inconclusive | V10.4 verdict `inconclusive_artifact_control` | Matches; processing control exceeded the planned gate. |

One **versioned interval difference** remains in repository outputs: the older
`v10/publication/tables/table01_torgo_primary_v10.csv` gives TORGO ROC-AUC
0.850 [0.693, 0.975], while the V10.2 canonical bootstrap gives 0.850
[0.679, 0.979]. The manuscript uses the latter. The point estimate and 15
speaker cohort agree; these are distinct saved bootstrap summaries, so neither
interval was overwritten. Any final typeset table should identify the chosen
canonical version and avoid mixing the two intervals.

This was a targeted consistency check of the primary claims, cohorts,
intervals and mechanism controls, not a complete line-by-line verification of
every table and figure in the 33-page manuscript and 10-page supplement. No
checked discrepancy requires retraining.

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
