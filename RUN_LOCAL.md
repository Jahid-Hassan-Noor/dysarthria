# Run Dysarthria V4 or V5 locally

Open this folder in VS Code, open `v5/dysarthria_v5.ipynb`, and select
**Python 3 (dysarthria-v5)** in the notebook kernel picker. V4 is at
`v4/dysarthria-v4.ipynb` and still uses **Python 3 (dysarthria-v4)**. Both
kernels use this project's `.venv`.

Restart the kernel and run cells in order. The runtime check should show
`Device : mps`, batch size 16, and this project's `.venv/bin/python`.
The notebook automatically uses CPU if MPS is unavailable.

```zsh
cd /path/to/dysarthia
source .venv/bin/activate
jupyter lab v5/dysarthria_v5.ipynb
```

Dependencies are already installed on this Mac. To recreate the environment:

```zsh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mac.txt
.venv/bin/python -m ipykernel install --user --name dysarthria-v5 --display-name 'Python 3 (dysarthria-v5)'
```

TORGO is already downloaded in KaggleHub's local cache. The notebook reuses
that download. To use another local TORGO copy, set this in a cell before
the dataset-acquisition cell (the `UASpeech` folder is a different corpus):

```python
import os
os.environ['TORGO_DATASET_PATH'] = '/absolute/path/to/TORGO'
```

V5 has independent feature caches, metadata, splits, scalers, fitted models,
training checkpoints, and evaluation caches under `v5/backups/`:

```text
v5/backups/dysarthria_detection_v5/
```

Its pipeline signature is `v5_resumable_pipeline_1`, its configuration version
is `V5`, and its main checkpoints are named `v5_pretrain_*` and `v5_proposed_*`.
The first run builds fresh features and trains fresh models. It never reads
V4's derived caches or checkpoints. Only the raw TORGO audio is reused as
input. V4's notebook and cache files remain untouched.

Set `DYSARTHRIA_V5_CACHE_ROOT` before the configuration cell to override the
V5 location. The V4 cache environment variable is ignored. Batch size defaults
to 16 (`DYSARTHRIA_BATCH_SIZE`); changing it creates a different run ID.
Checkpoint interval defaults to 25 batches (`DYSARTHRIA_CHECKPOINT_INTERVAL`).

Restart the kernel before running the updated notebook to clear any old
in-memory data or models. Subsequent restarts resume only V5's own on-disk
cache. No training has been started as part of this setup.

V5 figures and its final results JSON are saved in `v5/outputs/`. Before each
run, the notebook also copies the saved notebook into `v5/backups/` with a
timestamp. V5's severity mapping remains an
optional TODO and requires your verified speaker-to-severity mapping.
