# Dysarthria V9 research notebook

Open [dysarthria_v9.ipynb](dysarthria_v9.ipynb). It contains the complete V9 implementation, analysis cells, 15 embedded figures, 22 embedded result tables, and the saved research audit. It does not import local V9 Python modules. Colleagues can read the results and charts without running the analysis.

To recompute results, open it from the project root or `v9/` with the project's `.venv` Python kernel. Raw TORGO and UASpeech audio, the frozen V8 reference and pretrained weights, and the installed scientific Python dependencies are still required. The notebook writes research artifacts to `outputs/cache/dysarthria_detection_v9/` and `v9_publication/`; those artifacts are not needed to view the embedded snapshots. Run code cells from top to bottom. Compatible caches let the workflow resume.

The user confirmed the label-independent 1.0-second fallback after the TORGO VAD audit showed the example 90% coverage rule was unattainable. The TORGO-only deployment lock selected K=20. Under that fixed protocol, the official UASpeech M3 cohort and all M2–M8 channels are non-estimable because too many physical speakers lack 20 eligible recordings. The notebook and audit state those limits; the embedded gallery contains only available results.
