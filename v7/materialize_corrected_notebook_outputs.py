"""Execute only the appended corrected V7 cells and keep historical cells intact.

The research functions are resumable and read the completed V7 result files. This
script runs after the long architecture comparison has finished, so the existing
notebook displays its final tables and audit without replaying V6 or earlier V7.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import nbformat
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "v7" / "dysarthria_v7_methodology_fixed.ipynb"
FIRST_CORRECTED_CELL = 123


def materialize():
    original_raw = json.loads(NOTEBOOK.read_text())
    historical_fingerprint = json.dumps(
        original_raw["cells"][:FIRST_CORRECTED_CELL], sort_keys=True,
        ensure_ascii=False)
    original = nbformat.read(NOTEBOOK, as_version=4)
    if len(original.cells) != 149:
        raise RuntimeError("Unexpected V7 cell count; refusing to rewrite notebook")
    indices = [index for index in range(FIRST_CORRECTED_CELL, len(original.cells))
               if original.cells[index].cell_type == "code"]
    scratch = nbformat.v4.new_notebook(
        cells=[copy.deepcopy(original.cells[index]) for index in indices],
        metadata=copy.deepcopy(original.metadata),
    )
    executed = NotebookClient(
        scratch, timeout=7200, allow_errors=False, kernel_name="dysarthria-v5",
        resources={"metadata": {"path": str(ROOT)}},
    ).execute()
    latest = json.loads(NOTEBOOK.read_text())
    if len(latest["cells"]) != len(original.cells) or json.dumps(
            latest["cells"][:FIRST_CORRECTED_CELL], sort_keys=True,
            ensure_ascii=False) != historical_fingerprint:
        raise RuntimeError("Historical notebook cells changed during execution")
    for index, completed in zip(indices, executed.cells, strict=True):
        source = latest["cells"][index]["source"]
        source_text = "".join(source) if isinstance(source, list) else source
        if source_text != original.cells[index].source:
            raise RuntimeError(f"Corrected cell {index} changed during execution")
        latest["cells"][index]["outputs"] = completed.outputs
        latest["cells"][index]["execution_count"] = completed.execution_count
    nbformat.validate(nbformat.reads(json.dumps(latest), as_version=4))
    temporary = NOTEBOOK.with_suffix(".ipynb.tmp")
    temporary.write_text(json.dumps(latest, ensure_ascii=False, indent=1) + "\n")
    os.replace(temporary, NOTEBOOK)
    return dict(executed_corrected_cells=len(indices),
                historical_cells_preserved=FIRST_CORRECTED_CELL,
                final_audit_output_count=len(latest["cells"][-1]["outputs"]))


if __name__ == "__main__":
    print(materialize())
