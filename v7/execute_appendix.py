"""Execute only the appended V7 cells and retain original V6 cells verbatim."""
import json
from pathlib import Path

import nbformat
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/"v6"/"dysarthria_v6.ipynb"
TARGET=ROOT/"v7"/"dysarthria_v7_methodology_fixed.ipynb"


def main():
    old=json.loads(SOURCE.read_text())
    document=json.loads(TARGET.read_text())
    n_original=len(old["cells"])
    if document["cells"][:n_original]!=old["cells"]:
        raise AssertionError("Original V6 cells changed before V7 appendix execution")
    cells=[]
    for c in document["cells"][n_original:]:
        prepared=dict(c)
        if isinstance(prepared["source"],list):prepared["source"]="".join(prepared["source"])
        cells.append(nbformat.from_dict(prepared))
    appendix=nbformat.v4.new_notebook(cells=cells)
    appendix.metadata.kernelspec={"display_name":"Dysarthria project venv","language":"python","name":"dysarthria-v5"}
    executed=NotebookClient(appendix,timeout=None,kernel_name="dysarthria-v5",
                            resources={"metadata":{"path":str(ROOT)}}).execute()
    for original_cell,executed_cell in zip(document["cells"][n_original:],executed.cells):
        if original_cell["cell_type"]=="code":
            original_cell["execution_count"]=executed_cell.execution_count
            original_cell["outputs"]=json.loads(nbformat.writes(nbformat.v4.new_notebook(cells=[executed_cell])))["cells"][0]["outputs"]
    if document["cells"][:n_original]!=old["cells"]:
        raise AssertionError("Original V6 cells changed during V7 appendix execution")
    TARGET.write_text(json.dumps(document,ensure_ascii=False,indent=1)+"\n")
    print("Executed V7 appendix and preserved",n_original,"V6 cells exactly:",TARGET)


if __name__=="__main__":main()
