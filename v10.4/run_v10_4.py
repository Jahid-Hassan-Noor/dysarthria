"""Run or resume all five isolated V10.4 analyses.

From the project root: .venv/bin/python v10.4/run_v10_4.py all
"""
from __future__ import annotations

import argparse

from common import HERE, read_json, source_audit, write_json


def audit() -> None:
    current = source_audit()
    path = HERE / "source_manifest.json"
    if path.is_file():
        previous = read_json(path)
        if previous != current:
            raise RuntimeError("A frozen input or the V10.4 analysis plan changed; inspect the source manifest")
    else:
        write_json(path, current)
    print(f"Verified {current['n_inputs']} read-only source artifacts", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage", choices=("audit", "reanalyses", "background", "spread", "report", "all"),
        nargs="?", default="all",
    )
    args = parser.parse_args()
    audit()
    if args.stage in ("reanalyses", "all"):
        from reanalyses import main as run_reanalyses
        run_reanalyses()
    if args.stage in ("background", "all"):
        from background import main as run_background
        run_background()
    if args.stage in ("spread", "all"):
        from spread import main as run_spread
        run_spread()
    if args.stage in ("report", "all"):
        from report import main as write_report
        write_report()
    audit()


if __name__ == "__main__":
    main()
