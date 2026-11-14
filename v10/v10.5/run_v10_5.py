"""Run exploratory V10.5 follow-up analyses without changing frozen V10.4 inputs."""
from __future__ import annotations

import argparse
from pathlib import Path

from support import OUT, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prompt_match", "background_torgo", "background_uaspeech", "report", "all"),
                        nargs="?", default="all")
    parser.add_argument("--prompt-map", type=Path, default=None)
    args = parser.parse_args()
    outputs = {}
    if args.stage in ("prompt_match", "all"):
        from prompt_match import PROMPT_MAP, run
        outputs["prompt_match"] = run(args.prompt_map or PROMPT_MAP)
    if args.stage in ("background_torgo", "all"):
        from background_intervention import run_corpus
        outputs["background_torgo"] = run_corpus("TORGO")
    if args.stage in ("background_uaspeech", "all"):
        from background_intervention import run_corpus
        outputs["background_uaspeech"] = run_corpus("UASpeech")
    if args.stage in ("report", "all"):
        from report import main as report
        outputs["report"] = report()
    summary_path = OUT / "run_summary.json"
    if args.stage != "all" and summary_path.is_file():
        previous = read_json(summary_path)
        previous.update(outputs)
        outputs = previous
    write_json(summary_path, outputs)
    print(outputs, flush=True)


if __name__ == "__main__":
    main()
