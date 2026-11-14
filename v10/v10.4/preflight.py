"""Read-only V10.4 preflight. Run from any directory with Python 3.12+.

    python v10/v10.4/preflight.py

No dataset, feature, checkpoint, or scientific result is created or replaced.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MANIFEST = HERE / "source_manifest.json"
REQUIREMENTS = ROOT / "requirements-mac.txt"
FROZEN_META = ROOT / "v8/v8_publication_completion/frozen_reference/run/features/usable_metadata.csv"
SUBSET = HERE / "outputs/spread/subset_manifest.csv"
SEEDS = (42, 43, 44, 45, 46)


def digest(path: Path) -> str:
    hashed = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hashed.update(block)
    return hashed.hexdigest()


def validate_manifest(data: dict, root: Path, *, hashes: bool = True) -> list[str]:
    if data.get("version") != "v10.4-1" or not isinstance(data.get("inputs"), list):
        raise ValueError("Unexpected V10.4 source-manifest format")
    if data.get("n_inputs") != len(data["inputs"]):
        raise ValueError("Source-manifest entry count differs from n_inputs")
    missing = []
    seen = set()
    for item in data["inputs"]:
        rel = item.get("path")
        if not isinstance(rel, str) or Path(rel).is_absolute() or ".." in Path(rel).parts:
            raise ValueError(f"Unsafe or invalid manifest path: {rel!r}")
        if rel in seen:
            raise ValueError(f"Duplicate manifest path: {rel}")
        seen.add(rel)
        if not isinstance(item.get("bytes"), int) or not isinstance(item.get("sha256"), str):
            raise ValueError(f"Incomplete manifest entry: {rel}")
        target = root / rel
        if not target.is_file():
            missing.append(rel)
            continue
        if target.stat().st_size != item["bytes"]:
            raise ValueError(f"Frozen input size changed: {rel}")
        if hashes and digest(target) != item["sha256"]:
            raise ValueError(f"Frozen input hash changed: {rel}")
    return missing


def validate_folds(root: Path, seeds: tuple[int, ...] = SEEDS) -> tuple[int, set[str]]:
    """Validate 25 saved physical-speaker outer/inner partitions."""
    total = 0
    cohort = None
    for seed in seeds:
        test_union = set()
        for outer in range(5):
            path = root / f"v10/outputs/reference_training/folds/speech_only/seed{seed}_outer{outer}/fold_audit.json"
            audit = json.loads(path.read_text())
            train = set(audit["train_physical_speakers"])
            test = set(audit["test_physical_speakers"])
            if (audit.get("seed"), audit.get("outer_fold")) != (seed, outer):
                raise ValueError(f"Fold identity differs: {path}")
            if not audit.get("complete") or len(train) != 12 or len(test) != 3 or train & test:
                raise ValueError(f"Speaker overlap or incomplete outer fold: {path}")
            if test_union & test:
                raise ValueError(f"Outer test speaker repeated within seed {seed}")
            test_union |= test
            validation_union = set()
            for inner in audit["inner_folds"]:
                inner_train = set(inner["train_physical_speakers"])
                validation = set(inner["validation_physical_speakers"])
                if inner_train & validation or inner_train | validation != train:
                    raise ValueError(f"Speaker overlap or incomplete inner fold: {path}")
                if validation_union & validation:
                    raise ValueError(f"Inner validation speaker repeated: {path}")
                validation_union |= validation
            if validation_union != train:
                raise ValueError(f"Inner folds do not cover outer training speakers: {path}")
            total += 1
        if cohort is None:
            cohort = test_union
        elif test_union != cohort:
            raise ValueError(f"Physical speaker cohort differs for seed {seed}")
    return total, cohort or set()


def validate_sources(metadata: Path, subset: Path, dataset_root: Path | None) -> tuple[int, int, list[str]]:
    """Check selected source IDs, microphone/speaker identity, and local audio paths."""
    with metadata.open(newline="") as stream:
        original = list(csv.DictReader(stream))
    with subset.open(newline="") as stream:
        selected = list(csv.DictReader(stream))
    source_ids = [row["source_id"] for row in original]
    if len(source_ids) != len(set(source_ids)):
        raise ValueError("Frozen metadata repeats a source ID across microphone channels")
    aliases = {}
    for row in original:
        alias = row["speaker_id"]
        physical = row["physical_speaker_id"]
        if alias in aliases and aliases[alias] != physical:
            raise ValueError(f"Microphone speaker alias maps to multiple physical speakers: {alias}")
        aliases[alias] = physical
    if len(selected) != 750 or len({row["source_id"] for row in selected}) != 750:
        raise ValueError("Spread subset must contain 750 unique source IDs")
    missing = []
    counts = {}
    for row in selected:
        index = int(row["frozen_index"])
        if index < 0 or index >= len(original):
            raise ValueError("Spread subset has an invalid frozen index")
        source = original[index]
        for key in ("source_id", "physical_speaker_id", "label", "mic_type"):
            if row[key] != source[key]:
                raise ValueError(f"Spread source mismatch at index {index}: {key}")
        speaker = row["physical_speaker_id"]
        counts[speaker] = counts.get(speaker, 0) + 1
        path = Path(source["file_path"])
        if not path.is_file() and dataset_root is not None:
            path = dataset_root / source["relative_path"]
        if not path.is_file():
            missing.append(source["relative_path"])
    if len(counts) != 15 or set(counts.values()) != {50}:
        raise ValueError("Spread subset does not have 50 recordings per physical speaker")
    return len(original), len(selected), missing


SUMMARY_FIELDS = {
    "threshold": ("original_mean_balanced_accuracy", "youden_mean_balanced_accuracy"),
    "fold_auc": ("n_folds", "mean_seed_pooled_auc_primary"),
    "duration_increment": ("mean_auc_by_arm", "incremental_delta_c"),
    "background": ("background_mean_auc", "matched_speech_mean_auc"),
    "spread": ("mean_auc_by_condition", "verdict"),
}


def validate_summary_schema(outputs: Path) -> int:
    for stage, fields in SUMMARY_FIELDS.items():
        path = outputs / stage / "summary.json"
        data = json.loads(path.read_text())
        if not isinstance(data, dict) or any(key not in data for key in fields):
            raise ValueError(f"Missing result fields in {path}")
        for key in fields:
            value = data[key]
            if isinstance(value, (int, float)) and not math.isfinite(value):
                raise ValueError(f"Non-finite result in {path}: {key}")
    return len(SUMMARY_FIELDS)


def check(root: Path = ROOT, *, hashes: bool = True) -> list[tuple[str, str, str]]:
    results = []
    def add(stage: str, status: str, detail: str) -> None:
        results.append((stage, status, detail))

    if sys.version_info[:2] == (3, 12):
        add("Python", "PASS", f"{sys.version.split()[0]} (validated major/minor version)")
    else:
        add("Python", "BLOCKED", f"{sys.version.split()[0]}; validated setup uses Python 3.12")

    pins = {}
    for line in (root / "requirements-mac.txt").read_text().splitlines():
        if "==" in line and not line.lstrip().startswith("#"):
            name, version = line.split("==", 1)
            pins[name.strip()] = version.strip()
    mismatches = []
    for name, version in pins.items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            installed = "missing"
        if installed != version:
            mismatches.append(f"{name}={installed} (requires {version})")
    add("Dependencies", "PASS" if not mismatches else "BLOCKED",
        f"{len(pins)} exact pins installed" if not mismatches else "; ".join(mismatches))

    if not (root / "v10/v10.4/analysis_plan.json").is_file():
        add("Configuration", "FAIL", "V10.4 analysis plan missing")
    else:
        plan = json.loads((root / "v10/v10.4/analysis_plan.json").read_text())
        add("Configuration", "PASS" if tuple(plan.get("seeds", ())) == SEEDS else "FAIL",
            "Five frozen seeds and analysis plan present" if tuple(plan.get("seeds", ())) == SEEDS else "Unexpected seeds")

    path = root / "v10/v10.4/source_manifest.json"
    if not path.is_file():
        add("Frozen inputs", "BLOCKED", "source_manifest.json missing")
    else:
        try:
            manifest = json.loads(path.read_text())
            missing = validate_manifest(manifest, root, hashes=hashes)
            add("Frozen inputs", "BLOCKED" if missing else "PASS",
                f"{len(manifest['inputs'])} entries checked; {len(missing)} missing" +
                (": " + ", ".join(missing[:5]) if missing else ""))
        except (ValueError, KeyError, json.JSONDecodeError) as exc:
            add("Frozen inputs", "FAIL", str(exc))

    try:
        folds, cohort = validate_folds(root)
        add("Speaker splits", "PASS", f"{folds} outer folds; {len(cohort)} physical speakers; inner/outer disjoint")
    except FileNotFoundError:
        add("Speaker splits", "BLOCKED", "Saved V10 fold audits missing")
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        add("Speaker splits", "FAIL", str(exc))

    tor = os.environ.get("TORGO_DATASET_PATH")
    tor_root = Path(tor).expanduser() if tor else None
    if tor_root is not None and not tor_root.is_dir():
        add("TORGO configuration", "FAIL", f"TORGO_DATASET_PATH is not a directory: {tor_root}")
    else:
        add("TORGO configuration", "PASS", "Configured directory exists" if tor_root else "Using frozen metadata paths")
    try:
        count, selected, missing_audio = validate_sources(root / FROZEN_META.relative_to(ROOT),
                                                           root / SUBSET.relative_to(ROOT), tor_root)
        add("TORGO source IDs/audio", "BLOCKED" if missing_audio else "PASS",
            f"{count} frozen records; {selected} selected IDs validated; {len(missing_audio)} audio files missing" +
            (": " + ", ".join(missing_audio[:3]) if missing_audio else ""))
    except FileNotFoundError:
        add("TORGO source IDs/audio", "BLOCKED", "Frozen metadata or saved spread subset missing")
    except (ValueError, KeyError, IndexError) as exc:
        add("TORGO source IDs/audio", "FAIL", str(exc))

    try:
        with tempfile.NamedTemporaryFile(dir=root / "v10/v10.4/outputs"):
            pass
        add("Output directory", "PASS", "V10.4 output directory is writable")
    except OSError as exc:
        add("Output directory", "FAIL", str(exc))
    try:
        count = validate_summary_schema(root / "v10/v10.4/outputs")
        add("Saved result schema", "PASS", f"{count} stage summaries validated")
    except FileNotFoundError:
        add("Saved result schema", "BLOCKED", "One or more V10.4 stage summaries missing")
    except (ValueError, KeyError, json.JSONDecodeError) as exc:
        add("Saved result schema", "FAIL", str(exc))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-hash", action="store_true", help="Check file presence and sizes without SHA-256")
    args = parser.parse_args()
    results = check(hashes=not args.no_hash)
    for stage, status, detail in results:
        print(f"{status:7} {stage}: {detail}")
    return 1 if any(status != "PASS" for _, status, _ in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
