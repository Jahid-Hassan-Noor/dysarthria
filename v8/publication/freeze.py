"""Independent, read-only snapshots of a completed V8 run; never executes a notebook.

Usage: python -m v8.publication.freeze freeze [--project-root PATH]
       python -m v8.publication.freeze verify [--compare-originals]

The snapshot contains separate byte-for-byte copies, not hard links. SHA256 checks
fail closed. Read-only permissions protect against accidental writes; they are not
an operating-system security boundary. Raw corpora remain external dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REFERENCE_NAME = "V8 Reference: WavLM Last-4 Trainable Layer Mix + Speaker Attention"
SEEDS = [42, 43, 44, 45, 46]


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _read(path: Path) -> Any:
    return json.loads(path.read_text())


def _write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _split_audit(run: Path, seeds: list[int], outer_folds: int) -> tuple[list[dict], list[dict]]:
    membership, thresholds = [], []
    seen = set()
    for path in sorted(run.glob("folds/speech_only/seed*_outer*/fold_audit.json")):
        audit = _read(path)
        _require(audit.get("complete") is True, f"Incomplete fold: {path}")
        seed, outer = int(audit["seed"]), int(audit["outer_fold"])
        seen.add((seed, outer))
        train, test = set(audit["train_physical_speakers"]), set(audit["test_physical_speakers"])
        _require(train.isdisjoint(test), f"Outer physical-speaker leakage: {path}")
        thresholds.append(dict(seed=seed, outer_fold=outer, scope="held_out_outer_fold",
                               threshold=audit["selected_threshold"], source=audit["threshold_source"],
                               manifest=str(path.relative_to(run)), manifest_sha256=sha256_file(path)))
        for role, speakers in [("outer_train", train), ("outer_test", test)]:
            membership.extend(dict(seed=seed, outer_fold=outer, inner_fold="", role=role,
                                   physical_speaker_id=s, source_manifest=str(path.relative_to(run)))
                              for s in sorted(speakers))
        inner_validation = []
        for inner in audit["inner_folds"]:
            itrain, val = set(inner["train_physical_speakers"]), set(inner["validation_physical_speakers"])
            _require(itrain.isdisjoint(val) and itrain.isdisjoint(test) and val.isdisjoint(test),
                     f"Inner physical-speaker leakage: {path}")
            _require(itrain | val == train, f"Inner split does not cover outer training speakers: {path}")
            inner_validation.extend(val)
            for role, speakers in [("inner_train", itrain), ("inner_validation", val), ("outer_test", test)]:
                membership.extend(dict(seed=seed, outer_fold=outer, inner_fold=inner["inner_fold"],
                                       role=role, physical_speaker_id=s,
                                       source_manifest=str(path.relative_to(run))) for s in sorted(speakers))
        _require(len(inner_validation) == len(set(inner_validation)) == len(train),
                 f"Inner validation is not an exact partition: {path}")
    _require(seen == {(s, f) for s in seeds for f in range(outer_folds)},
             "Missing or unexpected reference outer folds")
    return membership, thresholds


def freeze_reference(project_root: str | Path | None = None, *, notebook: str | Path | None = None,
                     destination: str | Path | None = None) -> dict[str, Any]:
    """Freeze every saved run artifact, pretrained weights, source and exact thresholds.

    Existing snapshots are verified and returned; never overwritten. This function
    performs no training, feature extraction, calibration, or model selection.
    """
    root = Path(project_root or Path(__file__).resolve().parents[2]).resolve()
    dest = Path(destination or root / "v8/v8_publication_completion/frozen_reference").resolve()
    if dest.exists():
        return verify_reference(dest)
    pointer = root / "v8/outputs/v8_latest_run.json"
    run = Path(_read(pointer)["output_dir"]).resolve()
    manifest, summary = _read(run / "run_manifest.json"), _read(run / "v8_summary.json")
    lock = _read(run / "deployment/v8_torgo_model_lock.json")
    external_audit = _read(run / "external_uaspeech/v8_uaspeech_zero_shot_audit.json")
    external_summary = _read(run / "external_uaspeech/v8_uaspeech_summary.json")
    if notebook is None:
        candidates = [root / "v8/dysarthria_v8(1).ipynb", root / "dysarthria_v8(1).ipynb",
                      root / "v8/dysarthria_v8.ipynb"]
        notebook_path = next((p for p in candidates if p.is_file()), None)
        _require(notebook_path is not None, "Current V8 notebook not found")
    else:
        notebook_path = Path(notebook).resolve()
    weights = Path(manifest["pretrained_weight_path"]).resolve()
    _require(manifest["config"]["seeds"] == SEEDS, "Reference must contain seeds 42–46")
    _require(manifest["config"]["selected_layers"] == 4 and not manifest["config"]["use_fusion"],
             "Reference configuration differs from requested V8 speech-only last-four mix")
    for key in ["run_signature", "pipeline_signature", "pretrained_weight_sha256"]:
        _require(lock[key] == manifest[key], f"Reference lock has mismatched {key}")
    _require(lock["torgo_source_signature"] == manifest["raw_source_signature"], "Source signature mismatch")
    _require(external_audit["torgo_run_signature"] == manifest["run_signature"], "External run mismatch")
    _require(external_audit["torgo_model_lock_sha256"] == sha256_file(run / "deployment/v8_torgo_model_lock.json"),
             "External model-lock checksum mismatch")
    for name in ["training", "fine_tuning", "threshold_selection", "calibration", "model_selection"]:
        key = f"external_dataset_used_for_{name}"
        _require(external_audit.get(key) is False, f"External audit does not prove {key}=false")
    _require(external_summary.get("complete") is True and external_summary["primary_channel"] == "M3",
             "Completed primary M3 external result required before freeze")
    _require(sha256_file(weights) == manifest["pretrained_weight_sha256"], "Pretrained WavLM SHA256 mismatch")
    models = [m for m in lock["models"] if m["model_arm"] == "speech_only"]
    _require(sorted(m["seed"] for m in models) == SEEDS, "Five deployment checkpoints required")
    for model in models:
        for key in ["checkpoint", "acoustic_scaler"]:
            _require(sha256_file(run / model[key]) == model[key + "_sha256"],
                     f"Locked {key} SHA256 mismatch for seed {model['seed']}")
        audit_key = f"speech_only_seed{model['seed']}"
        _require(external_audit["frozen_checkpoint_sha256"][audit_key] == model["checkpoint_sha256"],
                 f"External checkpoint audit mismatch: {audit_key}")
    memberships, thresholds = _split_audit(run, SEEDS, manifest["config"]["outer_folds"])
    for model in models:
        thresholds.append(dict(seed=model["seed"], outer_fold="", scope="locked_torgo_deployment",
                               threshold=model["threshold"], source=model["threshold_source"],
                               manifest="deployment/v8_torgo_model_lock.json",
                               manifest_sha256=sha256_file(run / "deployment/v8_torgo_model_lock.json")))
    stage = dest.with_name(dest.name + ".incomplete")
    _require(not stage.exists(), f"Incomplete snapshot already exists: {stage}; inspect before retrying")
    stage.mkdir(parents=True)
    records = []
    selected = [(p, Path("run") / p.relative_to(run), "run_artifact") for p in sorted(run.rglob("*")) if p.is_file()]
    selected += [(weights, Path("pretrained") / weights.name, "pretrained_weights"),
                 (notebook_path, Path("source") / notebook_path.name, "notebook_source"),
                 (pointer, Path("source") / pointer.name, "original_run_pointer")]
    external_pointer = root / "v8/outputs/v8_latest_external_run.json"
    if external_pointer.is_file():
        selected.append((external_pointer, Path("source") / external_pointer.name, "original_run_pointer"))
    for index, (source, relative, role) in enumerate(selected, 1):
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        _require(not source.is_symlink(), f"Refusing symlink in successful run: {source}")
        original_hash = sha256_file(source)
        shutil.copy2(source, target)
        _require(source.stat().st_ino != target.stat().st_ino, f"Snapshot unexpectedly shares inode: {source}")
        _require(sha256_file(target) == original_hash == sha256_file(source), f"File changed during freeze: {source}")
        records.append(dict(path=relative.as_posix(), source=str(source), role=role,
                            size_bytes=target.stat().st_size, sha256=original_hash))
        if index % 5000 == 0:
            print(f"Frozen and verified {index}/{len(selected)} files", flush=True)
    _csv(stage / "locked_thresholds.csv", thresholds,
         ["seed", "outer_fold", "scope", "threshold", "source", "manifest", "manifest_sha256"])
    _csv(stage / "split_membership.csv", memberships,
         ["seed", "outer_fold", "inner_fold", "role", "physical_speaker_id", "source_manifest"])
    checkpoint_rows = []
    for entry in records:
        if entry["path"].endswith("/best.pt"):
            checkpoint_rows.append({key: entry[key] for key in ["path", "sha256", "size_bytes"]})
    _csv(stage / "checkpoint_hashes.csv", checkpoint_rows, ["path", "sha256", "size_bytes"])
    caveats = [
        "Raw TORGO and UASpeech audio corpora remain external dependencies; saved inventories preserve absolute source paths.",
        "Original source signatures use path, file size and modification time rather than waveform-content hashes.",
        "Notebook source is captured as present at freeze time; the historical run did not record a full notebook-file SHA256.",
        "A byte-identical snapshot and cached representations preserve the completed result; bitwise retraining depends on the original runtime and hardware, which were not fully locked.",
        "Read-only permissions and SHA256 detect accidental modification; a privileged user can change permissions.",
        "Official UASpeech was studied in earlier versions; use locked zero-shot cross-corpus evaluation wording, not historically untouched prospective validation.",
    ]
    if notebook_path.name != "dysarthria_v8(1).ipynb":
        caveats.append(f"Requested dysarthria_v8(1).ipynb was not present; snapshotted the available current V8 notebook: {notebook_path.name}.")
    metadata = dict(schema_version=1, created_utc=datetime.now(timezone.utc).isoformat(),
                    reference_name=REFERENCE_NAME, original_run_dir=str(run), original_notebook=str(notebook_path),
                    run_signature=manifest["run_signature"], pipeline_signature=manifest["pipeline_signature"],
                    torgo_source_signature=manifest["raw_source_signature"],
                    pretrained_weight_sha256=manifest["pretrained_weight_sha256"],
                    seed_checkpoints=models, seeds=SEEDS, config=manifest["config"],
                    feature_spec=manifest["feature_spec"],
                    torgo_summary=summary, uaspeech_summary=external_summary,
                    physical_speaker_leakage_checks="PASS", outer_folds_verified=len(SEEDS)*manifest["config"]["outer_folds"],
                    limitations=caveats, files=records)
    (stage / "README.md").write_text(
        "# Frozen V8 publication reference\n\n" + REFERENCE_NAME + "\n\n"
        "The `run/` tree is a byte-for-byte copy of the completed successful V8 run, including all inner/outer/deployment checkpoints, predictions, metrics, cached representations, histories, figures, and manifests. "
        "`source/` preserves the notebook before publication changes. `pretrained/` preserves the frozen WavLM weights. "
        "No training or evaluation was executed to create this snapshot.\n\n"
        "Verify with `python -m v8.publication.freeze verify --compare-originals` from the project root. "
        "Original notebook changes are intentionally allowed after this snapshot; original result/model changes fail verification.\n\n"
        "## Integrity and reproducibility limits\n\n" + "\n".join("- " + item for item in caveats) + "\n")
    generated_files = ["locked_thresholds.csv", "split_membership.csv", "checkpoint_hashes.csv", "README.md"]
    metadata["generated_artifacts"] = [dict(path=p, sha256=sha256_file(stage / p)) for p in generated_files]
    _write(stage / "snapshot_manifest.json", metadata)
    (stage / "snapshot_manifest.sha256").write_text(sha256_file(stage / "snapshot_manifest.json") + "  snapshot_manifest.json\n")
    for p in stage.rglob("*"):
        if p.is_file():
            p.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    for p in sorted((p for p in stage.rglob("*") if p.is_dir()), key=lambda x: len(x.parts), reverse=True):
        p.chmod(0o555)
    stage.chmod(0o555)
    stage.rename(dest)
    return verify_reference(dest, compare_originals=True)


def verify_reference(snapshot: str | Path | None = None, *, compare_originals: bool = False) -> dict[str, Any]:
    """Hard-fail on tampering, missing files, or modified original results/models.

    Notebook additions are expected after freezing and do not invalidate the saved
    original source. `compare_originals` checks every historical run artifact and
    pretrained weight, including representation caches.
    """
    directory = Path(snapshot or Path(__file__).resolve().parents[1] / "v8_publication_completion/frozen_reference").resolve()
    manifest_path = directory / "snapshot_manifest.json"
    expected = (directory / "snapshot_manifest.sha256").read_text().split()[0]
    _require(sha256_file(manifest_path) == expected, "Snapshot manifest SHA256 mismatch")
    manifest = _read(manifest_path)
    checked, total = 0, 0
    for record in manifest["files"] + manifest.get("generated_artifacts", []):
        path = directory / record["path"]
        _require(path.is_file() and sha256_file(path) == record["sha256"], f"Frozen reference corrupted: {path}")
        _require(path.stat().st_mode & 0o222 == 0, f"Frozen reference became writable: {path}")
        checked += 1
        total += path.stat().st_size
        if compare_originals and record.get("role") in {"run_artifact", "pretrained_weights"}:
            source = Path(record["source"])
            _require(source.is_file() and sha256_file(source) == record["sha256"], f"Original reference modified: {source}")
    return dict(status="PASS", snapshot=str(directory), run_signature=manifest["run_signature"],
                snapshot_manifest_sha256=expected, n_files_verified=checked, bytes_verified=total,
                originals_compared=compare_originals, limitations=manifest["limitations"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["freeze", "verify"])
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--compare-originals", action="store_true")
    args = parser.parse_args()
    if args.action == "freeze":
        result = freeze_reference(args.project_root, notebook=args.notebook, destination=args.snapshot)
    else:
        result = verify_reference(args.snapshot or args.project_root / "v8/v8_publication_completion/frozen_reference",
                                  compare_originals=args.compare_originals)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
