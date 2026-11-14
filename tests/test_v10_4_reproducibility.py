"""Synthetic checks for the public V10.4 preflight and speaker-level statistics."""
from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
V104 = ROOT / "v10/v10.4"
sys.path.insert(0, str(V104))
from common import mean_seed_auc, paired_auc_bootstrap  # noqa: E402
from reanalyses import youden_threshold  # noqa: E402

spec = importlib.util.spec_from_file_location("v10_4_preflight", V104 / "preflight.py")
preflight = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def test_repository_root_and_configuration(self):
        self.assertEqual(preflight.ROOT, ROOT)
        plan = json.loads((V104 / "analysis_plan.json").read_text())
        self.assertEqual(tuple(plan["seeds"]), preflight.SEEDS)

    def test_manifest_hash_size_and_missing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "input.txt").write_text("frozen")
            data = {"version": "v10.4-1", "n_inputs": 1, "inputs": [{
                "path": "input.txt", "bytes": 6, "sha256": preflight.digest(root / "input.txt"),
            }]}
            self.assertEqual(preflight.validate_manifest(data, root), [])
            (root / "input.txt").write_text("edited")
            with self.assertRaisesRegex(ValueError, "hash changed"):
                preflight.validate_manifest(data, root)
            (root / "input.txt").unlink()
            self.assertEqual(preflight.validate_manifest(data, root), ["input.txt"])

    def test_manifest_refuses_escape_and_duplicate(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            item = {"path": "../private.wav", "bytes": 1, "sha256": "0" * 64}
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                preflight.validate_manifest({"version": "v10.4-1", "n_inputs": 1, "inputs": [item]}, root)
            item["path"] = "same.txt"
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                preflight.validate_manifest({"version": "v10.4-1", "n_inputs": 2, "inputs": [item, item]}, root)

    @staticmethod
    def make_folds(root: Path) -> None:
        speakers = [f"S{i:02d}" for i in range(15)]
        for outer in range(5):
            test = speakers[outer * 3:(outer + 1) * 3]
            train = [speaker for speaker in speakers if speaker not in test]
            inner = [{"train_physical_speakers": [s for s in train if s not in train[i * 4:(i + 1) * 4]],
                      "validation_physical_speakers": train[i * 4:(i + 1) * 4]} for i in range(3)]
            path = root / f"v10/outputs/reference_training/folds/speech_only/seed42_outer{outer}/fold_audit.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"complete": True, "seed": 42, "outer_fold": outer,
                                        "train_physical_speakers": train,
                                        "test_physical_speakers": test, "inner_folds": inner}))

    def test_physical_speaker_disjoint_across_outer_and_inner_folds(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.make_folds(root)
            self.assertEqual(preflight.validate_folds(root, (42,)), (5, {f"S{i:02d}" for i in range(15)}))
            path = root / "v10/outputs/reference_training/folds/speech_only/seed42_outer0/fold_audit.json"
            data = json.loads(path.read_text())
            data["train_physical_speakers"][0] = data["test_physical_speakers"][0]
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "Speaker overlap"):
                preflight.validate_folds(root, (42,))

    def test_microphone_alias_cannot_switch_physical_speaker(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            meta = root / "metadata.csv"
            subset = root / "subset.csv"
            rows = []
            for speaker in range(15):
                for recording in range(50):
                    rows.append({"source_id": f"{speaker}-{recording}",
                                 "speaker_id": f"alias{speaker}",
                                 "physical_speaker_id": f"S{speaker:02d}",
                                 "label": str(speaker % 2), "mic_type": "array" if recording % 2 else "headset",
                                 "relative_path": f"S{speaker:02d}/{recording}.wav", "file_path": str(root / "absent.wav")})
            with meta.open("w", newline="") as stream:
                writer = __import__("csv").DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            selected = [{**row, "frozen_index": str(i)} for i, row in enumerate(rows)]
            with subset.open("w", newline="") as stream:
                writer = __import__("csv").DictWriter(stream, fieldnames=selected[0]); writer.writeheader(); writer.writerows(selected)
            self.assertEqual(preflight.validate_sources(meta, subset, None)[1:], (750, [r["relative_path"] for r in rows]))
            rows[1]["physical_speaker_id"] = "S14"
            with meta.open("w", newline="") as stream:
                writer = __import__("csv").DictWriter(stream, fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "alias maps to multiple physical speakers"):
                preflight.validate_sources(meta, subset, None)

    def test_result_summary_schema(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for stage, fields in preflight.SUMMARY_FIELDS.items():
                path = root / stage / "summary.json"
                path.parent.mkdir()
                path.write_text(json.dumps({name: 0.5 for name in fields}))
            self.assertEqual(preflight.validate_summary_schema(root), 5)
            (root / "threshold/summary.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "Missing result fields"):
                preflight.validate_summary_schema(root)


class SpeakerStatisticTests(unittest.TestCase):
    @staticmethod
    def scores() -> pd.DataFrame:
        return pd.DataFrame([{"seed": seed, "physical_speaker_id": f"S{i:02d}",
                              "label": int(i >= 7), "probability": (i + 1) / 17}
                             for seed in (42, 43, 44, 45, 46) for i in range(15)])

    def test_seed_level_auc_uses_speakers(self):
        self.assertEqual(mean_seed_auc(self.scores()), 1.0)

    def test_youden_threshold_uses_twelve_inner_speakers(self):
        frame = self.scores().query("seed == 42").iloc[:12].copy()
        threshold, score = youden_threshold(frame)
        self.assertEqual(score, 1.0)
        self.assertEqual(int((frame.probability >= threshold).sum()), 5)
        with self.assertRaisesRegex(AssertionError, "12 inner-OOF"):
            youden_threshold(frame.iloc[:11])

    def test_paired_bootstrap_is_deterministic_and_speaker_clustered(self):
        frame = self.scores()
        first = paired_auc_bootstrap(frame, frame, draws=20, seed=123)
        second = paired_auc_bootstrap(frame, frame, draws=20, seed=123)
        self.assertEqual(first, second)
        self.assertEqual(first["point_delta_auc"], 0.0)
        self.assertEqual(first["resampling_unit"], "physical_speaker")

    def test_duplicate_speaker_probability_is_rejected(self):
        frame = self.scores()
        duplicate = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
        with self.assertRaises(Exception):
            paired_auc_bootstrap(duplicate, frame, draws=2)


if __name__ == "__main__":
    unittest.main()
