"""Focused checks for raw UASpeech inventory and frozen external inference."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf
import torch

from v8.tests.test_pipeline import pipeline


class UASpeechExternalTests(unittest.TestCase):
    def test_external_data_is_not_opened_before_torgo_lock(self):
        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(pipeline, "lock_v8_torgo_deployment",
                               side_effect=RuntimeError("TORGO lock unavailable")),
                  patch.object(pipeline, "resolve_uaspeech_root") as external_root):
                with self.assertRaisesRegex(RuntimeError, "TORGO lock unavailable"):
                    pipeline.run_v8_uaspeech_external(pipeline.Config(), Path(temp))
                external_root.assert_not_called()

    def test_inventory_parses_normalized_release_and_rejects_bad_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for release in ("UASpeech_normalized_C", "UASpeech_normalized_FM"):
                (root / release / "audio" / "normalized").mkdir(parents=True)
            control = root / "UASpeech_normalized_C" / "audio" / "normalized" / "CF02"
            dys = root / "UASpeech_normalized_FM" / "audio" / "normalized" / "F02"
            control.mkdir()
            dys.mkdir()
            sf.write(control / "CF02_B1_CW1_M3.wav", np.ones(2048, np.float32) * .01, 16_000)
            sf.write(dys / "F02_B1_CW1_M3.wav", np.ones(2048, np.float32) * .01, 16_000)
            (dys / "F02_B1_CW2_M3.wav").write_bytes(b"not a WAV")
            (control / "F02_B1_CW3_M3.wav").write_bytes(b"not a WAV")
            meta, issues = pipeline.inventory_uaspeech(root, "M3", expected_full=False)
            self.assertEqual(len(meta), 2)
            self.assertEqual(set(meta.label), {0, 1})
            self.assertEqual(set(meta.physical_speaker_id), {"CF02", "F02"})
            self.assertEqual(set(meta.mic_type), {"external"})
            self.assertEqual(len(issues), 2)
            self.assertIn("speaker_label_path_disagreement", set(issues.reason))
            self.assertTrue(issues.reason.str.startswith("audio_decode_failed").any())

    def test_external_prediction_uses_only_locked_model_and_scaler(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = pipeline.Config(selected_layers=2, seeds=(42,))
            meta = pd.DataFrame({"physical_speaker_id": ["CF02", "CF02", "F02", "F02"],
                                 "label": [0, 0, 1, 1]})
            speech = np.zeros((4, 2, 6), dtype=np.float32)
            acoustic = np.zeros((4, 3), dtype=np.float32)
            model = pipeline._new_head(config, speech, acoustic, False, torch.device("cpu"))
            checkpoint = root / "best.pt"
            scaler_path = root / "scaler.npz"
            torch.save(model.state_dict(), checkpoint)
            np.savez(scaler_path, mean=np.zeros(3, np.float32), scale=np.ones(3, np.float32))
            lock = {"models": [{"model_arm": "speech_only", "seed": 42, "threshold": .5,
                                "checkpoint": checkpoint.name,
                                "checkpoint_sha256": pipeline._file_sha256(checkpoint),
                                "acoustic_scaler": scaler_path.name,
                                "acoustic_scaler_sha256": pipeline._file_sha256(scaler_path)}]}
            with patch.object(pipeline, "train_head", side_effect=AssertionError("external fit")):
                prediction = pipeline.predict_uaspeech_from_lock(
                    config, root, lock, meta, speech, acoustic, torch.device("cpu"))
            self.assertEqual(set(prediction.physical_speaker_id), {"CF02", "F02"})
            self.assertEqual(set(prediction.n_utterances), {2})
            self.assertTrue(prediction.probability.between(0, 1).all())
            self.assertTrue(prediction.predicted_label.eq(prediction.probability.ge(.5)).all())
            checkpoint.write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "Locked checkpoint changed"):
                pipeline.predict_uaspeech_from_lock(
                    config, root, lock, meta, speech, acoustic, torch.device("cpu"))

    def test_deployment_lock_verifies_all_expected_seeds_and_hashes(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "best.pt"
            scaler = root / "scaler.npz"
            checkpoint.write_bytes(b"weights")
            scaler.write_bytes(b"scaler")
            config = pipeline.Config(seeds=(42,))
            manifest = {"run_signature": "run-1", "raw_source_signature": "torgo-1",
                        "pretrained_weight_sha256": "encoder-1"}
            lock = {"source_corpus": "TORGO only",
                    "pipeline_signature": pipeline.PIPELINE_SIGNATURE,
                    "run_signature": "run-1", "torgo_source_signature": "torgo-1",
                    "pretrained_weight_sha256": "encoder-1",
                    "models": [{"model_arm": "speech_only", "seed": 42,
                                "checkpoint": checkpoint.name,
                                "checkpoint_sha256": pipeline._file_sha256(checkpoint),
                                "acoustic_scaler": scaler.name,
                                "acoustic_scaler_sha256": pipeline._file_sha256(scaler)}]}
            pipeline._verify_torgo_deployment_lock(lock, config, root, manifest)
            checkpoint.write_bytes(b"changed")
            with self.assertRaisesRegex(RuntimeError, "artifact changed"):
                pipeline._verify_torgo_deployment_lock(lock, config, root, manifest)


if __name__ == "__main__":
    unittest.main()
