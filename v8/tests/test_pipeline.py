"""Small offline checks for V8's notebook data boundary and speaker-level model."""
from __future__ import annotations

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio


def _load_notebook_definitions() -> types.ModuleType:
    notebook_path = Path(__file__).resolve().parents[1] / "dysarthria_v8.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    cells = [cell for cell in notebook["cells"]
             if cell["cell_type"] == "code"
             and "v8-definitions" in cell.get("metadata", {}).get("tags", [])]
    if not cells:
        raise RuntimeError(f"No v8-definitions cells found in {notebook_path}")
    module = types.ModuleType("v8_notebook_definitions_for_tests")
    sys.modules[module.__name__] = module  # dataclass and inspect resolve this name
    for cell in cells:
        source = cell["source"]
        exec(compile("".join(source) if isinstance(source, list) else source,
                     str(notebook_path), "exec"), module.__dict__)
    return module


pipeline = _load_notebook_definitions()


def _wav(root: Path, group: str, folder: str, name: str, *, frames: int = 2048) -> Path:
    path = root / group / folder / name
    path.parent.mkdir(parents=True, exist_ok=True)
    time = np.arange(frames, dtype=np.float32) / 16_000
    sf.write(path, .1 * np.sin(2 * np.pi * 220 * time), 16_000)
    return path


class InventoryTests(unittest.TestCase):
    def test_raw_inventory_rejects_mismatched_paths_and_unreadable_audio(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            _wav(root, "F_Dys", "wav_headMic_F01S01", "wav_headMic_F01S01_0001.wav")
            _wav(root, "F_Con", "wav_arrayMic_FC01S01", "wav_arrayMic_FC01S01_0001.wav")
            _wav(root, "F_Con", "wav_headMic_F01S01", "wav_headMic_F01S01_0002.wav")
            _wav(root, "F_Dys", "wav_headMic_F01S01", "wrong_basename.wav")
            _wav(root, "F_Dys", "unexpected_folder", "unexpected_folder_0001.wav")
            _wav(root, "F_Con", "wav_arrayMic_FC01S01", "wav_arrayMic_FC01S01_0002.wav",
                 frames=0)
            bad = root / "F_Dys" / "wav_headMic_F01S01" / "wav_headMic_F01S01_0003.wav"
            bad.write_bytes(b"not-a-waveform")

            meta, issues = pipeline.inventory_dataset(root, expected_full=False)

            self.assertEqual(len(meta), 2)
            self.assertEqual(set(meta.physical_speaker_id), {"F01", "FC01"})
            self.assertEqual(set(meta.label), {0, 1})
            self.assertEqual(len(issues), 5)
            self.assertEqual(issues.reason.str.startswith("audio_decode_failed").sum(), 2)
            self.assertIn("speaker_label_path_disagreement", set(issues.reason))
            self.assertIn("unexpected_group_or_session_folder", set(issues.reason))
            self.assertTrue(meta.file_path.map(lambda value: Path(value).is_file()).all())


class GroupingAndScalingTests(unittest.TestCase):
    def test_all_sessions_and_microphones_follow_physical_speaker(self):
        rows = []
        for speaker, label in (("F01", 1), ("F03", 1), ("M01", 1),
                               ("FC01", 0), ("FC02", 0), ("MC01", 0)):
            for session in ("S01", "S02"):
                for mic in ("head", "array"):
                    rows.append(dict(physical_speaker_id=speaker, speaker_id=speaker + session,
                                     mic_type=mic, label=label))
        meta = pd.DataFrame(rows)
        held_out = []
        for _, train, test in pipeline.grouped_speaker_splits(meta, seed=42, folds=3):
            self.assertFalse(train & test)
            self.assertEqual(len(train | test), 6)
            held_out.extend(test)
            self.assertTrue(meta.loc[meta.physical_speaker_id.isin(test)]
                            .groupby("physical_speaker_id").size().eq(4).all())
        self.assertEqual(set(held_out), set(meta.physical_speaker_id))
        self.assertEqual(len(held_out), 6)

    def test_acoustic_scaler_excludes_held_out_speaker_and_weights_speakers_equally(self):
        meta = pd.DataFrame({"physical_speaker_id": ["A", "A", "A", "B", "H"]})
        features = np.array([[0.], [0.], [6.], [10.], [1000.]], dtype=np.float32)
        mean, scale = pipeline.fit_acoustic_scaler(meta, features, {"A", "B"})
        self.assertAlmostEqual(float(mean[0]), 6.0)
        self.assertGreater(float(scale[0]), 0)
        features[-1] = -1000.
        new_mean, new_scale = pipeline.fit_acoustic_scaler(meta, features, {"A", "B"})
        np.testing.assert_allclose(mean, new_mean)
        np.testing.assert_allclose(scale, new_scale)

    def test_nested_fold_never_trains_on_outer_test_speaker(self):
        rows = [dict(physical_speaker_id=sp, label=label)
                for sp, label in (("D1", 1), ("D2", 1), ("C1", 0), ("C2", 0),
                                  ("D3", 1), ("C3", 0))]
        meta = pd.DataFrame(rows)
        train_speakers = {"D1", "D2", "C1", "C2"}
        test_speakers = {"D3", "C3"}
        calls = []

        def fake_train(_config, _meta, _speech, _acoustic, train, validation, **_kwargs):
            self.assertFalse(set(train) & test_speakers)
            self.assertFalse(set(validation or []) & test_speakers)
            self.assertFalse(set(train) & set(validation or []))
            calls.append((set(train), set(validation or [])))
            return object(), 2, object()

        def fake_predict(_model, _store, speakers, _bag_size):
            return pd.DataFrame([dict(physical_speaker_id=sp,
                                      label=int(meta.set_index("physical_speaker_id").loc[sp, "label"]),
                                      probability=.8 if sp.startswith("D") else .2,
                                      n_utterances=1)
                                 for sp in sorted(speakers)])

        with tempfile.TemporaryDirectory() as temp:
            with (patch.object(pipeline, "train_head", side_effect=fake_train),
                  patch.object(pipeline, "predict_speakers", side_effect=fake_predict)):
                predictions = pipeline._run_outer_fold(
                    pipeline.Config(inner_folds=2), meta, np.zeros((len(meta), 1, 2), np.float32),
                    np.zeros((len(meta), 2), np.float32), seed=42, outer=0,
                    train_speakers=train_speakers, test_speakers=test_speakers,
                    use_fusion=False, run_dir=Path(temp), run_signature="test",
                    device=torch.device("cpu"))
        self.assertEqual(set(predictions.physical_speaker_id), test_speakers)
        self.assertEqual(len(calls), 3)  # two inner fits, then one final fit
        self.assertEqual(calls[-1], (train_speakers, set()))
        self.assertAlmostEqual(float(predictions.threshold.iloc[0]), .8)


class EncoderAndHeadTests(unittest.TestCase):
    def test_real_wavlm_accepts_unpadded_waveforms(self):
        # Torchaudio's WavLM attention rejects the padding mask created when
        # extract_features receives lengths; exercise the actual model API.
        model = torchaudio.models.wavlm_base().eval()
        waveforms = [np.zeros(16_000, np.float32), np.zeros(12_000, np.float32)]
        embedding = pipeline._encode_batch(
            model, waveforms, pipeline.Config(encoder_name="WAVLM_BASE", selected_layers=4),
            torch.device("cpu"))
        self.assertEqual(embedding.shape, (2, 4, 1536))
        self.assertTrue(np.isfinite(embedding).all())

    def test_fake_encoder_uses_returned_frame_lengths_for_pooling(self):
        class FakeEncoder(torch.nn.Module):
            def extract_features(self, padded, lengths):
                layer = padded[..., None].clone()
                invalid = torch.arange(layer.shape[1])[None, :] >= lengths[:, None]
                layer[invalid] = 10_000.
                return [layer], lengths

        config = pipeline.Config(selected_layers=1)
        embedding = pipeline._encode_batch(
            FakeEncoder(), [np.array([1., 3., 5.], np.float32),
                            np.array([2.], np.float32)], config, torch.device("cpu"))
        self.assertEqual(embedding.shape, (2, 1, 2))
        np.testing.assert_allclose(embedding[0, 0], [3., np.sqrt(8 / 3)], rtol=1e-6)
        np.testing.assert_allclose(embedding[1, 0], [2., 0.])

    def test_masked_encoder_statistics_ignore_padded_frames(self):
        layers = [torch.zeros(2, 3, 1),
                  torch.tensor([[[1.], [3.], [10_000.]], [[2.], [10_000.], [10_000.]]]),
                  torch.tensor([[[2.], [4.], [10_000.]], [[6.], [10_000.], [10_000.]]])]
        embedding = pipeline.masked_layer_statistics(layers, torch.tensor([2, 1]), 2)
        self.assertEqual(embedding.shape, (2, 2, 2))
        np.testing.assert_allclose(embedding[0], [[2., 1.], [3., 1.]])
        np.testing.assert_allclose(embedding[1], [[2., 0.], [6., 0.]])

    def test_speaker_attention_ignores_masked_utterance_and_outputs_probability(self):
        torch.manual_seed(3)
        model = pipeline.SpeakerAttentionClassifier(2, 4, 3, use_fusion=True).eval()
        speech = torch.randn(3, 2, 4)
        acoustic = torch.randn(3, 3)
        speech[2] = 10_000.
        acoustic[2] = 10_000.
        with torch.no_grad():
            logit, weights = model(speech, acoustic, torch.tensor([True, True, False]),
                                   return_attention=True)
            two_only = model(speech[:2], acoustic[:2])
            probability = torch.sigmoid(logit)
        self.assertEqual(tuple(weights.shape), (3,))
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
        self.assertEqual(float(weights[2]), 0.0)
        self.assertTrue(torch.allclose(logit, two_only, atol=1e-6))
        self.assertGreater(float(probability), 0.0)
        self.assertLess(float(probability), 1.0)

    def test_deterministic_chunking_covers_each_utterance_without_mixing_speakers(self):
        meta = pd.DataFrame({"physical_speaker_id": ["A"] * 5 + ["B"] * 3,
                             "label": [1] * 5 + [0] * 3})
        speech = np.arange(8, dtype=np.float32).reshape(8, 1, 1)
        acoustic = np.zeros((8, 1), dtype=np.float32)
        store = pipeline.BagStore(meta, speech, acoustic,
                                  (np.zeros(1, np.float32), np.ones(1, np.float32)),
                                  torch.device("cpu"))

        class RecordingModel:
            def __init__(self):
                self.calls = []

            def eval(self):
                return self

            def __call__(self, bag_speech, _bag_acoustic):
                ids = [int(x) for x in bag_speech[:, 0, 0].tolist()]
                assert all(x < 5 for x in ids) or all(x >= 5 for x in ids)
                self.calls.append(ids)
                return bag_speech[:, 0, 0].mean()

        model = RecordingModel()
        first = pipeline.predict_speakers(model, store, {"B", "A"}, bag_size=2)
        first_calls = list(model.calls)
        second = pipeline.predict_speakers(model, store, {"A", "B"}, bag_size=2)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(first_calls, model.calls[len(first_calls):])
        self.assertEqual(sorted(x for bag in first_calls for x in bag), list(range(8)))
        self.assertEqual(first.set_index("physical_speaker_id").n_eval_bags.to_dict(),
                         {"A": 3, "B": 2})
        self.assertAlmostEqual(float(first.iloc[0].probability),
                               float(torch.sigmoid(torch.tensor(2.)).item()), places=6)
        self.assertAlmostEqual(float(first.iloc[1].probability),
                               float(torch.sigmoid(torch.tensor(6.)).item()), places=6)
        with self.assertRaises(AssertionError):
            store.bag("A", explicit_indices=np.array([0, 5]))


class CacheAndThresholdTests(unittest.TestCase):
    def test_feature_spec_changes_with_encoder_weights_and_preprocessing(self):
        default = pipeline.Config(selected_layers=2)
        spec = pipeline._feature_spec(default, "weight-a")
        self.assertIn("v8", spec["pipeline_signature"])
        self.assertEqual(spec["pretrained_weight_sha256"], "weight-a")
        self.assertNotEqual(spec, pipeline._feature_spec(default, "weight-b"))
        self.assertNotEqual(spec, pipeline._feature_spec(
            pipeline.Config(selected_layers=2, max_seconds=6.0), "weight-a"))
        self.assertNotEqual(spec, pipeline._feature_spec(
            pipeline.Config(encoder_name="HUBERT_BASE", selected_layers=2), "weight-a"))

    def test_v8_feature_cache_reuses_matching_shards_and_invalidates_changed_sources(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw = root / "raw"
            _wav(raw, "F_Dys", "wav_headMic_F01S01", "wav_headMic_F01S01_0001.wav")
            meta, _ = pipeline.inventory_dataset(raw, expected_full=False)
            run_dir = root / "v8" / "outputs" / "cache" / "dysarthria_detection_v8" / "run_test"
            config = pipeline.Config(selected_layers=1)
            calls = []

            def fake_encode(_model, waveforms, cfg, _device):
                calls.append(len(waveforms))
                return np.full((len(waveforms), cfg.selected_layers, 2),
                               float(len(calls)), dtype=np.float32)

            with (patch.object(pipeline, "preprocess_waveform", return_value=np.zeros(1024, np.float32)),
                  patch.object(pipeline, "acoustic_features",
                               return_value=np.zeros(len(pipeline.ACOUSTIC_NAMES), np.float32)),
                  patch.object(pipeline, "_encode_batch", side_effect=fake_encode)):
                first = pipeline._feature_spec(config, "weights-a")
                _, embeddings, _, _ = pipeline.extract_dataset_features(
                    meta, object(), config, run_dir, first, torch.device("cpu"))
                self.assertEqual(len(calls), 1)
                self.assertEqual(float(embeddings[0, 0, 0]), 1.)
                pipeline.extract_dataset_features(
                    meta, object(), config, run_dir, first, torch.device("cpu"))
                self.assertEqual(len(calls), 1)

                changed_weights = pipeline._feature_spec(config, "weights-b")
                pipeline.extract_dataset_features(
                    meta, object(), config, run_dir, changed_weights, torch.device("cpu"))
                self.assertEqual(len(calls), 2)

                _wav(raw, "F_Dys", "wav_headMic_F01S01", "wav_headMic_F01S01_0001.wav",
                     frames=4096)
                changed_meta, _ = pipeline.inventory_dataset(raw, expected_full=False)
                self.assertNotEqual(pipeline.source_signature(meta),
                                    pipeline.source_signature(changed_meta))
                pipeline.extract_dataset_features(
                    changed_meta, object(), config, run_dir, changed_weights,
                    torch.device("cpu"))
                self.assertEqual(len(calls), 3)

    def test_threshold_uses_inner_speaker_scores_and_rejects_duplicate_speaker(self):
        validation = pd.DataFrame({"physical_speaker_id": ["C1", "C2", "D1", "D2"],
                                   "label": [0, 0, 1, 1],
                                   "probability": [.1, .7, .8, .9]})
        self.assertAlmostEqual(pipeline.select_threshold(validation, 1.0), .8)
        duplicate = pd.concat([validation, validation.iloc[[0]]], ignore_index=True)
        with self.assertRaises(AssertionError):
            pipeline.select_threshold(duplicate)


if __name__ == "__main__":
    unittest.main()
