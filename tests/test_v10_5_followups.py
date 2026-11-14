"""Focused synthetic checks for the exploratory V10.5 confound follow-ups."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "v10/v10.5"))

from background_intervention import make_conditions, quiet_mask  # noqa: E402
from prompt_match import align_prompts, select_common_prompts, duration_oof  # noqa: E402
from support import paired_bootstrap, paired_effect_bootstrap, verify_outer_checkpoints, checked_folds, load_backbone, SEEDS  # noqa: E402


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.meta = pd.DataFrame([
            {"source_id": f"{speaker}-{prompt}-{mic}",
             "relative_path": f"F_Con/wav_{mic}_{speaker}S01/wav_{mic}_{speaker}S01_{prompt}.wav",
             "physical_speaker_id": speaker, "label": int(speaker == "D1"),
             "mic_type": mic}
            for speaker in ("C1", "D1") for prompt in ("one", "two")
            for mic in ("head", "array")])

    def test_match_requires_verified_annotations_and_deduplicates_mics(self):
        with self.assertRaisesRegex(ValueError, "prompt_id or transcript"):
            align_prompts(self.meta, self.meta[["source_id"]])
        mapping = self.meta[["source_id"]].copy()
        mapping["transcript"] = mapping.source_id.str.split("-").str[1]
        selected = select_common_prompts(align_prompts(self.meta, mapping), {"C1", "D1"})
        self.assertEqual(len(selected), 4)
        self.assertEqual(set(selected.mic_type), {"head"})
        self.assertEqual(set(selected.prompt_key), {"one", "two"})

    def test_conflicting_prompt_id_is_rejected(self):
        mapping = self.meta[["source_id"]].copy()
        mapping["prompt_id"] = "P1"
        mapping["transcript"] = mapping.source_id.str.split("-").str[1]
        with self.assertRaisesRegex(ValueError, "conflicting transcripts"):
            align_prompts(self.meta, mapping)

    def test_no_partial_prompt_distribution(self):
        mapping = self.meta[["source_id"]].copy()
        mapping["transcript"] = mapping.source_id.str.split("-").str[1]
        mapping = mapping[~mapping.source_id.str.startswith("D1-two")]
        selected = select_common_prompts(align_prompts(self.meta, mapping), {"C1", "D1"})
        self.assertEqual(set(selected.prompt_key), {"one"})


class BackgroundTests(unittest.TestCase):
    def test_foreground_duration_clipping_and_sham_identity(self):
        sr = 16000
        t = np.arange(sr) / sr
        audio = (np.sin(2 * np.pi * 300 * t) * .5).astype(np.float32)
        audio[:3200] = np.sin(2 * np.pi * 50 * t[:3200]) * .002
        donor = (np.sin(2 * np.pi * 80 * t[:4000]) * .003).astype(np.float32)
        result, diagnostics = make_conditions(audio, donor, sr)
        np.testing.assert_array_equal(result["sham"], audio)
        self.assertTrue(np.array_equal(result["swapped"][3200:], audio[3200:]))
        self.assertFalse(np.array_equal(result["swapped"], audio))
        mask = quiet_mask(audio, sr)
        transitions = np.diff(np.r_[False, mask, False].astype(int))
        starts, ends = np.flatnonzero(transitions == 1), np.flatnonzero(transitions == -1)
        for condition in result.values():
            np.testing.assert_array_equal(condition[starts], audio[starts])
            np.testing.assert_array_equal(condition[ends - 1], audio[ends - 1])
        self.assertTrue(all(x["duration_seconds"] == 1.0 for x in diagnostics))
        self.assertTrue(all(x["clipped_samples"] == 0 for x in diagnostics))
        self.assertTrue(all(x["foreground_max_abs_difference"] == 0 for x in diagnostics))

    def test_rejects_no_quiet_foreground(self):
        with self.assertRaisesRegex(ValueError, "Insufficient quiet"):
            make_conditions(np.ones(16000, dtype=np.float32) * .5,
                            np.ones(1600, dtype=np.float32) * .01, 16000)


class StatisticalTests(unittest.TestCase):
    def test_paired_speaker_bootstrap_uses_same_people_across_seeds(self):
        data = pd.DataFrame([{"seed": seed, "physical_speaker_id": f"S{i}",
                              "label": int(i >= 7), "probability": .1 + .8 * int(i >= 7),
                              "threshold": .5} for seed in SEEDS for i in range(15)])
        result = paired_bootstrap(data, data, draws=20, seed=42)
        self.assertTrue((result.delta_first_minus_second == 0).all())
        self.assertTrue((result.lower_95 == 0).all())
        self.assertEqual(set(result.unit), {"physical_speaker"})
        with self.assertRaisesRegex(ValueError, "coverage"):
            paired_bootstrap(data.iloc[:-1], data, draws=2)

    @unittest.skipUnless((ROOT / "v10/outputs/reference_training/folds/speech_only/seed42_outer0/final/best.pt").is_file(),
                         "Frozen local checkpoints are not distributed with the public clone")
    def test_fold_adapter_requires_saved_disjoint_speakers(self):
        folds = checked_folds()
        self.assertEqual(len(folds), 25)
        self.assertEqual(len(verify_outer_checkpoints()), 25)
        speakers = sorted({s for _, _, _, audit in folds for s in audit["test_physical_speakers"]})
        labels = {s: int(s.startswith(("F0", "M0"))) for s in speakers}
        table = pd.DataFrame([{"physical_speaker_id": s, "label": labels[s],
                               "n_utterances": 5, "duration_mean": i + 1.,
                               "duration_median": i + 1., "duration_sd": .1 + i / 10,
                               "short_fraction": .0} for i, s in enumerate(speakers)])
        backbone, _ = load_backbone()
        outer, inner = duration_oof(table, backbone)
        self.assertEqual(len(outer), 75)
        self.assertEqual(len(inner), 300)
        self.assertFalse(outer.duplicated(["seed", "physical_speaker_id"]).any())
        for seed, fold, _, audit in folds:
            part = outer[(outer.seed == seed) & (outer.outer_fold == fold)]
            self.assertEqual(set(part.physical_speaker_id), set(audit["test_physical_speakers"]))

    def test_paired_effect_interval_is_zero_for_sham_identity(self):
        data = pd.DataFrame([{"seed": seed, "physical_speaker_id": f"S{i}",
                              "label": int(i >= 7), "delta_logit": 0.,
                              "delta_probability": 0., "decision_changed": False}
                             for seed in SEEDS for i in range(15)])
        result = paired_effect_bootstrap(data, draws=20)
        self.assertTrue((result.mean_condition_minus_sham == 0).all())
        self.assertTrue((result.upper_95 == 0).all())


if __name__ == "__main__":
    unittest.main()
