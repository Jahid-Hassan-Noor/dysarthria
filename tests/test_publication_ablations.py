"""Scientific controls: arm isolation, paired estimands, and saved split parity."""
from types import SimpleNamespace
import unittest
import numpy as np
import pandas as pd
import torch

from v8.publication.ablations import (
    ARMS, METRICS, _metric_values, make_ablation_head, paired_statistics,
    component_override,
)
from v8.publication.common import load_pipeline


class AblationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipeline = load_pipeline()

    def test_reference_forward_and_parameter_initialization_are_exact(self):
        base = self.pipeline.SpeakerAttentionClassifier
        torch.manual_seed(42)
        original = base(4, 24, 7, False).eval()
        speech = torch.randn(9, 4, 24)
        mask = torch.tensor([True] * 7 + [False] * 2)
        expected, weights = original(speech, valid_mask=mask, return_attention=True)
        for arm in ARMS:
            with self.subTest(arm=arm):
                torch.manual_seed(42)
                ablation = make_ablation_head(base, arm)(4, 24, 7, False).eval()
                for name, value in original.state_dict().items():
                    torch.testing.assert_close(value, ablation.state_dict()[name], rtol=0, atol=0)
                self.assertEqual(ablation.classifier[0].in_features, 256)
                actual, attention = ablation(speech, valid_mask=mask, return_attention=True)
                if arm == "B2":
                    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    torch.testing.assert_close(attention, weights, rtol=0, atol=0)
                if arm in {"A1", "B1"}:
                    torch.testing.assert_close(attention, mask.float() / mask.sum())
                    self.assertFalse(any(parameter.requires_grad for parameter in ablation.attention.parameters()))
                if arm in {"A1", "A2"}:
                    changed = speech.clone()
                    changed[:, :-1, :] += 100
                    torch.testing.assert_close(ablation(changed, valid_mask=mask), actual, rtol=0, atol=0)
                    self.assertFalse(ablation.layer_logits.requires_grad)

    def test_override_restores_original_after_failure(self):
        original = self.pipeline.SpeakerAttentionClassifier
        with self.assertRaises(RuntimeError):
            with component_override(self.pipeline, "A1"):
                self.assertIsNot(self.pipeline.SpeakerAttentionClassifier, original)
                raise RuntimeError("interrupted training")
        self.assertIs(self.pipeline.SpeakerAttentionClassifier, original)

    def test_vectorized_metrics_match_notebook_with_ties_and_boundary_bins(self):
        rng = np.random.default_rng(98)
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        probabilities = np.concatenate([rng.uniform(0, 1, (20, 8)),
            np.array([[0, .1, .2, .5, .5, .8, .9, 1], [1] * 8, [0] * 8])])
        thresholds = np.broadcast_to(np.linspace(.1, .8, 8), probabilities.shape)
        actual = _metric_values(y, probabilities, thresholds)
        for index, probability in enumerate(probabilities):
            frame = pd.DataFrame(dict(label=y, probability=probability, threshold=thresholds[index]))
            expected = self.pipeline.speaker_metrics(frame)
            np.testing.assert_allclose(actual[index], [expected[name] for name in METRICS], atol=1e-14)

    @staticmethod
    def paired_frame():
        rows = []
        for arm in ARMS:
            for seed in (42, 43):
                for speaker, label, probability in [("c1", 0, .2), ("c2", 0, .8),
                                                     ("d1", 1, .4), ("d2", 1, .9)]:
                    rows.append(dict(model_arm=arm, seed=seed, physical_speaker_id=speaker,
                                     label=label, probability=probability, threshold=.5))
        return pd.DataFrame(rows)

    def test_paired_identical_models_have_zero_difference_and_interaction(self):
        result = paired_statistics(self.paired_frame(), resamples=40)
        for key in ("paired_differences", "factorial_effects"):
            np.testing.assert_array_equal(result[key][["delta", "lower_95", "upper_95"]], 0)
        draws = result["bootstrap_draws"]
        self.assertTrue(draws[["draw_0", "draw_1"]].map(lambda value: value.startswith("c")).to_numpy().all())
        self.assertTrue(draws[["draw_2", "draw_3"]].map(lambda value: value.startswith("d")).to_numpy().all())

    def test_missing_or_duplicate_pair_fails(self):
        frame = self.paired_frame()
        with self.assertRaises(AssertionError):
            paired_statistics(frame.iloc[:-1], resamples=10)
        with self.assertRaises(AssertionError):
            paired_statistics(pd.concat((frame, frame.iloc[:1])), resamples=10)

    def test_saved_draws_reproduce_bootstrap_values_from_notebook_metrics(self):
        frame = self.paired_frame()
        # Different arms and seeds ensure the paired arithmetic is exercised.
        for ai, arm in enumerate(ARMS):
            select = frame.model_arm.eq(arm)
            frame.loc[select, "probability"] = np.clip(
                frame.loc[select, "probability"] + ai * .031 + (frame.loc[select, "seed"] - 42) * .047,
                0, 1)
        result = paired_statistics(frame, resamples=10)
        for row in result["bootstrap_draws"].itertuples(index=False, name=None):
            repetition, *speakers = row
            for arm in ARMS:
                metric_rows = []
                for seed in (42, 43):
                    selected = frame[frame.model_arm.eq(arm) & frame.seed.eq(seed)]
                    sampled = selected.set_index("physical_speaker_id").loc[speakers].reset_index()
                    metric_rows.append(self.pipeline.speaker_metrics(sampled))
                expected = pd.DataFrame(metric_rows)[list(METRICS)].mean()
                distributions = result["bootstrap_distributions"]
                actual = distributions[distributions.model_arm.eq(arm)
                    & distributions.repetition.eq(repetition)].set_index("metric").value.loc[list(METRICS)]
                np.testing.assert_allclose(actual, expected, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
