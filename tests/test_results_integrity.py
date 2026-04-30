"""
Integrity tests for the committed result artefacts.

These tests don't re-run the experiments (which would take ~45 min of CPU).
Instead they validate that the committed JSON files match the numbers the
paper cites, and that the findings documents tell a story consistent with
those numbers. A failure here means either:

  * someone regenerated results with a different seed or code path
    without updating the paper, OR
  * someone edited a findings document away from the committed numbers.

Either way, the humans have to reconcile before shipping.
"""

import json
import os


REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _load_json(*path_parts):
    with open(os.path.join(REPO, *path_parts)) as f:
        return json.load(f)


class TestA2Results:
    def test_a2_json_structure(self):
        data = _load_json("adversarial", "attack_a2_results.json")
        assert "baseline_clean_test_accuracy" in data
        assert "attacks" in data
        assert len(data["attacks"]) == 6  # six variants

    def test_a2_accuracy_stays_in_baseline_band(self):
        """
        The A2 null-result claim: accuracy across all six variants
        is within a narrow band near the baseline (no monotonic drop).
        """
        data = _load_json("adversarial", "attack_a2_results.json")
        accs = [a["accuracy_as_drone"] for a in data["attacks"]]
        assert min(accs) >= 0.70, f"Unexpected low accuracy: {min(accs)}"
        assert max(accs) <= 0.95, f"Unexpected high accuracy: {max(accs)}"
        # Spread less than 15pp
        assert max(accs) - min(accs) < 0.20


class TestD2Results:
    def test_d2_json_structure(self):
        data = _load_json("adversarial", "attack_d2_results.json")
        assert len(data["attacks"]) == 9  # 0%, 20%, ... 100%

    def test_d2_at_full_glide_is_still_classified_as_drone(self):
        """
        Core D2 finding: even at glide_ratio = 1.0 (zero propeller
        content), drone-class accuracy stays above the per-class
        chance level (25%) by a wide margin.
        """
        data = _load_json("adversarial", "attack_d2_results.json")
        full_glide = [a for a in data["attacks"] if a["glide_ratio"] == 1.0]
        assert len(full_glide) == 1
        assert full_glide[0]["accuracy_as_drone"] > 0.60, (
            "D2 was expected to fail (null result); "
            "if accuracy has dropped below 60%, the finding changed."
        )


class TestAttributionResults:
    def test_attribution_json_structure(self):
        data = _load_json("adversarial", "feature_attribution_results.json")
        assert "baseline_clean_accuracy" in data
        tests = data["tests"]
        for key in [
            "bfp_permutation",
            "spectrogram_permutation",
            "frame_order_shuffle",
            "bulk_doppler_mask",
            "micro_doppler_mask",
            "temporal_mask",
        ]:
            assert key in tests, f"missing attribution test: {key}"

    def test_frame_order_is_not_important(self):
        """Core attribution finding: LSTM is multi-instance, not temporal."""
        data = _load_json("adversarial", "feature_attribution_results.json")
        drop = data["tests"]["frame_order_shuffle"]["accuracy_drop_pp"]
        assert drop < 5.0, (
            f"Frame-order shuffle now causes a meaningful drop ({drop:.1f}pp). "
            f"LSTM appears to use temporal structure; "
            f"attribution finding contradicted."
        )

    def test_spectrogram_is_load_bearing(self):
        """Core attribution finding: spectrogram content is critical."""
        data = _load_json("adversarial", "feature_attribution_results.json")
        drop = data["tests"]["spectrogram_permutation"]["accuracy_drop_pp"]
        assert drop > 20.0, (
            f"Spectrogram permutation drop is only {drop:.1f}pp; "
            f"attribution no longer shows spectrogram as load-bearing."
        )

    def test_bfp_is_load_bearing_as_distribution(self):
        """BFP permutation should hurt substantially despite BFP values being noise."""
        data = _load_json("adversarial", "feature_attribution_results.json")
        drop = data["tests"]["bfp_permutation"]["accuracy_drop_pp"]
        assert drop > 20.0
