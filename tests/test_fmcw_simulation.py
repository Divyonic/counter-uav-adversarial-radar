"""Tests for the FMCW radar simulator and feature extractors."""

import numpy as np

from fmcw_simulation import (
    generate_drone_signal,
    generate_bird_signal,
    generate_friendly_uav_signal,
    generate_aircraft_signal,
    compute_spectrogram,
    extract_bfp_features,
)


# ---------------------------------------------------------------------------
# Shape and finiteness
# ---------------------------------------------------------------------------

class TestSignalGeneration:
    def test_drone_signal_shape(self):
        """Drone signal returns a 2-D complex beat matrix."""
        beat = generate_drone_signal(
            R0=1000, v_bulk=12, snr_db=20, n_blades=2, n_props=4,
            rpm=5000, blade_len=0.12, tilt_angle=45,
        )
        assert beat.ndim == 2
        assert np.iscomplexobj(beat)
        assert np.all(np.isfinite(beat))

    def test_bird_signal_shape(self):
        beat = generate_bird_signal(
            R0=800, v_bulk=10, snr_db=20, flap_freq=8, wingspan=0.5,
        )
        assert beat.ndim == 2
        assert np.all(np.isfinite(beat))

    def test_fixed_wing_signal_shape(self):
        beat = generate_friendly_uav_signal(
            R0=900, v_bulk=25, snr_db=20, n_blades=2, rpm=3500, blade_len=0.2,
        )
        assert beat.ndim == 2
        assert np.all(np.isfinite(beat))

    def test_aircraft_signal_shape(self):
        beat = generate_aircraft_signal(R0=3000, v_bulk=80, snr_db=20)
        assert beat.ndim == 2
        assert np.all(np.isfinite(beat))


# ---------------------------------------------------------------------------
# Determinism with seeded NumPy RNG
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_drone_signal_is_deterministic_under_seed(self):
        """Same seed and same params should produce identical signals."""
        kw = dict(R0=1000, v_bulk=12, snr_db=20, n_blades=2, n_props=4,
                  rpm=5000, blade_len=0.12, tilt_angle=45)
        np.random.seed(7)
        a = generate_drone_signal(**kw)
        np.random.seed(7)
        b = generate_drone_signal(**kw)
        np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------

class TestSpectrogram:
    def test_spectrogram_shape(self):
        """STFT output is a real-valued 2-D array with non-negative power."""
        beat = generate_drone_signal(
            R0=1000, v_bulk=12, snr_db=20, n_blades=2, n_props=4,
            rpm=5000, blade_len=0.12, tilt_angle=45,
        )
        spec, f, t = compute_spectrogram(beat)
        assert spec.ndim == 2
        assert spec.shape[0] > 32 and spec.shape[1] > 32
        assert np.all(spec >= 0)
        assert len(f) == spec.shape[0]
        assert len(t) == spec.shape[1]


# ---------------------------------------------------------------------------
# BFP extractor (documents a known limitation)
# ---------------------------------------------------------------------------

class TestBFPExtractor:
    def test_bfp_returns_three_values(self):
        """BFP feature vector should always be 3-D."""
        beat = generate_drone_signal(
            R0=1000, v_bulk=12, snr_db=20, n_blades=2, n_props=4,
            rpm=5000, blade_len=0.12, tilt_angle=45,
        )
        spec, _, t = compute_spectrogram(beat)
        fs = len(t) / (t[-1] - t[0])
        bfp = extract_bfp_features(spec, fs)
        assert bfp.shape == (3,)
        assert np.all(np.isfinite(bfp))

    def test_bfp_is_noise_regardless_of_physics(self):
        """
        Regression guard on the A2/FINDINGS_A2 finding: the BFP extractor's
        measured frequency is approximately independent of ground-truth
        blade-flash physics. This test encodes that as a contract: if a
        future fix 'uncouples' the extractor from the noise, this test
        will fail and ask the maintainer to update the findings.
        """
        measured = []
        for n_blades, rpm in [(2, 5000), (1, 5000), (1, 800)]:
            rng_state = np.random.get_state()
            values = []
            for seed in range(12):
                np.random.seed(seed)
                beat = generate_drone_signal(
                    R0=1000, v_bulk=12, snr_db=15,
                    n_blades=n_blades, n_props=4,
                    rpm=rpm, blade_len=0.12, tilt_angle=45,
                )
                spec, _, t = compute_spectrogram(beat)
                fs = len(t) / (t[-1] - t[0])
                values.append(extract_bfp_features(spec, fs)[0])
            measured.append(float(np.mean(values)))
            np.random.set_state(rng_state)
        # All three cluster in the same band (< 30 Hz spread)
        spread = max(measured) - min(measured)
        assert spread < 30.0, (
            f"BFP extractor newly responds to ground-truth physics; "
            f"cluster spread = {spread:.1f} Hz. "
            f"Measured: {measured}. "
            f"If this is a real improvement, update FINDINGS_A2.md."
        )
