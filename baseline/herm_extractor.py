"""
HERM-Line Feature Extractor
============================
Replaces the autocorrelation-based BFP with a proper harmonic-line
analysis inspired by Helicopter Rotor Modulation (HERM) theory.

Theory: a rotating propeller with N_b blades at rotation rate Omega (Hz)
produces a temporal-envelope power signal with energy at harmonics
f_n = n * N_b * Omega for n = 1, 2, 3, ...

Current BFP extractor finds only the fundamental via autocorrelation peak,
which is noise-dominated in practice (A2 attack results showed measured
BFP of ~45 Hz regardless of physical ground truth spanning 13-167 Hz).

This module implements:
  1. Temporal-envelope spectrum (FFT of log-power envelope with windowing
     and zero-padding for finer frequency resolution)
  2. Harmonic Product Spectrum (HPS) to amplify the fundamental through
     harmonic alignment
  3. Feature vector: fundamental frequency, harmonic-to-noise ratio,
     spectral entropy, count of significant harmonics, total harmonic
     energy concentration

The module exposes a drop-in replacement for extract_bfp_features with
the same signature: extract_herm_features(spectrogram, fs_stft) -> np.ndarray
"""

import numpy as np
from scipy.signal import get_window


def _envelope_spectrum(envelope, fs, pad_to=512):
    """Compute magnitude spectrum of the temporal envelope."""
    if len(envelope) < 4:
        return None, None
    # Center and normalise
    e = envelope - np.mean(envelope)
    std = np.std(e)
    if std == 0:
        return None, None
    e = e / std
    # Hann window to reduce spectral leakage
    win = get_window('hann', len(e))
    e = e * win
    # Zero-pad to get finer frequency resolution
    pad_len = max(pad_to, len(e))
    spec = np.abs(np.fft.rfft(e, n=pad_len))
    freqs = np.fft.rfftfreq(pad_len, d=1.0 / fs)
    return freqs, spec


def _harmonic_product_spectrum(spec, n_harmonics=4):
    """
    Compute HPS by multiplying the spectrum with its downsampled versions.
    The true fundamental is amplified because its harmonics align.
    """
    hps = spec.copy()
    for h in range(2, n_harmonics + 1):
        # Downsample by factor h
        down = spec[::h]
        # Pad/truncate to match hps length
        if len(down) < len(hps):
            down = np.concatenate([down, np.zeros(len(hps) - len(down))])
        else:
            down = down[:len(hps)]
        hps = hps * down
    # Take n-th root to keep scale comparable
    hps = hps ** (1.0 / n_harmonics)
    return hps


def extract_herm_features(spectrogram, fs_stft, search_range=(5, 500),
                           n_harmonics=4):
    """
    Extract HERM-based features from a micro-Doppler spectrogram.

    Args:
        spectrogram: 2D array, power spectrogram P(freq, time) with
                     shape (n_freq, n_time).
        fs_stft: float, sample rate of the temporal envelope (Hz).
        search_range: (low_hz, high_hz), frequency band to search for
                      fundamental. Default 5-500 Hz covers bird flap
                      (5-20 Hz) through multi-rotor BFP (50-200 Hz) to
                      aircraft engine modulation (700+ Hz would need
                      wider range).
        n_harmonics: int, number of harmonics to consider in HPS.

    Returns:
        np.ndarray of shape (5,):
            [fundamental_hz, harmonic_to_noise_ratio, spectral_entropy,
             n_significant_harmonics, harmonic_energy_fraction]

        All zeros if extraction fails (e.g., envelope too short).
    """
    # Temporal envelope = power summed over all Doppler bins at each time
    envelope = np.sum(spectrogram, axis=0)

    freqs, spec = _envelope_spectrum(envelope, fs_stft, pad_to=512)
    if freqs is None:
        return np.zeros(5, dtype=np.float32)

    # Restrict to search range
    lo, hi = search_range
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 4:
        return np.zeros(5, dtype=np.float32)
    band_freqs = freqs[mask]
    band_spec = spec[mask]

    # Harmonic product spectrum within the search band
    # (operate on the full spectrum so harmonic alignment works, then
    # pick peak inside the band)
    hps_full = _harmonic_product_spectrum(spec, n_harmonics=n_harmonics)
    hps_band = hps_full[mask]

    # Find fundamental as peak of HPS in the search band
    if np.max(hps_band) == 0:
        return np.zeros(5, dtype=np.float32)
    peak_idx = int(np.argmax(hps_band))
    f_fund = float(band_freqs[peak_idx])

    if f_fund <= 0:
        return np.zeros(5, dtype=np.float32)

    # Feature 2: Harmonic-to-Noise Ratio
    # Sum power at n * f_fund for n = 1..n_harmonics, divide by total.
    total_power = float(np.sum(spec ** 2) + 1e-12)
    harmonic_power = 0.0
    n_sig_harm = 0
    noise_floor = float(np.median(spec))
    for n in range(1, n_harmonics + 1):
        target_f = n * f_fund
        if target_f >= freqs[-1]:
            break
        # Look within ±1 bin of the target harmonic
        bin_idx = int(np.argmin(np.abs(freqs - target_f)))
        lo_b = max(0, bin_idx - 1)
        hi_b = min(len(spec), bin_idx + 2)
        h_power = float(np.max(spec[lo_b:hi_b]) ** 2)
        harmonic_power += h_power
        # Count as "significant" if above 2x noise floor
        if spec[bin_idx] > 2 * noise_floor:
            n_sig_harm += 1

    hnr = float(harmonic_power / total_power)

    # Feature 3: Spectral entropy in the search band (lower = more periodic)
    p = band_spec / (np.sum(band_spec) + 1e-12)
    p = p[p > 0]
    spec_entropy = float(-np.sum(p * np.log(p))) if len(p) > 0 else 0.0

    # Feature 5: harmonic energy fraction in the search band
    harm_fraction = float(harmonic_power / (np.sum(band_spec ** 2) + 1e-12))

    return np.array([f_fund, hnr, spec_entropy, float(n_sig_harm),
                     harm_fraction], dtype=np.float32)


# ============================================================
# Diagnostic: compare HERM vs BFP on known synthetic data
# ============================================================

def compare_herm_vs_bfp(n_samples_per_class=50, snr_db=15):
    """
    Generate clean data from each class and compare HERM vs BFP feature
    distributions. If HERM discriminates classes but BFP does not, HERM
    is the better feature.
    """
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fmcw_simulation import (generate_drone_signal, generate_bird_signal,
                                   generate_friendly_uav_signal,
                                   generate_aircraft_signal,
                                   compute_spectrogram, extract_bfp_features)

    class_configs = {
        'Drone': (generate_drone_signal, lambda: dict(
            R0=np.random.uniform(500, 2000), v_bulk=np.random.uniform(5, 20),
            snr_db=snr_db, n_blades=2, n_props=4,
            rpm=np.random.uniform(4000, 6000),
            blade_len=np.random.uniform(0.10, 0.15),
            tilt_angle=np.random.uniform(30, 60))),
        'Bird': (generate_bird_signal, lambda: dict(
            R0=np.random.uniform(200, 1500), v_bulk=np.random.uniform(5, 15),
            snr_db=snr_db, flap_freq=np.random.uniform(2, 12),
            wingspan=np.random.uniform(0.2, 0.8))),
        'Fixed-wing UAV': (generate_friendly_uav_signal, lambda: dict(
            R0=np.random.uniform(300, 1500), v_bulk=np.random.uniform(15, 35),
            snr_db=snr_db, n_blades=2,
            rpm=np.random.uniform(2500, 4500),
            blade_len=np.random.uniform(0.15, 0.25))),
        'Aircraft': (generate_aircraft_signal, lambda: dict(
            R0=np.random.uniform(1000, 5000), v_bulk=np.random.uniform(50, 100),
            snr_db=snr_db)),
    }

    results = {}
    for name, (gen, p_fn) in class_configs.items():
        bfps, herms = [], []
        for _ in range(n_samples_per_class):
            params = p_fn()
            beat = gen(**params)
            spec, f, t = compute_spectrogram(beat)
            fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
            bfps.append(extract_bfp_features(spec, fs_stft))
            herms.append(extract_herm_features(spec, fs_stft))
        results[name] = {
            'bfp': np.array(bfps),       # shape (n, 3)
            'herm': np.array(herms),     # shape (n, 5)
        }
    return results


def print_comparison_report(results):
    """Pretty-print class-wise statistics for BFP vs HERM features."""
    print("=" * 80)
    print("FEATURE COMPARISON REPORT: BFP vs HERM")
    print("=" * 80)
    bfp_names = ['f_bfp', 'c_bfp', 'df_bfp']
    herm_names = ['f_fund', 'hnr', 'entropy', 'n_harm', 'harm_frac']
    classes = list(results.keys())

    print("\n--- Autocorrelation-based BFP ---")
    print(f"{'Class':<16} " + "  ".join([f"{n:>12}" for n in bfp_names]))
    for cls in classes:
        arr = results[cls]['bfp']
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        line = f"{cls:<16} "
        for m, s in zip(means, stds):
            line += f"  {m:6.2f}±{s:<4.2f}  "
        print(line)

    print("\n--- HERM harmonic-product-spectrum ---")
    print(f"{'Class':<16} " + "  ".join([f"{n:>12}" for n in herm_names]))
    for cls in classes:
        arr = results[cls]['herm']
        means = arr.mean(axis=0)
        stds = arr.std(axis=0)
        line = f"{cls:<16} "
        for m, s in zip(means, stds):
            line += f"  {m:6.2f}±{s:<4.2f}  "
        print(line)

    print("\n--- Class separability (Fisher ratio: between/within variance) ---")
    print("Higher = more discriminative feature")
    for feat_idx, fname in enumerate(bfp_names):
        vals = [results[c]['bfp'][:, feat_idx] for c in classes]
        grand_mean = np.mean(np.concatenate(vals))
        between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in vals)
        within = sum(np.sum((v - np.mean(v)) ** 2) for v in vals)
        fisher = between / (within + 1e-12)
        print(f"  BFP  {fname:<10}: Fisher = {fisher:.4f}")
    for feat_idx, fname in enumerate(herm_names):
        vals = [results[c]['herm'][:, feat_idx] for c in classes]
        grand_mean = np.mean(np.concatenate(vals))
        between = sum(len(v) * (np.mean(v) - grand_mean) ** 2 for v in vals)
        within = sum(np.sum((v - np.mean(v)) ** 2) for v in vals)
        fisher = between / (within + 1e-12)
        print(f"  HERM {fname:<10}: Fisher = {fisher:.4f}")


if __name__ == '__main__':
    print("Generating comparison data (50 samples per class at SNR=15dB)...\n")
    res = compare_herm_vs_bfp(n_samples_per_class=50, snr_db=15)
    print_comparison_report(res)
