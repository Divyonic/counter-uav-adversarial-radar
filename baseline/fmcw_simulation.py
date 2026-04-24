"""
FMCW Radar Counter-UAV Simulation Pipeline (Vectorized)
========================================================
Generates synthetic FMCW radar signals for drones, birds, friendly UAVs,
and manned aircraft. Produces Range-Doppler maps, micro-Doppler spectrograms.
All signal generation is fully vectorized (no per-chirp loops).
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftshift
import os

np.random.seed(42)

# ============================================================
# FMCW RADAR PARAMETERS
# ============================================================
class RadarParams:
    fc = 9.5e9
    B = 400e6
    Tc = 100e-6
    CRI = 120e-6
    Nc = 512
    Ns = 256
    Pt = 1.0
    G_dBi = 25
    c = 3e8
    lam = c / fc
    fs = Ns / Tc
    PRF = 1 / CRI
    delta_R = c / (2 * B)
    R_max = c * Tc / 2
    v_max = lam * PRF / 4
    delta_v = lam / (2 * Nc * CRI)

    @classmethod
    def print_params(cls):
        print(f"  fc = {cls.fc/1e9:.1f} GHz, λ = {cls.lam*100:.2f} cm")
        print(f"  B = {cls.B/1e6:.0f} MHz, Tc = {cls.Tc*1e6:.0f} µs")
        print(f"  Range res = {cls.delta_R:.3f} m, Max range = {cls.R_max/1e3:.1f} km")
        print(f"  v_max = {cls.v_max:.1f} m/s, Δv = {cls.delta_v:.3f} m/s")
        print(f"  PRF = {cls.PRF:.0f} Hz, Nc = {cls.Nc}, Ns = {cls.Ns}")


# ============================================================
# VECTORIZED TARGET MODELS
# ============================================================

def generate_drone_signal(R0, v_bulk, snr_db, params=RadarParams,
                          n_blades=2, n_props=4, rpm=5000, blade_len=0.12,
                          tilt_angle=45):
    """Vectorized FMCW beat signal for a multi-rotor drone."""
    Nc, Ns = params.Nc, params.Ns
    rcs_body = 0.01 + np.random.uniform(-0.005, 0.005)

    t_fast = np.arange(Ns) / params.fs                    # (Ns,)
    t_slow = np.arange(Nc) * params.CRI                   # (Nc,)

    # Bulk target
    R = R0 + v_bulk * t_slow                               # (Nc,)
    tau = 2 * R / params.c                                 # (Nc,)
    f_beat = (params.B / params.Tc) * tau + 2 * v_bulk / params.lam  # (Nc,)
    phase_base = 4 * np.pi * R / params.lam                # (Nc,)

    # phase(m, n) = 2π * f_beat[m] * t_fast[n] + phase_base[m]
    phase = np.outer(f_beat, t_fast) * 2 * np.pi + phase_base[:, None]
    amplitude = np.sqrt(rcs_body) * 1e3
    beat_matrix = amplitude * np.exp(1j * phase)

    # Micro-Doppler from propellers (vectorized over chirps)
    omega_r = 2 * np.pi * rpm / 60
    theta = np.radians(tilt_angle)
    for prop in range(n_props):
        prop_offset = np.random.uniform(0, 2 * np.pi)
        for blade in range(n_blades):
            blade_phase = 2 * np.pi * blade / n_blades + prop_offset
            r_tip = blade_len
            v_tip = omega_r * r_tip * np.sin(theta) * np.cos(omega_r * t_slow + blade_phase)
            f_md = 2 * v_tip / params.lam
            # Stronger micro-Doppler return (0.3 instead of 0.15)
            md_amp = amplitude * 0.3 * np.sqrt(r_tip)
            md_phase = np.outer(f_md, t_fast) * 2 * np.pi
            beat_matrix += md_amp * np.exp(1j * (phase + md_phase))
            # Also add blade-root contribution at half amplitude
            r_root = r_tip * 0.5
            v_root = omega_r * r_root * np.sin(theta) * np.cos(omega_r * t_slow + blade_phase)
            f_md_root = 2 * v_root / params.lam
            beat_matrix += md_amp * 0.5 * np.exp(1j * (phase + np.outer(f_md_root, t_fast) * 2 * np.pi))

    # Add noise
    sig_power = np.mean(np.abs(beat_matrix) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns))
    return beat_matrix + noise


def generate_bird_signal(R0, v_bulk, snr_db, params=RadarParams,
                         flap_freq=5.0, wingspan=0.4):
    """Vectorized FMCW beat signal for a bird."""
    Nc, Ns = params.Nc, params.Ns
    rcs_body = 0.005 + np.random.uniform(-0.002, 0.003)

    t_fast = np.arange(Ns) / params.fs
    t_slow = np.arange(Nc) * params.CRI

    R = R0 + v_bulk * t_slow
    tau = 2 * R / params.c
    f_beat = (params.B / params.Tc) * tau + 2 * v_bulk / params.lam
    phase_base = 4 * np.pi * R / params.lam
    phase = np.outer(f_beat, t_fast) * 2 * np.pi + phase_base[:, None]
    amplitude = np.sqrt(rcs_body) * 1e3

    body_osc = 0.4 * np.sin(2 * np.pi * flap_freq * t_slow)
    beat_matrix = amplitude * (1 + body_osc[:, None]) * np.exp(1j * phase)

    # Wing micro-Doppler, stronger and more asymmetric than drones
    wing_phase_val = 2 * np.pi * flap_freq * t_slow
    v_wing = wingspan * 2 * np.pi * flap_freq * np.cos(wing_phase_val)
    # Asymmetry: downstroke 2x stronger
    asym = 1.0 + 1.0 * np.maximum(0, np.cos(wing_phase_val))
    f_md_wing = 2 * v_wing / params.lam
    md_amp = amplitude * 0.4 * asym
    md_phase = np.outer(f_md_wing, t_fast) * 2 * np.pi
    beat_matrix += md_amp[:, None] * np.exp(1j * (phase + md_phase))
    # Second harmonic from wing tip
    v_wing2 = wingspan * 0.7 * 2 * np.pi * flap_freq * np.cos(2 * wing_phase_val)
    f_md2 = 2 * v_wing2 / params.lam
    beat_matrix += (md_amp * 0.3)[:, None] * np.exp(1j * (phase + np.outer(f_md2, t_fast) * 2 * np.pi))

    sig_power = np.mean(np.abs(beat_matrix) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns))
    return beat_matrix + noise


def generate_friendly_uav_signal(R0, v_bulk, snr_db, params=RadarParams,
                                  n_blades=2, rpm=3500, blade_len=0.20):
    """Fixed-wing UAV with single pusher propeller, distinct from multi-rotor."""
    Nc, Ns = params.Nc, params.Ns
    rcs_body = 0.05 + np.random.uniform(-0.01, 0.02)  # Larger than multi-rotor

    t_fast = np.arange(Ns) / params.fs
    t_slow = np.arange(Nc) * params.CRI

    R = R0 + v_bulk * t_slow
    tau = 2 * R / params.c
    f_beat = (params.B / params.Tc) * tau + 2 * v_bulk / params.lam
    phase_base = 4 * np.pi * R / params.lam
    phase = np.outer(f_beat, t_fast) * 2 * np.pi + phase_base[:, None]
    amplitude = np.sqrt(rcs_body) * 1e3
    beat_matrix = amplitude * np.exp(1j * phase)

    # Single propeller micro-Doppler (1 prop vs 4 for multi-rotor)
    omega_r = 2 * np.pi * rpm / 60
    theta = np.radians(15)  # Prop axis nearly aligned with flight
    for blade in range(n_blades):
        blade_phase = 2 * np.pi * blade / n_blades
        v_tip = omega_r * blade_len * np.sin(theta) * np.cos(omega_r * t_slow + blade_phase)
        f_md = 2 * v_tip / params.lam
        md_amp = amplitude * 0.15 * np.sqrt(blade_len)
        beat_matrix += md_amp * np.exp(1j * (phase + np.outer(f_md, t_fast) * 2 * np.pi))

    # Wing flex modulation (slow, periodic from turbulence)
    wing_flex_freq = 1.5 + np.random.uniform(-0.5, 0.5)
    wing_mod = 1.0 + 0.06 * np.sin(2 * np.pi * wing_flex_freq * t_slow)
    beat_matrix *= wing_mod[:, None]

    sig_power = np.mean(np.abs(beat_matrix) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns))
    return beat_matrix + noise


def generate_aircraft_signal(R0, v_bulk, snr_db, params=RadarParams):
    """Vectorized FMCW beat signal for manned aircraft."""
    Nc, Ns = params.Nc, params.Ns
    rcs_body = 1.0 + np.random.uniform(-0.3, 0.5)

    t_fast = np.arange(Ns) / params.fs
    t_slow = np.arange(Nc) * params.CRI

    R = R0 + v_bulk * t_slow
    tau = 2 * R / params.c
    f_beat = (params.B / params.Tc) * tau + 2 * v_bulk / params.lam
    phase_base = 4 * np.pi * R / params.lam
    phase = np.outer(f_beat, t_fast) * 2 * np.pi + phase_base[:, None]
    amplitude = np.sqrt(rcs_body) * 1e3

    # Jet engine modulation, high frequency, small amplitude
    engine_mod_freq = 800 + np.random.uniform(-100, 100)
    engine_mod = 1.0 + 0.08 * np.sin(2 * np.pi * engine_mod_freq * t_slow)
    # Also add compressor blade vibration harmonics
    engine_mod += 0.04 * np.sin(2 * np.pi * engine_mod_freq * 2 * t_slow)
    beat_matrix = amplitude * engine_mod[:, None] * np.exp(1j * phase)
    # Fuselage scintillation (slow amplitude variation)
    scint = 1.0 + 0.1 * np.sin(2 * np.pi * 1.5 * t_slow)
    beat_matrix *= scint[:, None]

    sig_power = np.mean(np.abs(beat_matrix) ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns))
    return beat_matrix + noise


# ============================================================
# SIGNAL PROCESSING
# ============================================================

def compute_range_doppler_map(beat_matrix, window='hann'):
    Nc, Ns = beat_matrix.shape
    if window == 'hann':
        win_fast = np.hanning(Ns)[None, :]
        win_slow = np.hanning(Nc)[:, None]
    else:
        win_fast = np.ones((1, Ns))
        win_slow = np.ones((Nc, 1))
    range_fft = fft(beat_matrix * win_fast, axis=1)
    rd_map = fftshift(fft(range_fft * win_slow, axis=0), axes=0)
    return np.abs(rd_map) ** 2


def compute_spectrogram(beat_matrix, params=RadarParams, nperseg=64, noverlap=56):
    Nc, Ns = beat_matrix.shape
    win_fast = np.hanning(Ns)[None, :]
    range_fft = fft(beat_matrix * win_fast, axis=1)
    range_profile = np.mean(np.abs(range_fft) ** 2, axis=0)
    peak_bin = np.argmax(range_profile[:Ns // 2])
    bins = [max(0, peak_bin - 1), peak_bin, min(Ns - 1, peak_bin + 1)]
    slow_time_signal = np.mean(range_fft[:, bins], axis=1)
    f, t, Sxx = sig.stft(slow_time_signal, fs=params.PRF,
                          nperseg=nperseg, noverlap=noverlap,
                          window='hann', return_onesided=False)
    return np.abs(Sxx) ** 2, f, t


def resize_spectrogram(spectrogram, target_size=(128, 128)):
    from scipy.ndimage import zoom
    # Convert to dB scale first, this is critical for CNN features
    spec_db = 10 * np.log10(spectrogram + 1e-12)
    # Clip to dynamic range of 40 dB
    max_val = spec_db.max()
    spec_db = np.clip(spec_db, max_val - 40, max_val)
    # Resize
    h, w = spec_db.shape
    resized = zoom(spec_db, (target_size[0] / h, target_size[1] / w), order=1)
    # Normalize to [0, 1]
    resized = resized - resized.min()
    if resized.max() > 0:
        resized /= resized.max()
    return resized


def apply_cfar(rd_map, n_train=16, n_guard=4, pfa=1e-4):
    Nc, Ns = rd_map.shape
    alpha = n_train * 2 * (pfa ** (-1 / (2 * n_train)) - 1)
    detections = np.zeros_like(rd_map, dtype=bool)
    pad = n_guard + n_train
    for i in range(pad, Nc - pad):
        for j in range(pad, Ns - pad):
            train_r = np.concatenate([
                rd_map[i, j - pad:j - n_guard],
                rd_map[i, j + n_guard + 1:j + pad + 1]
            ])
            train_d = np.concatenate([
                rd_map[i - pad:i - n_guard, j],
                rd_map[i + n_guard + 1:i + pad + 1, j]
            ])
            noise_est = (np.mean(train_r) + np.mean(train_d)) / 2
            if rd_map[i, j] > alpha * noise_est:
                detections[i, j] = True
    return detections, alpha


def extract_bfp_features(spectrogram, fs_stft):
    envelope = np.sum(spectrogram, axis=0)
    if len(envelope) < 4:
        return np.array([0.0, 0.0, 0.0])
    envelope = envelope - np.mean(envelope)
    if np.std(envelope) > 0:
        envelope /= np.std(envelope)
    autocorr = np.correlate(envelope, envelope, mode='full')
    autocorr = autocorr[len(autocorr) // 2:]
    if autocorr[0] == 0:
        return np.array([0.0, 0.0, 0.0])
    autocorr /= autocorr[0]
    min_lag, max_lag = 2, min(len(autocorr) - 1, len(envelope) // 2)
    if max_lag <= min_lag:
        return np.array([0.0, 0.0, 0.0])
    peaks, _ = sig.find_peaks(autocorr[min_lag:max_lag], height=0.1)
    if len(peaks) == 0:
        return np.array([0.0, 0.0, 0.0])
    peak_idx = peaks[0] + min_lag
    dt_stft = 1.0 / fs_stft if fs_stft > 0 else 1.0
    f_bfp = 1.0 / (peak_idx * dt_stft) if peak_idx > 0 else 0.0
    c_bfp = autocorr[peak_idx]
    width_indices = np.where(autocorr[min_lag:max_lag] > c_bfp / 2)[0]
    delta_f_bfp = len(width_indices) * dt_stft
    return np.array([f_bfp, c_bfp, delta_f_bfp])


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(n_samples_per_class=500, snr_db=15, save_dir=None):
    print(f"Generating dataset: {n_samples_per_class} samples/class, SNR={snr_db} dB")

    spectrograms = []
    bfp_features = []
    labels = []

    class_configs = {
        0: ('Enemy Drone', generate_drone_signal, lambda: dict(
            R0=np.random.uniform(500, 2000), v_bulk=np.random.uniform(5, 20),
            snr_db=snr_db, n_blades=2, n_props=4,
            rpm=np.random.uniform(4000, 6000),
            blade_len=np.random.uniform(0.10, 0.15),
            tilt_angle=np.random.uniform(30, 60))),
        1: ('Bird', generate_bird_signal, lambda: dict(
            R0=np.random.uniform(200, 1500), v_bulk=np.random.uniform(5, 15),
            snr_db=snr_db, flap_freq=np.random.uniform(2, 12),
            wingspan=np.random.uniform(0.2, 0.8))),
        2: ('Friendly UAV', generate_friendly_uav_signal, lambda: dict(
            R0=np.random.uniform(300, 1500), v_bulk=np.random.uniform(15, 35),
            snr_db=snr_db, n_blades=2,
            rpm=np.random.uniform(2500, 4500),
            blade_len=np.random.uniform(0.15, 0.25))),
        3: ('Manned Aircraft', generate_aircraft_signal, lambda: dict(
            R0=np.random.uniform(1000, 5000), v_bulk=np.random.uniform(50, 100),
            snr_db=snr_db)),
    }

    for class_id, (name, gen_func, param_fn) in class_configs.items():
        print(f"  Generating {name}...", end=' ', flush=True)
        import sys
        for _ in range(n_samples_per_class):
            params = param_fn()
            beat_matrix = gen_func(**params)
            spec, f, t = compute_spectrogram(beat_matrix)
            spec_resized = resize_spectrogram(spec, (128, 128))
            fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
            bfp = extract_bfp_features(spec, fs_stft)
            spectrograms.append(spec_resized)
            bfp_features.append(bfp)
            labels.append(class_id)
        print("done", flush=True)

    X = np.array(spectrograms, dtype=np.float32)
    X_bfp = np.array(bfp_features, dtype=np.float32)
    y = np.array(labels, dtype=np.int64)
    idx = np.random.permutation(len(y))
    X, X_bfp, y = X[idx], X_bfp[idx], y[idx]

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, 'spectrograms.npy'), X)
        np.save(os.path.join(save_dir, 'bfp_features.npy'), X_bfp)
        np.save(os.path.join(save_dir, 'labels.npy'), y)
        print(f"  Saved to {save_dir}")

    return X, X_bfp, y


if __name__ == '__main__':
    print("FMCW Radar Parameters:")
    RadarParams.print_params()
    print()
    print("Quick benchmark: 1 drone signal...")
    import time
    t0 = time.time()
    beat = generate_drone_signal(R0=1000, v_bulk=10, snr_db=20)
    t1 = time.time()
    print(f"  Drone signal: {(t1-t0)*1000:.0f} ms")
    
    t0 = time.time()
    beat = generate_bird_signal(R0=600, v_bulk=8, snr_db=20)
    t1 = time.time()
    print(f"  Bird signal: {(t1-t0)*1000:.0f} ms")
    
    t0 = time.time()
    rd_map = compute_range_doppler_map(beat)
    t1 = time.time()
    print(f"  Range-Doppler map: {(t1-t0)*1000:.0f} ms")
    
    t0 = time.time()
    spec, f, t = compute_spectrogram(beat)
    spec_r = resize_spectrogram(spec)
    t1 = time.time()
    print(f"  Spectrogram + resize: {(t1-t0)*1000:.0f} ms")
    print(f"  Spec shape: {spec.shape} -> {spec_r.shape}")
    
    # Estimate dataset gen time
    t0 = time.time()
    for _ in range(10):
        beat = generate_drone_signal(R0=1000, v_bulk=10, snr_db=20)
        spec, f, t = compute_spectrogram(beat)
        resize_spectrogram(spec)
    t1 = time.time()
    per_sample = (t1 - t0) / 10
    print(f"\n  Per-sample time (drone): {per_sample*1000:.0f} ms")
    print(f"  Estimated dataset gen (500/class × 4): {4*500*per_sample:.0f} s")
