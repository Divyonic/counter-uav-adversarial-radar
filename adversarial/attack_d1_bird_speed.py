"""
Attack D1: Bird-Speed Flight
=============================
Tests classifier robustness against drones flown slowly enough that
their bulk-Doppler peak overlaps the bird-class velocity distribution.

Hypothesis (from FINDINGS_attribution):
  The classifier identifies drones primarily by the position of the
  bulk-Doppler peak. Training: drones at v_bulk = 5-20 m/s, birds at
  v_bulk = 5-15 m/s. The class-discriminative bulk-Doppler region for
  drones is the upper end (~12-20 m/s); the lower end overlaps birds.
  An attacker who flies the drone at 5-10 m/s (or slower) should be
  classified as a bird at increasing rates.

Variants sweep tighter and lower v_bulk windows.
"""

import numpy as np
import torch
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baseline'))

from fmcw_simulation import (generate_drone_signal, compute_spectrogram,
                              resize_spectrogram, extract_bfp_features,
                              generate_dataset, RadarParams)
from model import CNNLSTMClassifier
from train_and_evaluate import train_cnn_lstm_model, create_sequences, CLASS_NAMES

SEED = int(os.environ.get('ATTACK_SEED', '42'))
N_ATTACK_SAMPLES = int(os.environ.get('ATTACK_N_SAMPLES', '150'))
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"(seed={SEED}, n_attack_samples={N_ATTACK_SAMPLES})")

DEVICE = 'cpu'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def predict_sequences(model, X_seq, bfp_seq):
    model.eval()
    X_t = torch.FloatTensor(X_seq).unsqueeze(2).to(DEVICE)
    bfp_t = torch.FloatTensor(bfp_seq).to(DEVICE)
    preds = []
    batch = 16
    with torch.no_grad():
        for i in range(0, len(X_seq), batch):
            logits = model(X_t[i:i+batch], bfp_t[i:i+batch])
            preds.append(logits.argmax(dim=1).cpu().numpy())
    return np.concatenate(preds)


def build_clean_dataset_and_train():
    print("=" * 70)
    print("STEP 1: Generating clean 4-class dataset @ SNR=15dB")
    print("=" * 70)
    X, X_bfp, y = generate_dataset(n_samples_per_class=300, snr_db=15)

    n = len(y)
    n_tr, n_val = int(n * 0.70), int(n * 0.15)
    X_tr, X_val, X_te = X[:n_tr], X[n_tr:n_tr+n_val], X[n_tr+n_val:]
    bfp_tr, bfp_val, bfp_te = X_bfp[:n_tr], X_bfp[n_tr:n_tr+n_val], X_bfp[n_tr+n_val:]
    y_tr, y_val, y_te = y[:n_tr], y[n_tr:n_tr+n_val], y[n_tr+n_val:]

    Xs_tr, bfps_tr, ys_tr = create_sequences(X_tr, bfp_tr, y_tr, seq_len=10)
    Xs_val, bfps_val, ys_val = create_sequences(X_val, bfp_val, y_val, seq_len=10)
    Xs_te, bfps_te, ys_te = create_sequences(X_te, bfp_te, y_te, seq_len=10)

    print("Training CNN+LSTM+BFP baseline...")
    model, best_val = train_cnn_lstm_model(
        Xs_tr, bfps_tr, ys_tr, Xs_val, bfps_val, ys_val, epochs=40)

    preds = predict_sequences(model, Xs_te, bfps_te)
    clean_acc = (preds == ys_te).mean()
    print(f"Clean-test accuracy: {clean_acc:.4f}")
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'cnn_lstm_bfp_baseline_d1.pt'))
    return model, float(clean_acc)


def generate_slow_drone_sample(v_lo, v_hi, snr_db=15):
    params = dict(
        R0=np.random.uniform(500, 2000),
        v_bulk=np.random.uniform(v_lo, v_hi),
        snr_db=snr_db,
        n_blades=2, n_props=4,
        rpm=np.random.uniform(4000, 6000),
        blade_len=np.random.uniform(0.10, 0.15),
        tilt_angle=np.random.uniform(30, 60),
    )
    beat = generate_drone_signal(**params)
    spec, f, t = compute_spectrogram(beat)
    fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
    return resize_spectrogram(spec, (128, 128)), extract_bfp_features(spec, fs_stft)


def build_slow_sequences(n_samples, v_lo, v_hi, seq_len=10):
    specs, bfps = [], []
    for _ in range(n_samples + seq_len - 1):
        s, b = generate_slow_drone_sample(v_lo, v_hi)
        specs.append(s); bfps.append(b)
    specs = np.array(specs, dtype=np.float32)
    bfps = np.array(bfps, dtype=np.float32)
    X_seq = np.array([specs[i:i+seq_len] for i in range(n_samples)])
    bfp_seq = np.array([bfps[i:i+seq_len] for i in range(n_samples)])
    return X_seq, bfp_seq


def expected_doppler_hz(v_bulk):
    return 2 * RadarParams.fc * v_bulk / RadarParams.c


def run_attack(model, v_lo, v_hi, n_samples):
    X_seq, bfp_seq = build_slow_sequences(n_samples, v_lo, v_hi)
    preds = predict_sequences(model, X_seq, bfp_seq)
    correct_as_drone = int((preds == 0).sum())
    accuracy = correct_as_drone / len(preds)
    class_distribution = {CLASS_NAMES[c]: int((preds == c).sum()) for c in range(4)}
    return {
        'attack_name': f'D1_v_{v_lo:.0f}_{v_hi:.0f}',
        'v_lo_mps': float(v_lo),
        'v_hi_mps': float(v_hi),
        'expected_doppler_lo_hz': float(expected_doppler_hz(v_lo)),
        'expected_doppler_hi_hz': float(expected_doppler_hz(v_hi)),
        'n_samples': int(n_samples),
        'accuracy_as_drone': float(accuracy),
        'class_distribution': class_distribution,
    }


def main():
    model, clean_acc = build_clean_dataset_and_train()

    # Sweep from drone-typical (15-20 m/s) down to bird-typical (3-5 m/s)
    velocity_windows = [
        (15.0, 20.0),  # control: drone-typical upper end
        (10.0, 15.0),  # mid drone range
        (8.0, 12.0),   # bird-drone overlap
        (5.0, 10.0),   # paper's stated D1 attack range
        (5.0, 8.0),    # peak bird overlap
        (3.0, 5.0),    # below typical drone, slow bird
    ]

    print("\n" + "=" * 70)
    print("STEP 2: Running D1 attack variants (bird-speed flight)")
    print("=" * 70)

    results = {
        'baseline_clean_test_accuracy': clean_acc,
        'attacks': [],
    }

    for v_lo, v_hi in velocity_windows:
        print(f"\n  Running D1 v_bulk={v_lo:.1f}-{v_hi:.1f} m/s "
              f"(Doppler {expected_doppler_hz(v_lo):.0f}-{expected_doppler_hz(v_hi):.0f} Hz)...")
        r = run_attack(model, v_lo, v_hi, N_ATTACK_SAMPLES)
        results['attacks'].append(r)
        print(f"    Accuracy (correct as drone): {r['accuracy_as_drone']:.3f}")
        print(f"    Class distribution: {r['class_distribution']}")

    out_path = os.path.join(RESULTS_DIR, 'attack_d1_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'v_bulk (m/s)':<15} {'Doppler (Hz)':<18} {'Accuracy':<12} {'Dominant class':<25}")
    print("-" * 73)
    for r in results['attacks']:
        dominant = max(r['class_distribution'].items(), key=lambda x: x[1])
        print(f"{r['v_lo_mps']:.0f}-{r['v_hi_mps']:<11.0f} "
              f"{r['expected_doppler_lo_hz']:.0f}-{r['expected_doppler_hi_hz']:<13.0f} "
              f"{r['accuracy_as_drone']:<12.3f} "
              f"{dominant[0]} ({dominant[1]})")


if __name__ == '__main__':
    main()
