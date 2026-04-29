"""
Attack A1: Pure RPM Reduction
==============================
Tests classifier robustness against drones flown at heavily reduced
propeller RPM but otherwise standard configuration (n_blades=2,
drone-typical bulk velocity).

Hypothesis (informed by A2 ablation):
  A2 demonstrated that the autocorrelation BFP extractor returns
  45 ± 59 Hz on clean drone data regardless of physical ground-truth
  blade-flash frequency. If BFP is numerically noise and the classifier
  uses bulk-Doppler peak position rather than harmonic content, A1
  should produce a null result (drone classified as drone) at every
  RPM in the sweep.

  This is an A-series completion test: A2 covered blade-count + RPM
  jointly. A1 isolates the RPM axis to confirm that BFP frequency
  manipulation alone cannot move the classifier.
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
    model, _ = train_cnn_lstm_model(
        Xs_tr, bfps_tr, ys_tr, Xs_val, bfps_val, ys_val, epochs=40)

    preds = predict_sequences(model, Xs_te, bfps_te)
    clean_acc = (preds == ys_te).mean()
    print(f"Clean-test accuracy: {clean_acc:.4f}")
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'cnn_lstm_bfp_baseline_a1.pt'))
    return model, float(clean_acc)


def generate_low_rpm_drone_sample(rpm, snr_db=15):
    params = dict(
        R0=np.random.uniform(500, 2000),
        v_bulk=np.random.uniform(5, 20),
        snr_db=snr_db,
        n_blades=2, n_props=4,    # standard drone (no blade-count change)
        rpm=rpm,
        blade_len=np.random.uniform(0.10, 0.15),
        tilt_angle=np.random.uniform(30, 60),
    )
    beat = generate_drone_signal(**params)
    spec, f, t = compute_spectrogram(beat)
    fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
    return resize_spectrogram(spec, (128, 128)), extract_bfp_features(spec, fs_stft)


def build_low_rpm_sequences(n_samples, rpm, seq_len=10):
    specs, bfps = [], []
    for _ in range(n_samples + seq_len - 1):
        s, b = generate_low_rpm_drone_sample(rpm=rpm)
        specs.append(s); bfps.append(b)
    specs = np.array(specs, dtype=np.float32)
    bfps = np.array(bfps, dtype=np.float32)
    X_seq = np.array([specs[i:i+seq_len] for i in range(n_samples)])
    bfp_seq = np.array([bfps[i:i+seq_len] for i in range(n_samples)])
    return X_seq, bfp_seq, bfps


def run_attack(model, rpm, n_samples):
    X_seq, bfp_seq, raw_bfps = build_low_rpm_sequences(n_samples, rpm=rpm)
    preds = predict_sequences(model, X_seq, bfp_seq)
    correct_as_drone = int((preds == 0).sum())
    accuracy = correct_as_drone / len(preds)
    class_distribution = {CLASS_NAMES[c]: int((preds == c).sum()) for c in range(4)}
    expected_bfp = 2 * rpm / 60   # n_blades=2
    return {
        'attack_name': f'A1_rpm_{rpm}',
        'rpm': int(rpm),
        'expected_bfp_hz': float(expected_bfp),
        'measured_bfp_hz_mean': float(np.mean(raw_bfps[:, 0])),
        'measured_bfp_hz_std': float(np.std(raw_bfps[:, 0])),
        'n_samples': int(n_samples),
        'accuracy_as_drone': float(accuracy),
        'class_distribution': class_distribution,
    }


def main():
    model, clean_acc = build_clean_dataset_and_train()

    rpms = [6000, 5000, 4000, 3000, 2000, 1500, 1000, 500]

    print("\n" + "=" * 70)
    print("STEP 2: Running A1 attack variants (pure RPM reduction)")
    print("=" * 70)

    results = {
        'baseline_clean_test_accuracy': clean_acc,
        'attacks': [],
    }

    for rpm in rpms:
        print(f"\n  Running A1 rpm={rpm} (expected BFP {2*rpm/60:.1f} Hz)...")
        r = run_attack(model, rpm, N_ATTACK_SAMPLES)
        results['attacks'].append(r)
        print(f"    Expected BFP: {r['expected_bfp_hz']:.1f} Hz | "
              f"Measured: {r['measured_bfp_hz_mean']:.1f} ± {r['measured_bfp_hz_std']:.1f}")
        print(f"    Accuracy (correct as drone): {r['accuracy_as_drone']:.3f}")
        print(f"    Class distribution: {r['class_distribution']}")

    out_path = os.path.join(RESULTS_DIR, 'attack_a1_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'RPM':<8} {'Exp BFP':<10} {'Meas BFP':<14} {'Accuracy':<12} {'Dominant class':<25}")
    print("-" * 73)
    for r in results['attacks']:
        dominant = max(r['class_distribution'].items(), key=lambda x: x[1])
        print(f"{r['rpm']:<8} "
              f"{r['expected_bfp_hz']:<10.1f} "
              f"{r['measured_bfp_hz_mean']:<14.1f} "
              f"{r['accuracy_as_drone']:<12.3f} "
              f"{dominant[0]} ({dominant[1]})")


if __name__ == '__main__':
    main()
