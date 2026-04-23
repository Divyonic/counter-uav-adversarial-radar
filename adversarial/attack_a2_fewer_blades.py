"""
Attack A2: Fewer-Blade Adversarial Drones
==========================================
Tests classifier robustness against drones with 1 blade per propeller
(vs. the 2-blade drones in the training set) at progressively lower RPM.

Hypothesis: BFP frequency = n_blades * rpm / 60. The training set uses
n_blades=2, rpm ~5000, giving BFP ~167 Hz. An attacker using 1-blade
props at lowered RPM can push BFP into the bird-like 5-20 Hz range and
evade the classifier.

Outputs:
  - results/attack_a2_results.json
  - Console report with accuracy drop per attack variant
"""

import numpy as np
import torch
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from fmcw_simulation import (generate_drone_signal, compute_spectrogram,
                              resize_spectrogram, extract_bfp_features,
                              generate_dataset, RadarParams)
from model import CNNLSTMClassifier
from train_and_evaluate import train_cnn_lstm_model, create_sequences, CLASS_NAMES

np.random.seed(42)
torch.manual_seed(42)

DEVICE = 'cpu'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def predict_sequences(model, X_seq, bfp_seq):
    """Run model on sequences, return predictions array."""
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


# ============================================================
# STEP 1: Build clean dataset + train baseline model
# ============================================================

def build_clean_dataset_and_train():
    print("=" * 70)
    print("STEP 1: Generating clean 4-class dataset @ SNR=15dB")
    print("=" * 70)
    X, X_bfp, y = generate_dataset(n_samples_per_class=300, snr_db=15)
    print(f"Dataset: {X.shape[0]} samples")

    n = len(y)
    n_tr, n_val = int(n * 0.70), int(n * 0.15)
    X_tr, X_val, X_te = X[:n_tr], X[n_tr:n_tr + n_val], X[n_tr + n_val:]
    bfp_tr, bfp_val, bfp_te = X_bfp[:n_tr], X_bfp[n_tr:n_tr + n_val], X_bfp[n_tr + n_val:]
    y_tr, y_val, y_te = y[:n_tr], y[n_tr:n_tr + n_val], y[n_tr + n_val:]

    Xs_tr, bfps_tr, ys_tr = create_sequences(X_tr, bfp_tr, y_tr, seq_len=10)
    Xs_val, bfps_val, ys_val = create_sequences(X_val, bfp_val, y_val, seq_len=10)
    Xs_te, bfps_te, ys_te = create_sequences(X_te, bfp_te, y_te, seq_len=10)

    print(f"Train sequences: {len(ys_tr)}, Val: {len(ys_val)}, Test: {len(ys_te)}")
    print("\nTraining CNN+LSTM+BFP baseline...")
    model, best_val = train_cnn_lstm_model(
        Xs_tr, bfps_tr, ys_tr, Xs_val, bfps_val, ys_val, epochs=40)
    print(f"Best val accuracy: {best_val:.4f}")

    preds = predict_sequences(model, Xs_te, bfps_te)
    clean_acc = (preds == ys_te).mean()
    print(f"Clean-test accuracy: {clean_acc:.4f}")

    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'cnn_lstm_bfp_baseline.pt'))
    return model, float(clean_acc)


# ============================================================
# STEP 2: Adversarial drone generation
# ============================================================

def generate_adversarial_drone_sample(snr_db=15, n_blades=1, rpm=5000):
    params = dict(
        R0=np.random.uniform(500, 2000),
        v_bulk=np.random.uniform(5, 20),
        snr_db=snr_db,
        n_blades=n_blades,
        n_props=4,
        rpm=rpm,
        blade_len=np.random.uniform(0.10, 0.15),
        tilt_angle=np.random.uniform(30, 60),
    )
    beat = generate_drone_signal(**params)
    spec, f, t = compute_spectrogram(beat)
    spec_resized = resize_spectrogram(spec, (128, 128))
    fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
    bfp = extract_bfp_features(spec, fs_stft)
    return spec_resized, bfp


def build_adversarial_sequences(n_samples=100, n_blades=1, rpm=5000, seq_len=10):
    specs, bfps = [], []
    for _ in range(n_samples + seq_len - 1):
        s, b = generate_adversarial_drone_sample(n_blades=n_blades, rpm=rpm)
        specs.append(s)
        bfps.append(b)
    specs = np.array(specs, dtype=np.float32)
    bfps = np.array(bfps, dtype=np.float32)
    X_seq = np.array([specs[i:i + seq_len] for i in range(n_samples)])
    bfp_seq = np.array([bfps[i:i + seq_len] for i in range(n_samples)])
    return X_seq, bfp_seq, bfps


# ============================================================
# STEP 3: Run attacks
# ============================================================

def run_attack(model, attack_name, n_blades, rpm, n_samples=100):
    X_seq, bfp_seq, raw_bfps = build_adversarial_sequences(
        n_samples=n_samples, n_blades=n_blades, rpm=rpm)
    preds = predict_sequences(model, X_seq, bfp_seq)

    correct_as_drone = int((preds == 0).sum())
    accuracy = correct_as_drone / len(preds)
    class_distribution = {CLASS_NAMES[c]: int((preds == c).sum()) for c in range(4)}

    bfp_freqs = raw_bfps[:, 0]
    bfp_confs = raw_bfps[:, 1]
    expected_bfp = n_blades * rpm / 60

    return {
        'attack_name': attack_name,
        'n_blades': int(n_blades),
        'rpm': int(rpm),
        'expected_bfp_hz': float(expected_bfp),
        'measured_bfp_hz_mean': float(np.mean(bfp_freqs)),
        'measured_bfp_hz_std': float(np.std(bfp_freqs)),
        'bfp_confidence_mean': float(np.mean(bfp_confs)),
        'n_samples': int(n_samples),
        'accuracy_as_drone': float(accuracy),
        'class_distribution': class_distribution,
    }


def main():
    model, clean_acc = build_clean_dataset_and_train()

    attacks = [
        ('clean_drone_control', 2, 5000),
        ('A2_pure_1blade',      1, 5000),
        ('A2+A1_mild',          1, 3000),
        ('A2+A1_aggressive',    1, 2000),
        ('A2+A1_extreme',       1, 1200),
        ('A2+A1_bird_mimic',    1, 800),
    ]

    print("\n" + "=" * 70)
    print("STEP 2: Running A2 attack variants")
    print("=" * 70)

    results = {
        'baseline_clean_test_accuracy': clean_acc,
        'attacks': [],
    }

    for name, n_blades, rpm in attacks:
        print(f"\n  Running {name}: n_blades={n_blades}, rpm={rpm}...")
        r = run_attack(model, name, n_blades, rpm, n_samples=100)
        results['attacks'].append(r)
        print(f"    Expected BFP: {r['expected_bfp_hz']:.1f} Hz | "
              f"Measured: {r['measured_bfp_hz_mean']:.1f} ± {r['measured_bfp_hz_std']:.1f}")
        print(f"    Accuracy (correct as drone): {r['accuracy_as_drone']:.3f}")
        print(f"    Class distribution: {r['class_distribution']}")

    out_path = os.path.join(RESULTS_DIR, 'attack_a2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Attack':<25} {'BFP (Hz)':<12} {'Accuracy':<12} {'Dominant class':<25}")
    print("-" * 74)
    for r in results['attacks']:
        dominant = max(r['class_distribution'].items(), key=lambda x: x[1])
        print(f"{r['attack_name']:<25} "
              f"{r['measured_bfp_hz_mean']:<12.1f} "
              f"{r['accuracy_as_drone']:<12.3f} "
              f"{dominant[0]} ({dominant[1]})")


if __name__ == '__main__':
    main()
