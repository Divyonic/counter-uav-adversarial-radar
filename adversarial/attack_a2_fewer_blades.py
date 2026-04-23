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

from fmcw_simulation import (generate_drone_signal, generate_bird_signal,
                              generate_friendly_uav_signal, generate_aircraft_signal,
                              compute_spectrogram, resize_spectrogram,
                              extract_bfp_features, generate_dataset, RadarParams)
from model import CNNLSTMClassifier
from train_and_evaluate import (train_cnn_lstm_model, create_sequences,
                                  evaluate_model, CLASS_NAMES)
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)
torch.manual_seed(42)

DEVICE = 'cpu'
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


# ============================================================
# STEP 1: Build clean dataset + train baseline model
# ============================================================

def build_clean_dataset_and_train():
    print("="*70)
    print("STEP 1: Generating clean 4-class dataset @ SNR=15dB")
    print("="*70)
    X, X_bfp, y = generate_dataset(n_samples_per_class=300, snr_db=15)
    print(f"Dataset: {X.shape[0]} samples")

    # 70/15/15 split
    n = len(y)
    n_tr, n_val = int(n*0.70), int(n*0.15)
    X_tr, X_val, X_te = X[:n_tr], X[n_tr:n_tr+n_val], X[n_tr+n_val:]
    bfp_tr, bfp_val, bfp_te = X_bfp[:n_tr], X_bfp[n_tr:n_tr+n_val], X_bfp[n_tr+n_val:]
    y_tr, y_val, y_te = y[:n_tr], y[n_tr:n_tr+n_val], y[n_tr+n_val:]

    # Create sequences for LSTM training
    Xs_tr, bfps_tr, ys_tr = create_sequences(X_tr, bfp_tr, y_tr, seq_len=10)
    Xs_val, bfps_val, ys_val = create_sequences(X_val, bfp_val, y_val, seq_len=10)
    Xs_te, bfps_te, ys_te = create_sequences(X_te, bfp_te, y_te, seq_len=10)

    print(f"Train sequences: {len(ys_tr)}, Val: {len(ys_val)}, Test: {len(ys_te)}")
    print("\nTraining CNN+LSTM+BFP baseline...")
    model, best_val = train_cnn_lstm_model(
        Xs_tr, bfps_tr, ys_tr, Xs_val, bfps_val, ys_val, epochs=40)
    print(f"Best val accuracy: {best_val:.4f}")

    # Baseline clean-test accuracy
    preds = evaluate_model(model, Xs_te, ys_te, seq_bfp=bfps_te, is_sequence=True)
    clean_acc = accuracy_score(ys_te, preds)
    print(f"Clean-test accuracy: {clean_acc:.4f}")

    # Save model for reuse
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'cnn_lstm_bfp_baseline.pt'))
    return model, clean_acc


# ============================================================
# STEP 2: Generate A2 adversarial drones (1-blade at various RPMs)
# ============================================================

def generate_adversarial_drone_sample(snr_db=15, n_blades=1, rpm=5000):
    """Generate ONE adversarial drone signal with fewer blades / altered RPM."""
    params = dict(
        R0=np.random.uniform(500, 2000),
        v_bulk=np.random.uniform(5, 20),
        snr_db=snr_db,
        n_blades=n_blades,           # <-- ATTACK: default 2, attack uses 1
        n_props=4,                   # keep 4 propellers (physical realism)
        rpm=rpm,                     # <-- ATTACK: can lower to push BFP into bird range
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
    """Generate n_samples adversarial drones, group into sequences labelled as 'drone' (class 0)."""
    specs, bfps = [], []
    for _ in range(n_samples + seq_len - 1):
        s, b = generate_adversarial_drone_sample(n_blades=n_blades, rpm=rpm)
        specs.append(s); bfps.append(b)
    specs = np.array(specs, dtype=np.float32)
    bfps = np.array(bfps, dtype=np.float32)

    # Build overlapping 10-frame sequences (all labelled class 0 = drone)
    X_seq = np.array([specs[i:i+seq_len] for i in range(n_samples)])
    bfp_seq = np.array([bfps[i:i+seq_len] for i in range(n_samples)])
    y_seq = np.zeros(n_samples, dtype=np.int64)  # ground truth: drone
    return X_seq, bfp_seq, y_seq, bfps


# ============================================================
# STEP 3: Attack the trained model
# ============================================================

def run_attack(model, attack_name, n_blades, rpm, n_samples=100):
    """Run one attack variant, return metrics."""
    X_seq, bfp_seq, y_seq, raw_bfps = build_adversarial_sequences(
        n_samples=n_samples, n_blades=n_blades, rpm=rpm)
    preds = evaluate_model(model, X_seq, y_seq, seq_bfp=bfp_seq, is_sequence=True)

    # Ground truth: all sequences are DRONES (class 0)
    # Accuracy = fraction correctly labelled as drone
    correct_as_drone = (preds == 0).sum()
    accuracy = correct_as_drone / len(preds)

    # What else did the model think they were?
    class_distribution = {CLASS_NAMES[c]: int((preds == c).sum()) for c in range(4)}

    # BFP feature stats: did the frequency-detector fire?
    bfp_freqs = raw_bfps[:, 0]  # first col = f_bfp
    bfp_confs = raw_bfps[:, 1]  # second col = c_bfp
    expected_bfp = n_blades * rpm / 60

    result = {
        'attack_name': attack_name,
        'n_blades': n_blades,
        'rpm': rpm,
        'expected_bfp_hz': float(expected_bfp),
        'measured_bfp_hz_mean': float(np.mean(bfp_freqs)),
        'measured_bfp_hz_std': float(np.std(bfp_freqs)),
        'bfp_confidence_mean': float(np.mean(bfp_confs)),
        'n_samples': n_samples,
        'accuracy_as_drone': float(accuracy),
        'class_distribution': class_distribution,
    }
    return result


def main():
    # Step 1: train baseline
    model, clean_acc = build_clean_dataset_and_train()

    # Step 2: define attack variants
    attacks = [
        ('clean_drone_control',  2, 5000),   # baseline, sanity check
        ('A2_pure_1blade',       1, 5000),   # pure A2: just fewer blades
        ('A2+A1_mild',           1, 3000),   # combined with moderate RPM drop
        ('A2+A1_aggressive',     1, 2000),   # lower RPM (BFP ~33 Hz, edge of bird range)
        ('A2+A1_extreme',        1, 1200),   # BFP ~20 Hz, fully in bird range
        ('A2+A1_bird_mimic',     1, 800),    # BFP ~13 Hz (typical bird flap)
    ]

    print("\n" + "="*70)
    print("STEP 2: Running A2 attack variants")
    print("="*70)

    results = {
        'baseline_clean_test_accuracy': float(clean_acc),
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

    # Save results
    out_path = os.path.join(RESULTS_DIR, 'attack_a2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"{'Attack':<25} {'BFP (Hz)':<12} {'Accuracy':<12} {'Dominant class':<20}")
    print("-"*70)
    for r in results['attacks']:
        dominant = max(r['class_distribution'].items(), key=lambda x: x[1])
        print(f"{r['attack_name']:<25} "
              f"{r['measured_bfp_hz_mean']:<12.1f} "
              f"{r['accuracy_as_drone']:<12.3f} "
              f"{dominant[0]} ({dominant[1]})")


if __name__ == '__main__':
    main()
