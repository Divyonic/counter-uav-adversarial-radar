"""
Attack B1: RAM-wrap / Bulk-amplitude Reduction
===============================================
Tests classifier robustness against drones whose total radar return is
attenuated by `dB_drop` dB, modelling a radar-absorbent material (RAM)
wrap or otherwise reduced-cross-section drone.

Hypothesis (from FINDINGS_attribution):
  The classifier identifies drones by the position and amplitude of the
  bulk-Doppler peak, not by harmonic structure. If we attenuate the
  return amplitude (bulk + micro-Doppler scale together), the bulk peak
  drops below drone-typical levels and the classifier should confuse
  drones with birds (smaller-amplitude bulk peak) or noise.

Implementation:
  Equivalent to attenuating target backscatter while ambient noise is
  fixed: pass snr_db = base_snr - dB_drop to generate_drone_signal.
  The simulator's noise_power = signal_power / 10**(snr_db/10) — lowering
  snr_db by dB_drop is mathematically the same as keeping noise_power
  constant and scaling signal_power down by 10**(-dB_drop/10).
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
BASE_SNR = 15  # matches training-set SNR
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"(seed={SEED}, n_attack_samples={N_ATTACK_SAMPLES}, base_snr={BASE_SNR}dB)")

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
    print(f"STEP 1: Generating clean 4-class dataset @ SNR={BASE_SNR}dB")
    print("=" * 70)
    X, X_bfp, y = generate_dataset(n_samples_per_class=300, snr_db=BASE_SNR)

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
               os.path.join(RESULTS_DIR, 'cnn_lstm_bfp_baseline_b1.pt'))
    return model, float(clean_acc)


def generate_attenuated_drone_sample(dB_drop=0):
    """Drone signal with effective SNR reduced by dB_drop (same as
    attenuating target return by dB_drop dB against fixed noise)."""
    params = dict(
        R0=np.random.uniform(500, 2000),
        v_bulk=np.random.uniform(5, 20),
        snr_db=BASE_SNR - dB_drop,
        n_blades=2, n_props=4,
        rpm=np.random.uniform(4000, 6000),
        blade_len=np.random.uniform(0.10, 0.15),
        tilt_angle=np.random.uniform(30, 60),
    )
    beat = generate_drone_signal(**params)
    spec, f, t = compute_spectrogram(beat)
    fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
    return resize_spectrogram(spec, (128, 128)), extract_bfp_features(spec, fs_stft)


def build_attenuated_sequences(n_samples, dB_drop, seq_len=10):
    specs, bfps = [], []
    for _ in range(n_samples + seq_len - 1):
        s, b = generate_attenuated_drone_sample(dB_drop=dB_drop)
        specs.append(s); bfps.append(b)
    specs = np.array(specs, dtype=np.float32)
    bfps = np.array(bfps, dtype=np.float32)
    X_seq = np.array([specs[i:i+seq_len] for i in range(n_samples)])
    bfp_seq = np.array([bfps[i:i+seq_len] for i in range(n_samples)])
    return X_seq, bfp_seq


def run_attack(model, dB_drop, n_samples):
    X_seq, bfp_seq = build_attenuated_sequences(n_samples, dB_drop)
    preds = predict_sequences(model, X_seq, bfp_seq)
    correct_as_drone = int((preds == 0).sum())
    accuracy = correct_as_drone / len(preds)
    class_distribution = {CLASS_NAMES[c]: int((preds == c).sum()) for c in range(4)}
    return {
        'attack_name': f'B1_drop_{dB_drop}dB',
        'rcs_drop_db': float(dB_drop),
        'effective_snr_db': float(BASE_SNR - dB_drop),
        'n_samples': int(n_samples),
        'accuracy_as_drone': float(accuracy),
        'class_distribution': class_distribution,
    }


def main():
    model, clean_acc = build_clean_dataset_and_train()

    drops_db = [0, 3, 6, 10, 15, 20]

    print("\n" + "=" * 70)
    print("STEP 2: Running B1 attack variants (RAM wrap / amplitude attenuation)")
    print("=" * 70)

    results = {
        'baseline_clean_test_accuracy': clean_acc,
        'base_snr_db': BASE_SNR,
        'attacks': [],
    }

    for d in drops_db:
        print(f"\n  Running B1 dB_drop={d}dB (effective SNR={BASE_SNR-d}dB)...")
        r = run_attack(model, d, N_ATTACK_SAMPLES)
        results['attacks'].append(r)
        print(f"    Accuracy (correct as drone): {r['accuracy_as_drone']:.3f}")
        print(f"    Class distribution: {r['class_distribution']}")

    out_path = os.path.join(RESULTS_DIR, 'attack_b1_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'RCS drop':<12} {'Eff SNR':<12} {'Accuracy':<12} {'Dominant class':<25}")
    print("-" * 64)
    for r in results['attacks']:
        dominant = max(r['class_distribution'].items(), key=lambda x: x[1])
        print(f"{r['rcs_drop_db']:<12.0f} "
              f"{r['effective_snr_db']:<12.0f} "
              f"{r['accuracy_as_drone']:<12.3f} "
              f"{dominant[0]} ({dominant[1]})")


if __name__ == '__main__':
    main()
