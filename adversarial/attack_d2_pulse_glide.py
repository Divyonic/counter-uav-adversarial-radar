"""
Attack D2: Pulse-and-Glide Adversarial Drones
==============================================
Tests classifier robustness against drones that alternate between
propeller-on frames and propeller-off ("glide") frames within a 10-frame
LSTM window.

Hypothesis:
  The leakage test showed the LSTM works as a multi-instance classifier
  that aggregates evidence across parameter-diverse frames. If an attacker
  fills part of the 10-frame window with "glide" frames (body echo only,
  no propeller micro-Doppler), the multi-instance aggregation loses drone
  evidence and may default to bird/aircraft.

Attack variants span glide ratios from 0 (all pulse) to 1 (all glide).
"""

import numpy as np
import torch
import sys
import os
import json
sys.path.insert(0, os.path.dirname(__file__))

from fmcw_simulation import (generate_drone_signal, compute_spectrogram,
                              resize_spectrogram, extract_bfp_features,
                              generate_dataset)
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
               os.path.join(RESULTS_DIR, 'cnn_lstm_bfp_baseline_d2.pt'))
    return model, float(clean_acc)


def generate_pulse_frame(snr_db=15):
    """Drone with propellers ON (normal drone frame)."""
    params = dict(
        R0=np.random.uniform(500, 2000),
        v_bulk=np.random.uniform(5, 20),
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


def generate_glide_frame(snr_db=15):
    """Drone with propellers OFF (body echo only, no micro-Doppler)."""
    params = dict(
        R0=np.random.uniform(500, 2000),
        v_bulk=np.random.uniform(5, 20),
        snr_db=snr_db,
        n_blades=0,  # <-- no blades = no propeller micro-Doppler
        n_props=0,
        rpm=0,
        blade_len=0.0,
        tilt_angle=0,
    )
    beat = generate_drone_signal(**params)
    spec, f, t = compute_spectrogram(beat)
    fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
    return resize_spectrogram(spec, (128, 128)), extract_bfp_features(spec, fs_stft)


def build_pulse_glide_sequences(n_samples, glide_ratio, seq_len=10):
    """
    Build sequences where glide_ratio fraction of the seq_len frames are
    glide frames (no props) and the rest are pulse frames (normal drone).
    Positions are randomised within each sequence.
    """
    n_glide = int(round(seq_len * glide_ratio))
    n_pulse = seq_len - n_glide

    X_seq, bfp_seq = [], []
    for _ in range(n_samples):
        frames_specs, frames_bfps = [], []
        # generate n_pulse + n_glide frames
        for _ in range(n_pulse):
            s, b = generate_pulse_frame()
            frames_specs.append(s)
            frames_bfps.append(b)
        for _ in range(n_glide):
            s, b = generate_glide_frame()
            frames_specs.append(s)
            frames_bfps.append(b)
        # shuffle the order within the sequence (realistic: drone alternates)
        order = np.random.permutation(seq_len)
        X_seq.append([frames_specs[i] for i in order])
        bfp_seq.append([frames_bfps[i] for i in order])
    return np.array(X_seq, dtype=np.float32), np.array(bfp_seq, dtype=np.float32)


def run_attack(model, glide_ratio, n_samples):
    X_seq, bfp_seq = build_pulse_glide_sequences(n_samples, glide_ratio)
    preds = predict_sequences(model, X_seq, bfp_seq)
    correct_as_drone = int((preds == 0).sum())
    accuracy = correct_as_drone / len(preds)
    class_distribution = {CLASS_NAMES[c]: int((preds == c).sum()) for c in range(4)}
    return {
        'attack_name': f'D2_glide_{int(glide_ratio*100)}pct',
        'glide_ratio': float(glide_ratio),
        'n_pulse_frames': int(round((1 - glide_ratio) * 10)),
        'n_glide_frames': int(round(glide_ratio * 10)),
        'n_samples': int(n_samples),
        'accuracy_as_drone': float(accuracy),
        'class_distribution': class_distribution,
    }


def main():
    model, clean_acc = build_clean_dataset_and_train()

    glide_ratios = [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    print("\n" + "=" * 70)
    print("STEP 2: Running D2 attack variants (pulse-and-glide)")
    print("=" * 70)

    results = {
        'baseline_clean_test_accuracy': clean_acc,
        'attacks': [],
    }

    for gr in glide_ratios:
        print(f"\n  Running D2 glide_ratio={gr:.1f} ({int(gr*10)}/10 frames glide)...")
        r = run_attack(model, gr, N_ATTACK_SAMPLES)
        results['attacks'].append(r)
        print(f"    Accuracy (correct as drone): {r['accuracy_as_drone']:.3f}")
        print(f"    Class distribution: {r['class_distribution']}")

    out_path = os.path.join(RESULTS_DIR, 'attack_d2_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Glide ratio':<15} {'Pulse:Glide':<15} {'Accuracy':<12} {'Dominant class':<25}")
    print("-" * 67)
    for r in results['attacks']:
        dominant = max(r['class_distribution'].items(), key=lambda x: x[1])
        print(f"{r['glide_ratio']:<15.2f} "
              f"{r['n_pulse_frames']}:{r['n_glide_frames']:<13} "
              f"{r['accuracy_as_drone']:<12.3f} "
              f"{dominant[0]} ({dominant[1]})")


if __name__ == '__main__':
    main()
