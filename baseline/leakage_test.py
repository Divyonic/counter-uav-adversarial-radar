"""
Data Leakage Test: Does the LSTM exploit within-sequence parameter consistency?
================================================================================
Test: Create sequences where each frame has INDEPENDENTLY RANDOMIZED parameters
(different RPM, flap freq, range, velocity per frame within the same 10-frame window).
If LSTM accuracy collapses, the original 95.8% was a synthetic-data artifact.
"""
import numpy as np
import torch
import sys, os, json, time
sys.path.insert(0, os.path.dirname(__file__))

from fmcw_simulation import (generate_drone_signal, generate_bird_signal,
                               generate_friendly_uav_signal, generate_aircraft_signal,
                               compute_spectrogram, resize_spectrogram,
                               extract_bfp_features)
from model import CNNLSTMClassifier, count_parameters
from sklearn.metrics import accuracy_score, confusion_matrix
from train_and_evaluate import train_cnn_lstm_model, create_sequences, evaluate_model

np.random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')


def generate_single_sample(class_id, snr_db=15):
    """Generate one sample with random parameters for a given class."""
    if class_id == 0:  # drone
        params = dict(R0=np.random.uniform(500,2000), v_bulk=np.random.uniform(5,20),
                      snr_db=snr_db, n_blades=2, n_props=4,
                      rpm=np.random.uniform(4000,6000),
                      blade_len=np.random.uniform(0.10,0.15),
                      tilt_angle=np.random.uniform(30,60))
        beat = generate_drone_signal(**params)
    elif class_id == 1:  # bird
        params = dict(R0=np.random.uniform(200,1500), v_bulk=np.random.uniform(5,15),
                      snr_db=snr_db, flap_freq=np.random.uniform(2,12),
                      wingspan=np.random.uniform(0.2,0.8))
        beat = generate_bird_signal(**params)
    elif class_id == 2:  # fixed-wing UAV
        params = dict(R0=np.random.uniform(300,1500), v_bulk=np.random.uniform(15,35),
                      snr_db=snr_db, n_blades=2,
                      rpm=np.random.uniform(2500,4500),
                      blade_len=np.random.uniform(0.15,0.25))
        beat = generate_friendly_uav_signal(**params)
    else:  # aircraft
        params = dict(R0=np.random.uniform(1000,5000), v_bulk=np.random.uniform(50,100),
                      snr_db=snr_db)
        beat = generate_aircraft_signal(**params)

    spec, f, t = compute_spectrogram(beat)
    spec_r = resize_spectrogram(spec, (128, 128))
    fs_stft = len(t) / (t[-1] - t[0]) if len(t) > 1 and t[-1] > t[0] else 1.0
    bfp = extract_bfp_features(spec, fs_stft)
    return spec_r, bfp


def generate_randomized_sequences(n_seq_per_class=200, seq_len=10, snr_db=15):
    """
    Generate sequences where EACH FRAME has independently randomized parameters.
    This breaks within-sequence parameter consistency.
    """
    print(f"Generating randomized sequences: {n_seq_per_class}/class, T={seq_len}")
    X_seq, bfp_seq, y_seq = [], [], []

    for cls in range(4):
        cls_names = ['Drone', 'Bird', 'FW-UAV', 'Aircraft']
        print(f"  {cls_names[cls]}...", end=' ', flush=True)
        for _ in range(n_seq_per_class):
            frames = []
            bfps = []
            for t in range(seq_len):
                spec, bfp = generate_single_sample(cls, snr_db)
                frames.append(spec)
                bfps.append(bfp)
            X_seq.append(np.array(frames))
            bfp_seq.append(np.array(bfps))
            y_seq.append(cls)
        print("done", flush=True)

    X_seq = np.array(X_seq, dtype=np.float32)
    bfp_seq = np.array(bfp_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)

    idx = np.random.permutation(len(y_seq))
    return X_seq[idx], bfp_seq[idx], y_seq[idx]


def generate_consistent_sequences(n_seq_per_class=200, seq_len=10, snr_db=15):
    """
    Generate sequences where ALL frames share the SAME parameters (only noise differs).
    This is what the original pipeline does (within-class blocks).
    """
    print(f"Generating consistent sequences: {n_seq_per_class}/class, T={seq_len}")
    X_seq, bfp_seq, y_seq = [], [], []

    for cls in range(4):
        cls_names = ['Drone', 'Bird', 'FW-UAV', 'Aircraft']
        print(f"  {cls_names[cls]}...", end=' ', flush=True)
        for _ in range(n_seq_per_class):
            # Draw parameters ONCE for all frames
            if cls == 0:
                params = dict(R0=np.random.uniform(500,2000), v_bulk=np.random.uniform(5,20),
                              snr_db=snr_db, n_blades=2, n_props=4,
                              rpm=np.random.uniform(4000,6000),
                              blade_len=np.random.uniform(0.10,0.15),
                              tilt_angle=np.random.uniform(30,60))
                gen_func = generate_drone_signal
            elif cls == 1:
                params = dict(R0=np.random.uniform(200,1500), v_bulk=np.random.uniform(5,15),
                              snr_db=snr_db, flap_freq=np.random.uniform(2,12),
                              wingspan=np.random.uniform(0.2,0.8))
                gen_func = generate_bird_signal
            elif cls == 2:
                params = dict(R0=np.random.uniform(300,1500), v_bulk=np.random.uniform(15,35),
                              snr_db=snr_db, n_blades=2,
                              rpm=np.random.uniform(2500,4500),
                              blade_len=np.random.uniform(0.15,0.25))
                gen_func = generate_friendly_uav_signal
            else:
                params = dict(R0=np.random.uniform(1000,5000), v_bulk=np.random.uniform(50,100),
                              snr_db=snr_db)
                gen_func = generate_aircraft_signal

            frames, bfps = [], []
            for t in range(seq_len):
                beat = gen_func(**params)  # Same params, different noise
                spec, f, t_ax = compute_spectrogram(beat)
                spec_r = resize_spectrogram(spec, (128, 128))
                fs_stft = len(t_ax) / (t_ax[-1] - t_ax[0]) if len(t_ax) > 1 and t_ax[-1] > t_ax[0] else 1.0
                bfp = extract_bfp_features(spec, fs_stft)
                frames.append(spec_r)
                bfps.append(bfp)
            X_seq.append(np.array(frames))
            bfp_seq.append(np.array(bfps))
            y_seq.append(cls)
        print("done", flush=True)

    X_seq = np.array(X_seq, dtype=np.float32)
    bfp_seq = np.array(bfp_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.int64)

    idx = np.random.permutation(len(y_seq))
    return X_seq[idx], bfp_seq[idx], y_seq[idx]


def run_leakage_test():
    print("=" * 60)
    print("DATA LEAKAGE TEST")
    print("=" * 60)

    # === Test A: Train & eval on CONSISTENT sequences (same params within seq) ===
    print("\n[A] Consistent sequences (same params per sequence)...")
    X_con, bfp_con, y_con = generate_consistent_sequences(n_seq_per_class=150, seq_len=10)
    n = len(y_con)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    print(f"  Training on {n_train} sequences...")
    model_con, val_acc_con = train_cnn_lstm_model(
        X_con[:n_train], bfp_con[:n_train], y_con[:n_train],
        X_con[n_train:n_train+n_val], bfp_con[n_train:n_train+n_val], y_con[n_train:n_train+n_val],
        epochs=25
    )

    # Eval on consistent test set
    test_con = evaluate_model(model_con, X_con[n_train+n_val:], y_con[n_train+n_val:],
                               is_sequence=True, seq_bfp=bfp_con[n_train+n_val:])
    print(f"  [A] Consistent test acc: {test_con['accuracy']*100:.1f}%")

    # Eval SAME model on RANDOMIZED test set
    print("\n  Generating randomized test sequences for cross-eval...")
    X_rand_test, bfp_rand_test, y_rand_test = generate_randomized_sequences(
        n_seq_per_class=50, seq_len=10)
    test_rand = evaluate_model(model_con, X_rand_test, y_rand_test,
                                is_sequence=True, seq_bfp=bfp_rand_test)
    print(f"  [A→rand] Randomized test acc: {test_rand['accuracy']*100:.1f}%")

    # === Test B: Train & eval on RANDOMIZED sequences ===
    print("\n[B] Randomized sequences (different params per frame)...")
    X_rnd, bfp_rnd, y_rnd = generate_randomized_sequences(n_seq_per_class=150, seq_len=10)
    n = len(y_rnd)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    print(f"  Training on {n_train} sequences...")
    model_rnd, val_acc_rnd = train_cnn_lstm_model(
        X_rnd[:n_train], bfp_rnd[:n_train], y_rnd[:n_train],
        X_rnd[n_train:n_train+n_val], bfp_rnd[n_train:n_train+n_val], y_rnd[n_train:n_train+n_val],
        epochs=25
    )

    test_rnd = evaluate_model(model_rnd, X_rnd[n_train+n_val:], y_rnd[n_train+n_val:],
                               is_sequence=True, seq_bfp=bfp_rnd[n_train+n_val:])
    print(f"  [B] Randomized test acc: {test_rnd['accuracy']*100:.1f}%")

    # === Summary ===
    print("\n" + "=" * 60)
    print("LEAKAGE TEST RESULTS")
    print("=" * 60)
    print(f"  [A] Consistent-seq model on consistent test:   {test_con['accuracy']*100:.1f}%")
    print(f"  [A→R] Consistent-seq model on randomized test: {test_rand['accuracy']*100:.1f}%")
    print(f"  [B] Randomized-seq model on randomized test:   {test_rnd['accuracy']*100:.1f}%")
    print()

    gap = test_con['accuracy'] - test_rand['accuracy']
    if gap > 0.15:
        print(f"  ⚠️  LEAKAGE DETECTED: {gap*100:.1f} pp drop when parameters randomized.")
        print(f"      The LSTM is exploiting within-sequence parameter consistency.")
    elif gap > 0.05:
        print(f"  ⚠️  PARTIAL LEAKAGE: {gap*100:.1f} pp drop. Some benefit from consistency.")
    else:
        print(f"  ✅ NO LEAKAGE: Only {gap*100:.1f} pp drop. LSTM learns genuine class features.")

    results = {
        'consistent_on_consistent': test_con['accuracy'],
        'consistent_on_randomized': test_rand['accuracy'],
        'randomized_on_randomized': test_rnd['accuracy'],
        'leakage_gap_pp': gap * 100,
        'consistent_cm': test_con['confusion_matrix'],
        'randomized_cm': test_rnd['confusion_matrix'],
    }

    with open(os.path.join(RESULTS_DIR, 'leakage_test_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to results/leakage_test_results.json")

    return results


if __name__ == '__main__':
    run_leakage_test()
