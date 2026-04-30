"""
Training and Evaluation Pipeline
=================================
- Generates synthetic dataset at multiple SNR levels
- Trains CNN-only, CNN+BFP, CNN+LSTM, CNN+LSTM+BFP models
- Evaluates: accuracy, precision, recall, F1, FAR, confusion matrix
- Runs ablation study
- Measures inference latency
- Saves all results to JSON
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             confusion_matrix)
import time
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fmcw_simulation import (generate_dataset, apply_cfar,
                               compute_range_doppler_map, RadarParams)
from model import (CNNClassifier, CNNBPFClassifier, CNNLSTMClassifier)

DEVICE = 'cpu'  # No GPU available in this environment
CLASS_NAMES = ['Enemy Drone', 'Bird', 'Friendly UAV', 'Manned Aircraft']
RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_sequences(X, X_bfp, y, seq_len=10):
    """Create overlapping sequences for LSTM training."""
    # Group by class to create within-class sequences
    X_seq, bfp_seq, y_seq = [], [], []

    for cls in range(4):
        mask = y == cls
        X_cls = X[mask]
        bfp_cls = X_bfp[mask]

        for i in range(len(X_cls) - seq_len + 1):
            X_seq.append(X_cls[i:i+seq_len])
            bfp_seq.append(bfp_cls[i:i+seq_len])
            y_seq.append(cls)

    X_seq = np.array(X_seq)
    bfp_seq = np.array(bfp_seq)
    y_seq = np.array(y_seq)

    # Shuffle
    idx = np.random.permutation(len(y_seq))
    return X_seq[idx], bfp_seq[idx], y_seq[idx]


def train_cnn_model(X_train, y_train, X_val, y_val,
                    model_class=CNNClassifier, epochs=50, lr=1e-3,
                    bfp_train=None, bfp_val=None, use_bfp=False):
    """Train a CNN-based model."""
    model = model_class().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Prepare data
    X_t = torch.FloatTensor(X_train).unsqueeze(1).to(DEVICE)  # Add channel dim
    y_t = torch.LongTensor(y_train).to(DEVICE)
    X_v = torch.FloatTensor(X_val).unsqueeze(1).to(DEVICE)

    if use_bfp:
        bfp_t = torch.FloatTensor(bfp_train).to(DEVICE)
        bfp_v = torch.FloatTensor(bfp_val).to(DEVICE)

    best_val_acc = 0
    patience_counter = 0
    patience = 10

    for epoch in range(epochs):
        model.train()
        # Mini-batch training
        batch_size = 32
        indices = np.random.permutation(len(X_train))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = X_t[batch_idx]
            yb = y_t[batch_idx]

            if use_bfp:
                bb = bfp_t[batch_idx]
                logits = model(xb, bb)
            else:
                logits = model(xb)

            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            if use_bfp:
                val_logits = model(X_v, bfp_v)
            else:
                val_logits = model(X_v)
            val_preds = val_logits.argmax(dim=1).cpu().numpy()
            val_acc = accuracy_score(y_val, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_acc


def train_cnn_lstm_model(X_train_seq, bfp_train_seq, y_train_seq,
                          X_val_seq, bfp_val_seq, y_val_seq,
                          epochs=50, lr=1e-3):
    """Train the full CNN+LSTM+BFP model."""
    model = CNNLSTMClassifier(n_classes=4, bfp_dim=3, seq_len=10).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # Prepare data - add channel dimension to spectrograms
    X_t = torch.FloatTensor(X_train_seq).unsqueeze(2).to(DEVICE)  # (N, T, 1, 128, 128)
    bfp_t = torch.FloatTensor(bfp_train_seq).to(DEVICE)
    y_t = torch.LongTensor(y_train_seq).to(DEVICE)

    X_v = torch.FloatTensor(X_val_seq).unsqueeze(2).to(DEVICE)
    bfp_v = torch.FloatTensor(bfp_val_seq).to(DEVICE)

    best_val_acc = 0
    patience_counter = 0
    patience = 8
    batch_size = 16  # Smaller batch for sequences

    for epoch in range(epochs):
        model.train()
        indices = np.random.permutation(len(y_train_seq))
        total_loss = 0
        n_batches = 0

        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            xb = X_t[batch_idx]
            bb = bfp_t[batch_idx]
            yb = y_t[batch_idx]

            logits = model(xb, bb)
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            # Process in batches to avoid memory issues
            val_preds = []
            for i in range(0, len(y_val_seq), batch_size):
                vl = model(X_v[i:i+batch_size], bfp_v[i:i+batch_size])
                val_preds.append(vl.argmax(dim=1).cpu().numpy())
            val_preds = np.concatenate(val_preds)
            val_acc = accuracy_score(y_val_seq, val_preds)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"    Early stopping at epoch {epoch+1}")
            break

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}: loss={total_loss/n_batches:.4f}, val_acc={val_acc:.4f}")

    model.load_state_dict(best_state)
    return model, best_val_acc


def evaluate_model(model, X_test, y_test, bfp_test=None, use_bfp=False,
                   is_sequence=False, seq_bfp=None):
    """Evaluate model and return detailed metrics."""
    model.eval()

    with torch.no_grad():
        if is_sequence:
            X_t = torch.FloatTensor(X_test).unsqueeze(2).to(DEVICE)
            bfp_t = torch.FloatTensor(seq_bfp).to(DEVICE)
            preds = []
            batch_size = 16
            for i in range(0, len(y_test), batch_size):
                logits = model(X_t[i:i+batch_size], bfp_t[i:i+batch_size])
                preds.append(logits.argmax(dim=1).cpu().numpy())
            preds = np.concatenate(preds)
        elif use_bfp:
            X_t = torch.FloatTensor(X_test).unsqueeze(1).to(DEVICE)
            bfp_t = torch.FloatTensor(bfp_test).to(DEVICE)
            logits = model(X_t, bfp_t)
            preds = logits.argmax(dim=1).cpu().numpy()
        else:
            X_t = torch.FloatTensor(X_test).unsqueeze(1).to(DEVICE)
            logits = model(X_t)
            preds = logits.argmax(dim=1).cpu().numpy()

    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average=None, labels=[0,1,2,3])
    cm = confusion_matrix(y_test, preds, labels=[0,1,2,3])

    # False Alarm Rate: non-threats classified as enemy drone
    non_threat_mask = y_test != 0  # Classes 1,2,3 are non-threats
    if non_threat_mask.sum() > 0:
        far = (preds[non_threat_mask] == 0).sum() / non_threat_mask.sum()
    else:
        far = 0.0

    return {
        'accuracy': float(acc),
        'precision': prec.tolist(),
        'recall': rec.tolist(),
        'f1': f1.tolist(),
        'far': float(far),
        'confusion_matrix': cm.tolist(),
    }


def measure_latency(model, is_sequence=False):
    """Measure inference latency."""
    model.eval()

    # Check if model forward expects bfp arg (CNNBPFClassifier)
    import inspect
    sig = inspect.signature(model.forward)
    needs_bfp = 'bfp' in sig.parameters

    # Warm up
    with torch.no_grad():
        if is_sequence:
            dummy_x = torch.randn(1, 10, 1, 128, 128).to(DEVICE)
            dummy_bfp = torch.randn(1, 10, 3).to(DEVICE)
            for _ in range(10):
                model(dummy_x, dummy_bfp)
        elif needs_bfp:
            dummy_x = torch.randn(1, 1, 128, 128).to(DEVICE)
            dummy_bfp = torch.randn(1, 3).to(DEVICE)
            for _ in range(10):
                model(dummy_x, dummy_bfp)
        else:
            dummy_x = torch.randn(1, 1, 128, 128).to(DEVICE)
            for _ in range(10):
                model(dummy_x)

    # Measure
    times = []
    n_runs = 100
    with torch.no_grad():
        for _ in range(n_runs):
            if is_sequence:
                x = torch.randn(1, 10, 1, 128, 128).to(DEVICE)
                bfp = torch.randn(1, 10, 3).to(DEVICE)
                start = time.perf_counter()
                model(x, bfp)
                end = time.perf_counter()
            elif needs_bfp:
                x = torch.randn(1, 1, 128, 128).to(DEVICE)
                bfp = torch.randn(1, 3).to(DEVICE)
                start = time.perf_counter()
                model(x, bfp)
                end = time.perf_counter()
            else:
                x = torch.randn(1, 1, 128, 128).to(DEVICE)
                start = time.perf_counter()
                model(x)
                end = time.perf_counter()
            times.append((end - start) * 1000)  # ms

    return {
        'mean_ms': float(np.mean(times)),
        'std_ms': float(np.std(times)),
        'median_ms': float(np.median(times)),
        'p95_ms': float(np.percentile(times, 95)),
    }


def compute_cfar_far(n_samples=200, snr_db=15):
    """Measure FAR with and without CFAR using synthetic Range-Doppler maps."""
    print("  Computing CFAR false alarm rates...")

    # Generate noise-only frames (no target)
    fixed_threshold_fa = 0
    cfar_fa = 0
    total_cells = 0

    for i in range(n_samples):
        # Pure noise
        Nc, Ns = RadarParams.Nc, RadarParams.Ns
        noise = np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns)
        rd_map = compute_range_doppler_map(noise)

        # Fixed threshold: mean + 3*std
        threshold_fixed = np.mean(rd_map) + 3 * np.std(rd_map)
        fixed_threshold_fa += np.sum(rd_map > threshold_fixed)

        # CFAR threshold
        detections, _ = apply_cfar(rd_map, n_train=16, n_guard=4, pfa=1e-4)
        cfar_fa += np.sum(detections)

        # Count valid cells (excluding edges)
        valid_h = Nc - 2 * (16 + 4)
        valid_w = Ns - 2 * (16 + 4)
        total_cells += valid_h * valid_w

    far_fixed = fixed_threshold_fa / (n_samples * Nc * Ns) * 100
    far_cfar = cfar_fa / total_cells * 100 if total_cells > 0 else 0

    return {'far_fixed_pct': float(far_fixed), 'far_cfar_pct': float(far_cfar)}


def run_full_experiment():
    """Run complete experiment pipeline."""
    all_results = {}

    print("=" * 60)
    print("COUNTER-UAV SIMULATION EXPERIMENT")
    print("=" * 60)

    # ============================
    # 1. Generate dataset at SNR=15 dB
    # ============================
    print("\n[1/7] Generating training dataset (SNR=15 dB)...")
    X, X_bfp, y = generate_dataset(n_samples_per_class=300, snr_db=15)
    print(f"  Dataset: {X.shape[0]} samples, shape={X.shape[1:]}")

    # Split: 70/15/15
    n = len(y)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)

    X_train, X_val, X_test = X[:n_train], X[n_train:n_train+n_val], X[n_train+n_val:]
    bfp_train, bfp_val, bfp_test = X_bfp[:n_train], X_bfp[n_train:n_train+n_val], X_bfp[n_train+n_val:]
    y_train, y_val, y_test = y[:n_train], y[n_train:n_train+n_val], y[n_train+n_val:]

    print(f"  Train: {len(y_train)}, Val: {len(y_val)}, Test: {len(y_test)}")

    # ============================
    # 2. Train CNN-only model
    # ============================
    print("\n[2/7] Training CNN-only model...")
    cnn_model, cnn_val_acc = train_cnn_model(
        X_train, y_train, X_val, y_val,
        model_class=CNNClassifier, epochs=40
    )
    cnn_results = evaluate_model(cnn_model, X_test, y_test)
    cnn_latency = measure_latency(cnn_model, is_sequence=False)
    print(f"  CNN-only: acc={cnn_results['accuracy']:.4f}, FAR={cnn_results['far']:.4f}")
    all_results['cnn_only'] = {**cnn_results, 'latency': cnn_latency}

    # ============================
    # 3. Train CNN+BFP model
    # ============================
    print("\n[3/7] Training CNN+BFP model...")
    cnn_bfp_model, cnn_bfp_val_acc = train_cnn_model(
        X_train, y_train, X_val, y_val,
        model_class=CNNBPFClassifier, epochs=40,
        bfp_train=bfp_train, bfp_val=bfp_val, use_bfp=True
    )
    cnn_bfp_results = evaluate_model(cnn_bfp_model, X_test, y_test,
                                      bfp_test=bfp_test, use_bfp=True)
    cnn_bfp_latency = measure_latency(cnn_bfp_model, is_sequence=False)
    print(f"  CNN+BFP: acc={cnn_bfp_results['accuracy']:.4f}, FAR={cnn_bfp_results['far']:.4f}")
    all_results['cnn_bfp'] = {**cnn_bfp_results, 'latency': cnn_bfp_latency}

    # ============================
    # 4. Train CNN+LSTM+BFP model
    # ============================
    print("\n[4/7] Creating sequences and training CNN+LSTM+BFP...")
    X_train_seq, bfp_train_seq, y_train_seq = create_sequences(X_train, bfp_train, y_train, seq_len=10)
    X_val_seq, bfp_val_seq, y_val_seq = create_sequences(X_val, bfp_val, y_val, seq_len=10)
    X_test_seq, bfp_test_seq, y_test_seq = create_sequences(X_test, bfp_test, y_test, seq_len=10)
    print(f"  Sequences: train={len(y_train_seq)}, val={len(y_val_seq)}, test={len(y_test_seq)}")

    lstm_model, lstm_val_acc = train_cnn_lstm_model(
        X_train_seq, bfp_train_seq, y_train_seq,
        X_val_seq, bfp_val_seq, y_val_seq,
        epochs=30
    )
    lstm_results = evaluate_model(lstm_model, X_test_seq, y_test_seq,
                                   is_sequence=True, seq_bfp=bfp_test_seq)
    lstm_latency = measure_latency(lstm_model, is_sequence=True)
    print(f"  CNN+LSTM+BFP: acc={lstm_results['accuracy']:.4f}, FAR={lstm_results['far']:.4f}")
    all_results['cnn_lstm_bfp'] = {**lstm_results, 'latency': lstm_latency}

    # ============================
    # 5. SNR sweep
    # ============================
    print("\n[5/7] SNR sweep experiment...")
    snr_levels = [0, 5, 10, 15, 20]
    snr_results = {}

    for snr in snr_levels:
        print(f"  SNR = {snr} dB...")
        X_snr, X_bfp_snr, y_snr = generate_dataset(n_samples_per_class=100, snr_db=snr)

        # Use last 30% as test
        n_test_start = int(0.7 * len(y_snr))
        X_snr_test = X_snr[n_test_start:]
        bfp_snr_test = X_bfp_snr[n_test_start:]
        y_snr_test = y_snr[n_test_start:]

        # CNN-only eval
        cnn_snr = evaluate_model(cnn_model, X_snr_test, y_snr_test)

        # CNN+BFP eval
        cnn_bfp_snr = evaluate_model(cnn_bfp_model, X_snr_test, y_snr_test,
                                      bfp_test=bfp_snr_test, use_bfp=True)

        # CNN+LSTM+BFP eval
        X_snr_seq, bfp_snr_seq, y_snr_seq = create_sequences(
            X_snr_test, bfp_snr_test, y_snr_test, seq_len=10
        )
        if len(y_snr_seq) > 0:
            lstm_snr = evaluate_model(lstm_model, X_snr_seq, y_snr_seq,
                                       is_sequence=True, seq_bfp=bfp_snr_seq)
        else:
            lstm_snr = {'accuracy': 0.0}

        snr_results[snr] = {
            'cnn_only': cnn_snr['accuracy'],
            'cnn_bfp': cnn_bfp_snr['accuracy'],
            'cnn_lstm_bfp': lstm_snr['accuracy'],
        }
        print(f"    CNN={cnn_snr['accuracy']:.3f}, CNN+BFP={cnn_bfp_snr['accuracy']:.3f}, "
              f"CNN+LSTM+BFP={lstm_snr['accuracy']:.3f}")

    all_results['snr_sweep'] = snr_results

    # ============================
    # 6. CFAR FAR measurement
    # ============================
    print("\n[6/7] CFAR false alarm rate measurement...")
    cfar_results = compute_cfar_far(n_samples=30)
    print(f"  Fixed threshold FAR: {cfar_results['far_fixed_pct']:.2f}%")
    print(f"  CA-CFAR FAR: {cfar_results['far_cfar_pct']:.4f}%")
    all_results['cfar'] = cfar_results

    # ============================
    # 7. Latency summary
    # ============================
    print("\n[7/7] Latency summary...")
    print(f"  CNN-only inference: {cnn_latency['mean_ms']:.1f} ± {cnn_latency['std_ms']:.1f} ms")
    print(f"  CNN+LSTM+BFP inference: {lstm_latency['mean_ms']:.1f} ± {lstm_latency['std_ms']:.1f} ms")

    # ============================
    # Save results
    # ============================
    results_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)
    print("\nClassification Accuracy (SNR=15 dB):")
    print(f"  CNN-only:      {all_results['cnn_only']['accuracy']*100:.1f}%")
    print(f"  CNN+BFP:       {all_results['cnn_bfp']['accuracy']*100:.1f}%")
    print(f"  CNN+LSTM+BFP:  {all_results['cnn_lstm_bfp']['accuracy']*100:.1f}%")
    print("\nFalse Alarm Rate:")
    print(f"  CNN-only:      {all_results['cnn_only']['far']*100:.1f}%")
    print(f"  CNN+BFP:       {all_results['cnn_bfp']['far']*100:.1f}%")
    print(f"  CNN+LSTM+BFP:  {all_results['cnn_lstm_bfp']['far']*100:.1f}%")
    print("\nCFAR:")
    print(f"  Fixed threshold FAR: {cfar_results['far_fixed_pct']:.2f}%")
    print(f"  CA-CFAR FAR:         {cfar_results['far_cfar_pct']:.4f}%")
    print("\nInference Latency (CPU):")
    print(f"  CNN-only:      {cnn_latency['mean_ms']:.1f} ms")
    print(f"  CNN+LSTM+BFP:  {lstm_latency['mean_ms']:.1f} ms")

    return all_results


if __name__ == '__main__':
    results = run_full_experiment()
