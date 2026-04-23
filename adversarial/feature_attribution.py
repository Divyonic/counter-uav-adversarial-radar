"""
Feature Attribution via Permutation Importance
================================================
Formally answers: what is the CNN+LSTM+BFP classifier actually using?

For each feature group, we randomise/shuffle/mask that feature in the
test set and measure how much accuracy drops. Large drop = important
feature. Small drop = feature is not doing discriminative work.

Tests:
  1. BFP permutation — shuffle BFP vectors across samples. If accuracy
     stays high, BFP is inert (confirms our A2 and HERM findings).
  2. Spectrogram permutation — shuffle spectrograms across samples
     (sanity check: main input should be important).
  3. Frame-order permutation within sequence — shuffle the 10 frames
     inside each sequence. If accuracy stays high, the LSTM does not
     depend on temporal order — confirming multi-instance aggregation.
  4. Bulk-Doppler masking — zero out the central (low-Doppler) band of
     each spectrogram. Measures dependence on bulk target velocity.
  5. Micro-Doppler masking — zero out the outer (high-Doppler) bands.
     Measures dependence on micro-Doppler sidebands.

Output: results/feature_attribution_results.json with accuracy per test.
"""

import numpy as np
import torch
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))

from fmcw_simulation import generate_dataset
from model import CNNLSTMClassifier
from train_and_evaluate import train_cnn_lstm_model, create_sequences, CLASS_NAMES

SEED = int(os.environ.get('ATTR_SEED', '42'))
np.random.seed(SEED)
torch.manual_seed(SEED)
print(f"(seed={SEED})")

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


def accuracy(preds, labels):
    return float(np.mean(preds == labels))


# ============================================================
# Build model (train once, reuse for all tests)
# ============================================================

def build_model_and_test_set():
    print("=" * 70)
    print("Training baseline CNN+LSTM+BFP (one time, reused for all tests)")
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

    model, _ = train_cnn_lstm_model(Xs_tr, bfps_tr, ys_tr,
                                     Xs_val, bfps_val, ys_val, epochs=40)
    baseline_preds = predict_sequences(model, Xs_te, bfps_te)
    baseline_acc = accuracy(baseline_preds, ys_te)
    print(f"Clean baseline accuracy: {baseline_acc:.4f}")
    return model, Xs_te, bfps_te, ys_te, baseline_acc


# ============================================================
# Attribution tests
# ============================================================

def test_bfp_permutation(model, Xs_te, bfps_te, ys_te, n_trials=5):
    """Shuffle BFP vectors across samples; spectrograms unchanged."""
    drops = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 7 + 1)
        idx = rng.permutation(len(bfps_te))
        bfps_shuffled = bfps_te[idx]
        preds = predict_sequences(model, Xs_te, bfps_shuffled)
        drops.append(accuracy(preds, ys_te))
    return {
        'test': 'BFP permutation (across samples)',
        'mean_acc': float(np.mean(drops)),
        'std_acc': float(np.std(drops)),
        'n_trials': n_trials,
    }


def test_spectrogram_permutation(model, Xs_te, bfps_te, ys_te, n_trials=3):
    """Shuffle spectrograms across samples; BFP unchanged."""
    drops = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 11 + 2)
        idx = rng.permutation(len(Xs_te))
        X_shuffled = Xs_te[idx]
        preds = predict_sequences(model, X_shuffled, bfps_te)
        drops.append(accuracy(preds, ys_te))
    return {
        'test': 'Spectrogram permutation (across samples)',
        'mean_acc': float(np.mean(drops)),
        'std_acc': float(np.std(drops)),
        'n_trials': n_trials,
    }


def test_frame_order_shuffle(model, Xs_te, bfps_te, ys_te, n_trials=3):
    """Shuffle the 10 frames within each sequence; keep sample assignments."""
    drops = []
    for trial in range(n_trials):
        rng = np.random.default_rng(trial * 13 + 3)
        X_shuffled = Xs_te.copy()
        bfp_shuffled = bfps_te.copy()
        for i in range(len(Xs_te)):
            order = rng.permutation(Xs_te.shape[1])
            X_shuffled[i] = Xs_te[i][order]
            bfp_shuffled[i] = bfps_te[i][order]
        preds = predict_sequences(model, X_shuffled, bfp_shuffled)
        drops.append(accuracy(preds, ys_te))
    return {
        'test': 'Frame-order permutation within sequence',
        'mean_acc': float(np.mean(drops)),
        'std_acc': float(np.std(drops)),
        'n_trials': n_trials,
    }


def test_bulk_doppler_mask(model, Xs_te, bfps_te, ys_te, mask_frac=0.25):
    """Mask (zero out) the central band of each spectrogram — bulk Doppler."""
    H = Xs_te.shape[-1]  # 128
    mid = H // 2
    half_w = int(H * mask_frac / 2)
    X_masked = Xs_te.copy()
    X_masked[..., mid - half_w:mid + half_w, :] = 0.0
    preds = predict_sequences(model, X_masked, bfps_te)
    return {
        'test': f'Bulk-Doppler mask (central {mask_frac*100:.0f}% freq band)',
        'mean_acc': float(accuracy(preds, ys_te)),
        'std_acc': 0.0,
        'n_trials': 1,
    }


def test_micro_doppler_mask(model, Xs_te, bfps_te, ys_te, mask_frac=0.50):
    """Mask (zero out) the outer bands — micro-Doppler sidebands."""
    H = Xs_te.shape[-1]  # 128
    mid = H // 2
    half_w = int(H * (1 - mask_frac) / 2)
    X_masked = Xs_te.copy()
    X_masked[..., :mid - half_w, :] = 0.0
    X_masked[..., mid + half_w:, :] = 0.0
    preds = predict_sequences(model, X_masked, bfps_te)
    return {
        'test': f'Micro-Doppler mask (outer {mask_frac*100:.0f}% freq band)',
        'mean_acc': float(accuracy(preds, ys_te)),
        'std_acc': 0.0,
        'n_trials': 1,
    }


def test_temporal_mask(model, Xs_te, bfps_te, ys_te, mask_frac=0.50):
    """Zero out half the time axis of each spectrogram."""
    W = Xs_te.shape[-2]  # 128
    half_w = int(W * mask_frac / 2)
    mid = W // 2
    X_masked = Xs_te.copy()
    X_masked[..., mid - half_w:mid + half_w] = 0.0
    preds = predict_sequences(model, X_masked, bfps_te)
    return {
        'test': f'Temporal mask (central {mask_frac*100:.0f}% time band)',
        'mean_acc': float(accuracy(preds, ys_te)),
        'std_acc': 0.0,
        'n_trials': 1,
    }


# ============================================================
# Main
# ============================================================

def main():
    model, Xs_te, bfps_te, ys_te, baseline_acc = build_model_and_test_set()

    print("\n" + "=" * 70)
    print("Running feature attribution tests...")
    print("=" * 70)

    tests = [
        ('bfp_permutation', test_bfp_permutation),
        ('spectrogram_permutation', test_spectrogram_permutation),
        ('frame_order_shuffle', test_frame_order_shuffle),
        ('bulk_doppler_mask', test_bulk_doppler_mask),
        ('micro_doppler_mask', test_micro_doppler_mask),
        ('temporal_mask', test_temporal_mask),
    ]

    results = {
        'baseline_clean_accuracy': baseline_acc,
        'tests': {},
    }

    for name, fn in tests:
        print(f"\n  {name}...")
        r = fn(model, Xs_te, bfps_te, ys_te)
        r['accuracy_drop_pp'] = (baseline_acc - r['mean_acc']) * 100
        r['relative_drop'] = (baseline_acc - r['mean_acc']) / (baseline_acc + 1e-9)
        results['tests'][name] = r
        print(f"    {r['test']}")
        print(f"    Accuracy: {r['mean_acc']:.4f} ± {r['std_acc']:.4f}  "
              f"(drop: {r['accuracy_drop_pp']:+.2f}pp, {r['relative_drop']*100:+.1f}%)")

    out_path = os.path.join(RESULTS_DIR, 'feature_attribution_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY — feature importance (larger drop = more important)")
    print("=" * 70)
    print(f"Baseline clean accuracy: {baseline_acc:.4f}")
    print()
    print(f"{'Test':<45} {'Masked Acc':<12} {'Drop (pp)':<12}")
    print("-" * 69)
    # Sort by drop magnitude (most important first)
    sorted_tests = sorted(results['tests'].items(),
                           key=lambda x: -x[1]['accuracy_drop_pp'])
    for name, r in sorted_tests:
        print(f"{r['test'][:44]:<45} {r['mean_acc']:<12.4f} "
              f"{r['accuracy_drop_pp']:+.2f}")


if __name__ == '__main__':
    main()
