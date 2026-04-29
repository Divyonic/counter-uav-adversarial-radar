"""
Class-Conditional Bulk-Doppler Mask
====================================
The original feature_attribution.py mask tests are confounded: the
"central 25% freq band" zeros the bird/slow-drone region, the "outer
50%" zeros the aircraft region, but neither cleanly removes "bulk
Doppler" since the bulk-Doppler peak sits in a different band per class.

This script masks ±N bins around *each sample's own* bulk-Doppler peak,
disentangling "bulk Doppler peak position+amplitude" from "everything
else in the spectrogram."

Hypothesis (from FINDINGS_attribution): the classifier is reading bulk-
Doppler peak position+amplitude. A class-conditional mask on the peak
should produce a larger drop than either fixed-band mask.
"""

import numpy as np
import torch
import sys, os, json
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'baseline'))

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


def build_model_and_test_set():
    print("=" * 70)
    print("Training baseline CNN+LSTM+BFP")
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


def find_peak_freq_bin(spec_2d):
    """Return the frequency bin (axis=0) with maximum power summed over time."""
    return int(np.argmax(np.sum(spec_2d, axis=1)))


def class_conditional_mask(Xs, half_width):
    """For each frame in each sequence, zero out ±half_width freq bins
    centred on that frame's peak-power frequency bin."""
    Xm = Xs.copy()
    n_seq, seq_len, H, W = Xm.shape
    for i in range(n_seq):
        for k in range(seq_len):
            peak = find_peak_freq_bin(Xm[i, k])
            lo = max(0, peak - half_width)
            hi = min(H, peak + half_width + 1)
            Xm[i, k, lo:hi, :] = 0.0
    return Xm


def test_class_conditional(model, Xs_te, bfps_te, ys_te, half_width):
    Xm = class_conditional_mask(Xs_te, half_width)
    preds = predict_sequences(model, Xm, bfps_te)
    return {
        'test': f'Class-conditional bulk-Doppler mask (±{half_width} bins around peak)',
        'half_width_bins': int(half_width),
        'mean_acc': float(accuracy(preds, ys_te)),
    }


def per_class_breakdown(model, Xs, bfps, ys):
    preds = predict_sequences(model, Xs, bfps)
    out = {}
    for c in range(4):
        mask = ys == c
        if mask.sum() == 0:
            out[CLASS_NAMES[c]] = None
            continue
        acc = float((preds[mask] == ys[mask]).mean())
        confusion = {CLASS_NAMES[k]: int((preds[mask] == k).sum()) for k in range(4)}
        out[CLASS_NAMES[c]] = {'recall': acc, 'confusion': confusion}
    return out


def main():
    model, Xs_te, bfps_te, ys_te, baseline_acc = build_model_and_test_set()
    torch.save(model.state_dict(),
               os.path.join(RESULTS_DIR, 'cnn_lstm_bfp_baseline_classcond.pt'))

    half_widths = [1, 2, 4, 8, 16]

    print("\n" + "=" * 70)
    print("Class-conditional bulk-Doppler mask, sweeping half-width")
    print("=" * 70)

    results = {
        'baseline_clean_accuracy': baseline_acc,
        'baseline_per_class': per_class_breakdown(model, Xs_te, bfps_te, ys_te),
        'tests': [],
    }

    for hw in half_widths:
        print(f"\n  Mask half-width = ±{hw} bins ({(2*hw+1)/128*100:.1f}% of freq axis)...")
        r = test_class_conditional(model, Xs_te, bfps_te, ys_te, hw)
        r['accuracy_drop_pp'] = (baseline_acc - r['mean_acc']) * 100
        # Per-class breakdown for this half-width
        Xm = class_conditional_mask(Xs_te, hw)
        r['per_class'] = per_class_breakdown(model, Xm, bfps_te, ys_te)
        results['tests'].append(r)
        print(f"    Accuracy: {r['mean_acc']:.4f}  (drop: {r['accuracy_drop_pp']:+.2f} pp)")
        for cls, info in r['per_class'].items():
            if info is not None:
                print(f"      {cls}: recall {info['recall']:.3f}, conf {info['confusion']}")

    out_path = os.path.join(RESULTS_DIR, 'feature_attribution_class_conditional_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")

    print("\n" + "=" * 70)
    print("SUMMARY (larger drop = more bulk-Doppler-peak dependence)")
    print("=" * 70)
    print(f"Baseline clean accuracy: {baseline_acc:.4f}")
    print(f"{'Mask ±bins':<15} {'%freq axis':<14} {'Masked Acc':<12} {'Drop (pp)':<12}")
    print("-" * 65)
    for r in results['tests']:
        pct = (2 * r['half_width_bins'] + 1) / 128 * 100
        print(f"{r['half_width_bins']:<15} {pct:<14.1f} "
              f"{r['mean_acc']:<12.4f} {r['accuracy_drop_pp']:+.2f}")


if __name__ == '__main__':
    main()
