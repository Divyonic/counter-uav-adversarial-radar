# Adversarial experiments

This directory contains the three adversarial experiments that drive the preprint's methodology argument. Each experiment is self-contained, each re-trains the baseline classifier from scratch using [`../baseline/`](../baseline/), and each writes both a raw results JSON and a findings write-up.

For the full narrative see [`../paper/preprint.md`](../paper/preprint.md). For the attribution-first workflow argument see the top-level [README](../README.md).

---

## Experiments

### A2 &mdash; blade-count reduction

| Item | Value |
|:-----|:------|
| **Script** | [`attack_a2_fewer_blades.py`](attack_a2_fewer_blades.py) |
| **Write-up** | [`FINDINGS_A2.md`](FINDINGS_A2.md) |
| **Raw results** | [`attack_a2_results.json`](attack_a2_results.json) |
| **Outcome** | Null. Accuracy stays in the 83-89% band across six variants. |

Six variants spanning the gradient from "clean 2-blade control" to "1-blade at 800 RPM, BFP at typical bird-flap frequency." Hypothesis was that pushing the blade-flash fundamental into the bird-wingbeat band would destroy BFP as a discriminator.

Ablation on the BFP feature itself shows the autocorrelation extractor returns 45 &plusmn; 59 Hz on clean drone data regardless of the physical ground truth (13-167 Hz). The "physics-informed feature" is numerically noise; any attack that modifies only the physics underneath it has no measurable effect.

### D2 &mdash; pulse-and-glide flight

| Item | Value |
|:-----|:------|
| **Script** | [`attack_d2_pulse_glide.py`](attack_d2_pulse_glide.py) |
| **Write-up** | [`FINDINGS_D2.md`](FINDINGS_D2.md) |
| **Raw results** | [`attack_d2_results.json`](attack_d2_results.json) |
| **Run log** | [`run_log_d2.txt`](run_log_d2.txt) |
| **Outcome** | Null. At 100% glide (no propeller content anywhere in the sequence), the classifier still labels it drone 81.3% of the time. |

Nine variants, glide ratio from 0 (every frame has propeller content) to 1.0 (every frame in every 10-frame LSTM window is glide-only, body-echo only). Frame order randomised within each sequence to model realistic pulse-glide alternation.

This is a stronger null than A2. A2 could be explained by a noisy BFP extractor still driving a real feature downstream. D2 removes propeller content from the signal itself, not just from the extracted feature. The classifier's drone decision does not depend on propeller content at any level.

### Feature attribution

| Item | Value |
|:-----|:------|
| **Script** | [`feature_attribution.py`](feature_attribution.py) |
| **Write-up** | [`FINDINGS_attribution.md`](FINDINGS_attribution.md) |
| **Raw results** | [`feature_attribution_results.json`](feature_attribution_results.json) |
| **Run log** | [`run_log_feat_attr.txt`](run_log_feat_attr.txt) |

Six perturbation tests on the trained classifier: spectrogram permutation across samples, BFP permutation across samples, frame-order permutation within each sequence, frequency-band masks (central 25% and outer 50%), and a temporal mask (central 50% of time bins). Drops in accuracy measure each feature group's importance.

| Perturbation                            | Drop (pp) | Interpretation                                          |
|:----------------------------------------|:---------:|:--------------------------------------------------------|
| Frame-order permutation                 | &minus;1.6 | LSTM is multi-instance, not temporal                    |
| Temporal mask (central 50% time)        | +0.0      | In-frame time axis is redundant                         |
| Mask outer 50% frequency bins           | +33.3     | Frequency region holding bulk-Doppler is load-bearing   |
| Mask central 25% frequency bins         | +16.0     | Near-zero-Doppler region matters less                   |
| BFP permutation across samples          | +37.9     | BFP used as class-correlated noise distribution         |
| Spectrogram permutation across samples  | +41.2     | Spectrogram content is important overall                |

The attribution resolves A2 and D2. The classifier identifies drones by the position and amplitude of the bulk-Doppler peak, which both A2 and D2 leave intact. BFP is used, but as a class-correlated distributional fingerprint rather than as a physics measurement. Blade-flash harmonic structure and propeller micro-Doppler are not read by the classifier at any stage.

---

## Experiments proposed but not run

Attribution predicts that attacks against what the classifier *does* use should succeed. These are not implemented in this repository.

- **B1, radar-absorbent material (RAM) wrap.** Reduces `rcs_body` by ~10 dB; predicted to push drone classification toward bird or below detection threshold.
- **D1, bird-speed flight.** Flies the drone at 5-10 m/s instead of 10-20 m/s; predicted to push drone bulk-Doppler into the bird class distribution.
- **Class-conditional mask.** Zeroes ~N bins around each sample's own bulk-Doppler peak; a cleaner version of the frequency-band masks.

---

## Reading order

For a new reader:

1. Skim the preprint: [`../paper/preprint.md`](../paper/preprint.md).
2. Read [`FINDINGS_A2.md`](FINDINGS_A2.md) and [`FINDINGS_D2.md`](FINDINGS_D2.md), in order.
3. Read [`FINDINGS_attribution.md`](FINDINGS_attribution.md), which resolves the nulls.
4. Run `python3 feature_attribution.py` to reproduce the attribution numbers.
