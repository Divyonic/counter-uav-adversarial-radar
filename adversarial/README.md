# Adversarial experiments

Experiments probing the adversarial robustness of the baseline CNN + LSTM + BFP classifier defined in `../baseline/`. Three experiments were run; two attacks (both producing null results) and one attribution study (which explains the null results).

All experiments are self-contained. Each re-trains the baseline model from scratch using the synthetic simulator in `../baseline/fmcw_simulation.py`. Random seeds are pinned.

See the top-level preprint in `../paper/preprint.md` for the full methodology argument.

---

## Experiments in this directory

### A2: blade-count reduction

**Script:** `attack_a2_fewer_blades.py`
**Write-up:** `FINDINGS_A2.md`
**Raw results:** `attack_a2_results.json`, `attack_a2_results_seed42.json`, `attack_a2_results_seed123.json`
**Run logs:** `run_log_seed42.txt`, `run_log_seed123.txt`

Six variants spanning the gradient from "clean 2-blade control" to "1-blade, 800 RPM, BFP at typical bird-flap frequency." Hypothesis: pushing the blade-flash fundamental into the bird band should destroy BFP as a discriminator and force bird misclassification.

**Outcome, null.** Drone-class accuracy stays within 83.3%–89.3% across all six variants, indistinguishable from the unmasked baseline. Separate diagnostic shows the BFP extractor returns 45 ± 59 Hz on clean drone data regardless of ground-truth physics (13–167 Hz), the BFP feature is numerically noise, so an attack that modifies the physics underneath it has no measurable effect.

### D2: pulse-and-glide flight

**Script:** `attack_d2_pulse_glide.py`
**Write-up:** `FINDINGS_D2.md`
**Raw results:** `attack_d2_results.json`
**Run log:** `run_log_d2.txt`

Nine variants, glide ratio from 0 (every frame has propeller content) to 1.0 (every frame in every 10-frame LSTM window is glide-only, body echo only, no propeller content anywhere in the sequence). Frame order randomised within each sequence to model realistic pulse/glide alternation.

**Outcome, null.** Drone-class accuracy stays within 80.0%–89.3% across all glide ratios. Most striking: at glide ratio 1.0 (no propeller content in any frame), the classifier still returns drone 81.3% of the time.

This is a stronger null than A2 because it removes propeller content from the *signal*, not just from the extracted feature. It cannot be explained away as a noisy extractor. The classifier's drone decision does not depend on propeller content.

### Feature attribution

**Script:** `feature_attribution.py`
**Write-up:** `FINDINGS_attribution.md`
**Raw results:** `feature_attribution_results.json`
**Run log:** `run_log_feat_attr.txt`

Six perturbation tests on the trained classifier: spectrogram permutation across samples, BFP permutation across samples, frame-order permutation within each sequence, frequency-band masks (central 25%, outer 50%), and temporal masks (central 50% of time bins). Drops in accuracy measure each feature group's importance.

**Outcomes:**

| Perturbation                          | Drop (pp) | Interpretation                                          |
|:--------------------------------------|:---------:|:--------------------------------------------------------|
| Frame-order permutation               | −1.6      | LSTM is multi-instance, not temporal                    |
| Temporal mask (central 50% time)      | +0.0      | In-frame time axis is redundant                         |
| Mask outer 50% frequency bins         | +33.3     | Frequency region holding bulk-Doppler is load-bearing   |
| Mask central 25% frequency bins       | +16.0     | Near-zero-Doppler region matters less                   |
| BFP permutation across samples        | +37.9     | BFP used as class-correlated noise distribution         |
| Spectrogram permutation across samples| +41.2     | Spectrogram content is important overall                |

The attribution resolves A2 and D2. The classifier identifies drones by the position and amplitude of the bulk-Doppler peak, which both A2 and D2 leave intact. BFP is used, but as a class-correlated distributional fingerprint rather than as a physics measurement. Blade-flash harmonic structure and propeller micro-Doppler are not read by the classifier at any stage.

---

## Experiments proposed but not run

Attribution predicts that attacks targeting what the classifier actually *does* use should succeed.

- **B1, radar-absorbent material (RAM) wrap.** Reduces `rcs_body` by ~10 dB; predicted to push drone classification toward bird or below detection threshold.
- **D1, bird-speed flight.** Flies the drone at 5–10 m/s instead of 10–20 m/s; predicted to push drone bulk-Doppler into the bird class distribution.
- **Class-conditional mask.** Zeroes out ±N bins around each sample's own bulk-Doppler peak; a cleaner version of the frequency-band masks in the attribution study.

These are straightforward to implement using the same simulator and would close the "demonstrate one successful attack" gap in the preprint. They are not implemented in this repository.

---

## Reading order

For someone new to the repository:

1. Skim `../paper/preprint.md` (the main argument).
2. Read `FINDINGS_A2.md` and `FINDINGS_D2.md` (the two null results in order).
3. Read `FINDINGS_attribution.md` (what resolves the nulls).
4. Run `feature_attribution.py` to reproduce the attribution numbers.
