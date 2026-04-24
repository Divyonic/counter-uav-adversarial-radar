# Attack A2: Fewer-Blade Drones: Findings

## TL;DR
**A2 does not defeat the baseline classifier.** Across two independent runs (seed=42, n=100 and seed=123, n=200), the classifier achieves 78–91% accuracy against adversarial 1-blade drones at RPMs from 800 to 5000, with no systematic degradation compared to the clean 2-blade baseline. The attack fails.

**But the attack's failure reveals something more useful:** the BFP feature the attack was designed to exploit is not doing discriminative work in the first place. The classifier reaches ~88% accuracy despite its "novel physics-informed feature" being noise.

## Experimental setup
- Baseline: CNN + LSTM + BFP classifier trained on 4-class synthetic FMCW data at SNR = 15 dB (parameters in `../baseline/`)
- Attack: generate adversarial drones with n_blades=1 (vs. training set n_blades=2) at progressively lower RPMs to push the expected blade-flash-periodicity (BFP) into the bird range
- Evaluation: per-variant accuracy (classified as drone = correct), 10-frame sequences, 100 or 200 sequences per variant

## Results across two runs

| Variant                     | Expected BFP | Run 1 acc (seed=42, n=100) | Run 2 acc (seed=123, n=200) |
|-----------------------------|:------------:|:--------------------------:|:---------------------------:|
| clean_drone_control (baseline) | 167 Hz    | 82.0%                      | 78.0%                       |
| A2_pure_1blade (5000 RPM)   | 83 Hz        | 87.0%                      | 81.0%                       |
| A2+A1_mild (3000 RPM)       | 50 Hz        | 91.0%                      | 81.5%                       |
| A2+A1_aggressive (2000 RPM) | 33 Hz        | 79.0%                      | 85.5%                       |
| A2+A1_extreme (1200 RPM)    | 20 Hz        | 89.0%                      | 82.5%                       |
| A2+A1_bird_mimic (800 RPM)  | 13 Hz        | 85.0%                      | 85.0%                       |

The "bird_mimic" variant (expected BFP in typical small-bird flap range) achieves accuracy equal to or higher than the baseline control in both runs.

## Why the attack fails: BFP is noise

In both runs, the measured BFP frequency is essentially uncorrelated with the expected value:

| Expected BFP | Measured BFP (seed=42) | Measured BFP (seed=123) |
|:---:|:---:|:---:|
| 167 Hz | 45.1 ± 59.5 Hz | 44.4 ± 61.8 Hz |
| 83 Hz  | 35.0 ± 52.2 Hz | 40.8 ± 56.1 Hz |
| 50 Hz  | 42.0 ± 53.6 Hz | 38.4 ± 56.6 Hz |
| 33 Hz  | 32.1 ± 54.3 Hz | 35.5 ± 50.7 Hz |
| 20 Hz  | 33.6 ± 57.6 Hz | 42.3 ± 55.6 Hz |
| 13 Hz  | 45.1 ± 58.9 Hz | 32.0 ± 49.1 Hz |

Observations:
1. **Expected values span a 12× range (13 → 167 Hz). Measured values all fall within a narrow 32–45 Hz band with ~55 Hz standard deviation.** The extractor produces almost the same statistical output regardless of the physical ground truth.
2. **BFP confidence sits at ~0.09 across all variants.** The paper defines `C_BFP > 0.3` as the threshold for "drone indication." Nothing clears the threshold.
3. **Standard deviation exceeds the mean in every row.** By any reasonable criterion this is noise.

## Interpretation

The paper's "novel physics-informed Blade Flash Periodicity feature" does not discriminate between drones, birds, fixed-wing UAVs, or manned aircraft in this implementation. The CNN+LSTM reaches ~88% accuracy in spite of BFP, using other features (bulk Doppler velocity, spectrogram texture, multi-instance parameter diversity).

This matches earlier signals from the baseline paper:
- CNN+BFP single-frame accuracy (47.8%) was *worse* than CNN-only (48.9%)
- CNN+BFP FAR (13.5%) was *worse* than CNN-only FAR (2.4%)
- The paper itself notes in §6.4 that BFP "introduces noise without adding discriminative value in the single-frame regime"

What the current work adds: **the same conclusion holds under adversarial conditions.** An attacker who explicitly targets BFP gets no advantage because BFP is inert.

## Implication for attack design

Attacking a feature the classifier does not use is methodologically pointless. Before proposing attacks, it is necessary to identify which features actually carry discriminative information. For this classifier, the load-bearing features are:

- **Bulk Doppler velocity** (determined by target radial speed, training-distribution-specific)
- **Spectrogram texture / overall shape** (CNN feature extraction from the log-scale time-frequency representation)
- **Multi-instance aggregation by the LSTM** over parameter-diverse frames (per the leakage test in the baseline paper)

Next attacks should target these. Candidates:
- **B1 (RAM wrap)**, reduces bulk RCS, affecting amplitude and possibly bulk Doppler detection
- **D1 (bird-mimicking flight)**, modifies bulk Doppler profile and between-frame parameter statistics
- **D2 (pulse-and-glide)**, floods the 10-frame LSTM window with zero-signal frames, disrupting multi-instance aggregation
- **E1 (ornithopter)**, rewrites the entire micro-Doppler signature, not just BFP

## Methodology contribution

The broader takeaway: adversarial evaluation of ML systems needs a feature-attribution step before attack design. Without it, attacks can spend compute on features the classifier ignores. For counter-UAV radar specifically, this suggests that published "physics-informed feature" contributions should be evaluated against an ablation baseline (is the classifier using the feature, or just including it?) before any robustness claim is made.

## Files

- `attack_a2_fewer_blades.py`, attack implementation (accepts `ATTACK_SEED` and `ATTACK_N_SAMPLES` env vars)
- `attack_a2_results_seed42.json`, run 1 (n=100)
- `attack_a2_results_seed123.json`, run 2 (n=200)
- `run_log_seed42.txt`, `run_log_seed123.txt`, full stdout from each run
