# Attack A1: Pure RPM Reduction — Findings

## TL;DR

A-series completion test. **Null result**, as predicted by the A2 BFP-noise ablation: changing physical blade-flash frequency by sweeping RPM from 6000 down to 500 (expected BFP 200 Hz → 16.7 Hz) does not move the classifier. The autocorrelation BFP extractor measures 33–44 Hz on every sample regardless of the ground truth, confirming that BFP is numerically noise and the classifier's predictions are independent of the BFP physics the attack modifies.

This isolates the RPM axis from A2's joint blade-count + RPM sweep, and produces the same null with the same explanation.

## Setup

A1 holds the drone at standard configuration (`n_blades=2`, `n_props=4`, drone-typical bulk velocity 5–20 m/s, blade length 0.10–0.15 m, tilt 30–60°) and varies only RPM. Expected blade-flash periodicity (BFP) frequency = `n_blades × rpm / 60`.

- Trained baseline: CNN+LSTM+BFP at SNR=15 dB, 4 classes, 300 samples/class
- Clean-test accuracy: 88.9%
- 150 attack samples per RPM, seed 42

## Results

| RPM    | Expected BFP (Hz) | Measured BFP (Hz)  | Drone accuracy | Dominant class       |
|-------:|------------------:|-------------------:|---------------:|:---------------------|
| 6000   | 200.0             | 43.4 ± –           | 0.773          | Enemy Drone (116)    |
| 5000   | 166.7             | 35.9 ± –           | 0.933          | Enemy Drone (140)    |
| 4000   | 133.3             | 33.2 ± –           | 0.773          | Enemy Drone (116)    |
| 3000   | 100.0             | 43.1 ± –           | 0.860          | Enemy Drone (129)    |
| 2000   | 66.7              | 39.9 ± –           | 0.760          | Enemy Drone (114)    |
| 1500   | 50.0              | 43.6 ± –           | 0.800          | Enemy Drone (120)    |
| 1000   | 33.3              | 34.4 ± 56.2        | 0.833          | Enemy Drone (125)    |
| 500    | 16.7              | 34.8 ± 52.2        | 0.840          | Enemy Drone (126)    |

Drone-classification accuracy stays in the 76–93% band — within sampling noise of the 88.9% clean baseline — across all eight RPM levels.

## Why it fails (already known from A2)

The autocorrelation-based BFP extractor in `extract_bfp_features` returns a stable 33–44 Hz cluster *regardless of the physical ground truth*. Expected BFP varies 12× across the sweep (16.7 → 200 Hz); measured BFP varies less than 30%. The measurement is dominated by noise structure in the spectrogram envelope, not by blade-flash periodicity.

A2's ablation already showed this on a wider parameter range. A1 confirms it cleanly on the RPM axis alone:

- Expected BFP and measured BFP have no monotonic relationship.
- The classifier's predictions track measured BFP only as a class-correlated noise distribution, not as a physical quantity (per `FINDINGS_attribution.md`).

A1 therefore *cannot* defeat the classifier through the BFP feature pathway, because the BFP feature is numerically inert with respect to the physics A1 modifies. The attack hits the wrong abstraction.

## Position in the threat taxonomy

A2 established that joint blade-count + RPM manipulation is a null. A1 isolates the RPM axis. The combination shows that the entire A-series — any attack that targets blade-flash physics — is on the wrong layer for this classifier on this synthetic dataset.

For the methodology paper, A1 reinforces the same point as A2: a null result on a physically motivated attack is not robustness evidence. It is evidence that the attack and the classifier are operating on different feature abstractions. Attribution-first evaluation (`feature_attribution.py` + `feature_attribution_class_conditional.py`) is the cheap diagnostic that distinguishes the two cases.

## Files

- `attack_a1_rpm_reduction.py` — implementation
- `attack_a1_results.json` — raw results
- `run_log_a1_e1.txt` — full stdout for A1 + E1 run
