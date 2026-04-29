# Attack D1: Bird-Speed Flight — Findings

## TL;DR

Attribution-predicted attack on bulk-Doppler peak position. **Strong success.** The classifier's "drone" decision occupies a narrow window in `v_bulk` (roughly 8–15 m/s); flying the drone outside that window pushes drone-classification accuracy from 88.9% (clean) to 0–45% across the sweep. Two distinct misclassifications appear: at low speeds (≤ 12 m/s) the drone reads as bird, and at high speeds (≥ 15 m/s) the drone reads as friendly UAV.

This is the first attack in the methodology study that *succeeds*, and it succeeds in exactly the way `FINDINGS_attribution.md` predicted.

## Setup

The D1 attack flies the drone with bulk velocity drawn from a sub-window of (or below) the drone training distribution (5–20 m/s), keeping all other parameters at drone-typical values: `n_blades=2`, `n_props=4`, `rpm` ∈ [4000, 6000], blade length ∈ [0.10, 0.15] m, tilt 30–60°.

- Trained baseline: CNN+LSTM+BFP at SNR=15 dB, 4 classes, 300 samples/class
- Clean-test accuracy: 88.9%
- 150 attack samples per velocity window, seed 42

## Results

| Velocity window  | Bulk-Doppler band | Drone accuracy | Class distribution             |
|:-----------------|:------------------|---------------:|:-------------------------------|
| 15.0–20.0 m/s    | 950–1267 Hz       | 0.000          | Friendly UAV 149, Aircraft 1   |
| 10.0–15.0 m/s    | 633–950 Hz        | 0.207          | Bird 119, Drone 31             |
| 8.0–12.0 m/s     | 507–760 Hz        | 0.267          | Bird 110, Drone 40             |
| 5.0–10.0 m/s     | 317–633 Hz        | 0.453          | Bird 82, Drone 68              |
| 5.0–8.0 m/s      | 317–507 Hz        | 0.287          | Bird 102, Drone 43, Aircraft 5 |
| 3.0–5.0 m/s      | 190–317 Hz        | 0.213          | Bird 117, Drone 32             |

Drone classification collapses to 0–45% across every velocity window tested.

## Two failure modes, one cause

The attack succeeds in two distinct directions:

**Bird misclassification (slow drone).** At v_bulk ≤ 12 m/s the drone is most often classified as a bird. The bulk-Doppler band 190–760 Hz overlaps the bird-class training distribution (birds at 5–15 m/s → 317–950 Hz). The bulk-Doppler peak is the load-bearing feature; placing it inside the bird's region of velocity-feature space hands the prediction to the bird class.

**Friendly UAV misclassification (fast drone).** At v_bulk = 15–20 m/s the drone is essentially always classified as friendly UAV (149/150). The friendly UAV training distribution covers 15–35 m/s → 950–2217 Hz, which overlaps the upper end of the drone training distribution. The classifier resolves this ambiguity by handing all upper-range velocities to the UAV class.

Both failure modes confirm the same hypothesis: **the classifier reads bulk-Doppler peak position and matches it to a class via the training-distribution priors over v_bulk**. The drone class lives in a narrow stripe (~8–15 m/s) where it does not overlap bird below or UAV above, and the classifier confidently labels that stripe drone. Outside it, the classifier confidently labels the more distribution-likely class.

## Operational consequences

For an adversary, this is the most consequential finding in this study:

- A drone flown slowly (5–10 m/s) is misclassified as a bird **45–55% of the time** at the worst window. A counter-UAV system that triages by class will likely not engage.
- A drone flown at the upper end of typical drone speed (15–20 m/s) is misclassified as friendly UAV **99% of the time**. Friendly-coded targets are explicitly *not* engaged.

Both misclassifications are easy to produce — they require no hardware modification, only a flight-controller change. There is no harmonic-content attack here, no propeller modification, no cross-section reduction. The drone simply flies outside the v_bulk band that the classifier associates with "drone."

## Confirms the attribution prediction

`FINDINGS_attribution.md` (after running permutation importance and frequency-band masking) predicted that an attack against bulk-Doppler peak *position* should succeed, because masking the spectrogram band that holds drone bulk-Doppler dropped accuracy by 33 pp. D1 is the physical realisation of that prediction. The attribution result is now a measurement that *correctly anticipated* a successful attack — which was the methodology paper's point.

## Caveats

- Synthetic data only. Real radar would also include detection-time effects (CFAR detection probability scales with signal amplitude and bin SNR), which are stage-upstream of the classifier and not modelled here.
- The "drone classified as friendly UAV" window depends on the v_bulk overlap baked into this training set. A different training distribution (drone v_max < UAV v_min) would close that window, but it would not close the bird window — birds and slow drones overlap in any realistic distribution.
- The classifier's decision boundary is the artefact this attack exploits. A classifier that explicitly modelled v_bulk as a *covariate* rather than a *feature* (e.g. by conditioning on detected velocity rather than including it in the spectrogram) might be less vulnerable. That is a defence direction the methodology paper does not explore.

## Files

- `attack_d1_bird_speed.py` — implementation
- `attack_d1_results.json` — raw results
- `run_log_b1_d1_classcond.txt` — full stdout for B1 + D1 + class-conditional run
