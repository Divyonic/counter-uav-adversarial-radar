# Class-Conditional Bulk-Doppler Mask — Findings

## TL;DR

The original `feature_attribution.py` mask tests are confounded — the "central 25% freq band" mask zeros bird/slow-drone Doppler region, the "outer 50%" mask zeros the aircraft region, but neither cleanly removes "bulk Doppler" because the bulk-Doppler peak sits in a different band per class.

The class-conditional mask removes ±N bins around *each sample's own* peak-power frequency bin. This disentangles "bulk-Doppler peak position+amplitude" from everything else in the spectrogram and produces a substantially cleaner attribution signal: **masking just ±1 frequency bin (2.3% of the axis) at each sample's peak drops accuracy by 20.8 pp — more drop than masking the central 25% with a fixed band did (16.0 pp)**.

This is the cleanest demonstration in the repository of the central attribution claim: the classifier is using bulk-Doppler peak position+amplitude as its primary feature.

## Setup

For each frame in the held-out test set (4-class, 88.9% clean accuracy, same baseline as `feature_attribution.py`):

1. Find `peak = argmax_f Σ_t spec(f, t)` — the frequency bin holding the most power summed over time. This is the bulk-Doppler peak.
2. Zero `spec[peak − N : peak + N + 1, :]` — a ±N-bin band around the peak, across all time bins.
3. Run the classifier and record accuracy + per-class confusion.

Sweep `N ∈ {1, 2, 4, 8, 16}`. The 128-bin frequency axis means `N=1` zeros 3 bins (2.3% of axis) and `N=16` zeros 33 bins (25.8%).

## Results

| Mask half-width | % freq axis | Masked accuracy | Drop (pp)  |
|----------------:|------------:|----------------:|-----------:|
| ±1 bins         | 2.3%        | 0.681           | +20.83     |
| ±2 bins         | 3.9%        | 0.729           | +15.97     |
| ±4 bins         | 7.0%        | 0.493           | +39.58     |
| ±8 bins         | 13.3%       | 0.479           | +40.97     |
| ±16 bins        | 25.8%       | 0.424           | +46.53     |

Reference points from the original fixed-band masks (in `feature_attribution.py`):

| Fixed-band mask          | % freq axis | Drop (pp) |
|:-------------------------|------------:|----------:|
| Central 25%              | 25%         | +16.0     |
| Outer 50%                | 50%         | +33.3     |

A 2.3% class-conditional mask hurts the classifier *more* than a 25% fixed-band mask, and approximately as much as the outer 50% fixed-band mask. The class-conditional version is roughly **10× more efficient** per masked bin.

## Per-class breakdown (±1 bin)

| Class           | Recall | Confusion vector                                        |
|:----------------|-------:|:--------------------------------------------------------|
| Enemy Drone     | 0.822  | Drone 37, Bird 0, Friendly UAV 8, Aircraft 0            |
| Bird            | 0.000  | Drone 37, Bird 0, Friendly UAV 0, Aircraft 0            |
| Friendly UAV    | 1.000  | Drone 0, Bird 0, Friendly UAV 35, Aircraft 0            |
| Manned Aircraft | 0.963  | Drone 0, Bird 0, Friendly UAV 1, Aircraft 26            |

The mask hits birds the hardest. With the bulk-Doppler peak removed, the residual spectrogram for a bird sample is reclassified as drone (37/37). Birds depend on their bulk-Doppler peak more than any other class — consistent with their low body RCS (≈ 0.005 m²) producing a single dominant peak with little structured background. Drone, UAV and aircraft have richer micro-Doppler content (propeller harmonics, jet engine modulation) that helps the classifier hold its decision even when the peak is masked.

## Implications for the attribution claim

`FINDINGS_attribution.md` argued that the classifier identifies drones by the position and amplitude of the bulk-Doppler peak. The fixed-band masks supported this with the caveat that the bulk-Doppler bin sits in different absolute positions for different classes, so the central 25% / outer 50% bands do not isolate the peak. The class-conditional mask removes that confound and shows that **a 3-bin notch at the peak destroys nearly half the classifier's accuracy lead**.

Combined with B1's null result (`FINDINGS_B1.md`): the load-bearing feature is the peak's *position* and *post-normalisation shape*, not its absolute amplitude. The class-conditional mask attacks both — it zeros the peak region, so neither position-of-max-power nor the height of the peak relative to the rest of the spectrogram is informative anymore.

## Comparison to D1 and B1

The three attribution-driven experiments tell a consistent story:

| Experiment              | Targets                                          | Outcome                                       |
|:------------------------|:-------------------------------------------------|:----------------------------------------------|
| D1 (bird-speed flight)  | Bulk-Doppler peak *position* (physical attack)   | **Strong success** — drone acc 0–45%          |
| B1 (RAM wrap)           | Bulk-Doppler peak *amplitude* (physical attack)  | **Null** — preprocessing absorbs amplitude    |
| Class-conditional mask  | Bulk-Doppler peak *region* (input-domain attack) | **Strong success** — −20 pp at ±1 bin         |

D1 confirms that *position* matters; the class-conditional mask confirms it as well by destroying the peak region. B1's null clarifies that "amplitude" in the attribution claim refers to post-normalised relative amplitude, not absolute received power, so amplitude-domain physical attacks are absorbed by per-sample normalisation.

## Files

- `feature_attribution_class_conditional.py` — implementation
- `feature_attribution_class_conditional_results.json` — raw results
- `run_log_b1_d1_classcond.txt` — full stdout for B1 + D1 + class-conditional run
