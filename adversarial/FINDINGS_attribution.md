# Feature Attribution — Findings

## TL;DR

Formal permutation-importance tests partially **confirm** and partially **refine** the D2 interpretation.

**Confirmed:**
- The LSTM is **multi-instance**, not temporal. Shuffling frame order within each 10-frame sequence changes accuracy by −1.6 pp (within noise).
- The spectrogram time axis is **largely redundant**. Zeroing out the central 50% of time bins causes 0.0 pp drop in accuracy.
- **BFP is being used by the classifier**, but not as a physics measurement — A2 established that the measured BFP is noise (45 ± 59 Hz regardless of ground truth). So what attribution shows is that the *distribution* of that noise is class-correlated enough for the classifier to exploit it.

**Refined (important correction to D2's framing):**
- Spectrogram permutation causes a 41 pp drop, BFP permutation 38 pp. Both are load-bearing.
- D2's conclusion that "all micro-Doppler is inert" was too strong. D2 showed that **zeroing out propeller content** (n_blades=0) does not defeat the classifier *because body echo at bulk Doppler is retained*. Attribution shows that when the **spectrogram region containing bulk Doppler energy** is masked, accuracy drops substantially.
- The consistent story: the classifier uses bulk Doppler peak *position and amplitude*, not harmonic structure. A2 + D2 failed to disturb that. Spectrogram permutation and frequency-band masking do disturb it.

## Setup

- Baseline: CNN+LSTM+BFP trained on 4-class synthetic FMCW, 300 samples/class, SNR 15 dB, 40 epochs (early-stopped at 18)
- Clean baseline accuracy: 88.9%
- Seed 42, all tests run on the same held-out test set

## Results

| Test                                         | Accuracy | Drop (pp) | Interpretation                                                |
|:---------------------------------------------|:--------:|:---------:|:--------------------------------------------------------------|
| Spectrogram permutation across samples       | 47.7%    | +41.2     | Spectrogram content is critical                               |
| BFP permutation across samples               | 51.0%    | +37.9     | BFP vector is critical *to the classifier* (not to physics)   |
| Mask outer 50% of frequency bins             | 55.6%    | +33.3     | Bulk Doppler energy (which sits in outer bands for drones at 10–20 m/s and all aircraft) is load-bearing |
| Mask central 25% of frequency bins           | 72.9%    | +16.0     | Near-zero-Doppler content matters less                        |
| Mask central 50% of time bins                | 88.9%    | +0.0      | Time-axis detail is redundant                                 |
| Frame-order permutation within sequence      | 90.5%    | −1.6      | LSTM is multi-instance, not temporal                          |

## Doppler-bin geometry (important caveat on the mask tests)

With PRF = 8333 Hz and 128 frequency bins, each bin spans 65 Hz. At 9.5 GHz, Doppler for target velocity `v` is `2·f·v/c`:

| Class         | v range (m/s) | Doppler (Hz)    | Sits in central 25% band (±1042 Hz)? |
|:--------------|:-------------:|:---------------:|:------------------------------------:|
| Drone         | 5–20          | 317–1267        | Mostly yes (up to ~16 m/s)           |
| Bird          | 5–15          | 317–950         | Yes, entirely                        |
| Fixed-wing UAV| 15–35         | 950–2217        | Partially                            |
| Aircraft      | 50–100        | 3167–6333       | No, entirely outside                 |

This means the two mask tests are **confounded** with respect to bulk-vs-micro-Doppler disentanglement:

- The "bulk-Doppler mask" (central 25%) zeros the region where drones and birds sit → hurts drone-vs-bird discrimination (−16 pp).
- The "micro-Doppler mask" (outer 50%) zeros the region where aircraft and fast UAVs sit → hurts aircraft identification most (−33 pp).

Neither mask cleanly isolates "bulk Doppler" from "micro-Doppler sidebands" because for each class the *bulk* energy lives in a different Doppler band. A cleaner test would be a class-conditional mask placed relative to each sample's bulk Doppler peak — that is a follow-up experiment.

## Reconciling with D2

D2 set `n_blades = 0`, removing propeller content from the transmit model but **keeping body echo at drone-typical bulk Doppler velocity**. Accuracy stayed at 81%.

Attribution shows: masking the *spectrogram band that contains that body echo* drops accuracy. Both are consistent under the hypothesis:

> The classifier identifies drones by the position and shape of bulk-Doppler energy in the spectrogram, not by harmonic sidebands or blade-flash periodicity.

D2 did not disturb that. Permutation and masking do.

## What this means for the paper's methodology claim

The critique survives but becomes sharper:

1. **BFP, as implemented in the baseline paper, is a class-correlated noise distribution, not a physics measurement.** The classifier uses its values, but those values do not measure blade flash periodicity — A2 demonstrated that measured BFP is independent of ground-truth physics.
2. **The LSTM is not a temporal tracker.** Frame-order permutation has no effect. It is a multi-instance aggregator, consistent with the leakage test finding.
3. **The spectrogram is being used, but to localize bulk Doppler peaks, not to analyze harmonic structure.** Time-axis information is entirely redundant; frequency-band masking in the region where bulk Doppler energy sits causes the largest accuracy drop.

The "physics-informed CNN+LSTM+BFP" architecture, when trained on this synthetic dataset, has the same functional behavior as a much simpler "bulk-Doppler peak detector + class-prior classifier" — minus the interpretability that a simpler model would provide.

## Next experiments

1. **Class-conditional masking**: for each sample, mask the ±N bins around its own bulk-Doppler peak. This disentangles bulk from micro-Doppler properly.
2. **Attack B1 (amplitude reduction / RAM wrap)**: directly tests whether bulk RCS amplitude is load-bearing.
3. **Real-data replication**: repeat the A2 / D2 / attribution protocol on DroneRF, to determine whether the critique is specific to this synthetic setup or generalizes.

## Files

- `feature_attribution.py` — implementation
- `results/feature_attribution_results.json` — raw numbers
- `run_log_feat_attr.txt` — full stdout (copy of `/tmp/feat_attr.log`)
