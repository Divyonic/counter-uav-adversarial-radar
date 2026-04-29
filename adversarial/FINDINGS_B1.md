# Attack B1: RAM-wrap / Bulk-amplitude Reduction — Findings

## TL;DR

Attribution-predicted attack on bulk-Doppler peak amplitude. **Null result, but for an instructive reason: per-sample spectrogram normalisation in the preprocessing pipeline absorbs amplitude attacks before they reach the classifier.** This sharpens the attribution claim: the classifier reads bulk-Doppler peak *position* (load-bearing) and *relative shape* — not absolute amplitude.

## Setup

The B1 attack reduces the drone's effective radar return by `dB_drop` dB while leaving ambient noise unchanged. Implementation: pass `snr_db = base_snr − dB_drop` to `generate_drone_signal`. The simulator's `noise_power = sig_power / 10^(snr_db/10)` makes this mathematically equivalent to attenuating target backscatter against a fixed noise floor.

- Trained baseline: CNN+LSTM+BFP at SNR=15 dB, 4 classes, 300 samples/class
- Clean-test accuracy: 88.9%
- 150 attack samples per dB-drop level, seed 42

## Results

| RCS drop | Effective SNR | Drone-classification accuracy | Dominant class       |
|---------:|--------------:|------------------------------:|:---------------------|
| 0 dB     | 15 dB         | 0.900                         | Enemy Drone (135)    |
| 3 dB     | 12 dB         | 0.913                         | Enemy Drone (137)    |
| 6 dB     | 9 dB          | 0.827                         | Enemy Drone (124)    |
| 10 dB    | 5 dB          | 0.973                         | Enemy Drone (146)    |
| 15 dB    | 0 dB          | 0.947                         | Enemy Drone (142)    |
| 20 dB    | −5 dB         | 0.773                         | Enemy Drone (116)    |

Across the entire sweep, drone-classification accuracy stays within roughly ±10 pp of the clean baseline. Even at −5 dB SNR (target return below the noise floor on a per-chirp basis), the classifier still correctly labels 77% of attack samples as drone. The expected attribution-driven prediction (drone classified as bird or below detection threshold) does not occur.

## Why does the attack fail?

The spectrogram preprocessing in `compute_spectrogram` → `resize_spectrogram` does two normalisations that together neutralise amplitude attacks:

1. **dB clipping to a 40 dB dynamic range.** `resize_spectrogram` computes `spec_db = 10*log10(spec + 1e-12)` and then clips to `[max−40, max]`. This removes everything more than 40 dB below the per-sample peak, regardless of absolute power.
2. **Per-sample normalisation to [0, 1].** After clipping, the spectrogram is shifted and rescaled so that its minimum is 0 and its maximum is 1.

Together these discard absolute amplitude. What survives is the spectrogram's *shape* relative to its own peak. As long as the bulk-Doppler peak is still the largest peak (i.e. signal is above the noise floor at the peak), its position is preserved through preprocessing. At −5 dB SNR the noise floor approaches the peak, which is why accuracy starts to slip there, but bulk-Doppler localisation still survives in most samples.

The 10 dB and 15 dB rows showing accuracy *higher* than the 0 dB control are within sampling noise (n=150, ±~3 pp), not a real effect.

## Refining the attribution claim

`FINDINGS_attribution.md` stated:

> The classifier identifies drones by the position and amplitude of the bulk-Doppler peak.

B1 shows that "amplitude" here means *spectrogram amplitude after per-sample normalisation*, not pre-normalisation absolute power. The classifier reads:

- **Peak position** (which Doppler bin holds the maximum) — load-bearing.
- **Peak relative shape** (height of the peak relative to the rest of the post-normalised spectrogram, plus the spread/width) — load-bearing.
- **Absolute received power** — not load-bearing under per-sample normalisation.

This means amplitude-domain physical attacks (RAM wrap, signal attenuation, increased range) cannot defeat this classifier through the input-feature pathway. They can only defeat it by pushing the target below the system's *detection* threshold, which is a stage upstream of the classifier (CFAR / track confirmation), not a classifier-attribution problem.

## Methodology implication

The B1 null result is a different kind of null than A2/D2's nulls. A2 and D2 targeted features the classifier did not read at all (BFP physics, propeller harmonics). B1 targets a feature the classifier *does* read (bulk-Doppler peak), but a *facet* of that feature (absolute amplitude) that the preprocessing layer discards before the classifier sees it.

Practical consequence for the attribution-first workflow proposed in `paper/preprint.md`: feature attribution should be carried out **on the post-preprocessing input the classifier actually receives**, and physical attacks should be designed against features that survive preprocessing. For per-sample-normalised spectrogram pipelines, "reduce target RCS" is a CFAR/detection attack, not a classifier attack.

The attribution-driven attack that *does* defeat this classifier is D1 (bird-speed flight); see `FINDINGS_D1.md`.

## Files

- `attack_b1_ram_wrap.py` — implementation
- `attack_b1_results.json` — raw results
- `run_log_b1_d1_classcond.txt` — full stdout for B1 + D1 + class-conditional run
