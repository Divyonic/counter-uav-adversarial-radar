# Attack E1: Ornithopter Substitution — Findings

## TL;DR

Five variants of a custom signal generator combining drone-scale body RCS and drone-scale bulk velocity with bird-style asymmetric flapping-wing micro-Doppler. **Four out of five variants produce null results (drone classified correctly at 83–91%); the fifth — which restricts bulk velocity to the bird-overlap range 5–15 m/s — drops drone accuracy to 32% and reclassifies most samples as bird.**

The conclusion is exactly the attribution prediction: the classifier does not read flap-vs-prop micro-Doppler structure. Substituting bird-style wing modulation for propeller modulation has no effect when bulk-Doppler peak position is held in the drone band. The only variant that succeeds collapses back to D1 — it is a velocity attack, not a micro-Doppler attack.

## Setup

A custom `generate_ornithopter_signal` was added in `attack_e1_ornithopter.py`. It builds the signal with:
- Body amplitude `√rcs_body × 1e3` (configurable, defaults 0.01 → drone-scale)
- Drone-scale bulk-Doppler position from `v_bulk` (configurable per variant)
- Bird-style sinusoidal body oscillation (40% amplitude modulation at flap frequency)
- Bird-style asymmetric wing micro-Doppler (downstroke 2× stronger, second-harmonic wing-tip return)
- No propeller blades

Five variants test specific factor combinations.

- Trained baseline: CNN+LSTM+BFP at SNR=15 dB, 4 classes, 300 samples/class
- Clean-test accuracy: 88.9%
- 150 attack samples per variant, seed 42

## Results

| Variant                  | Body RCS | v_bulk (m/s) | Flap (Hz) | Drone accuracy | Dominant class       |
|:-------------------------|---------:|:-------------|----------:|---------------:|:---------------------|
| `E1_ornithopter_std`     | 0.01     | 5–20         | 8.0       | 0.907          | Enemy Drone (136)    |
| `E1_low_flap`            | 0.01     | 5–20         | 3.0       | 0.893          | Enemy Drone (134)    |
| `E1_high_flap`           | 0.01     | 5–20         | 15.0      | 0.833          | Enemy Drone (125)    |
| `E1_birdRCS_droneV`      | 0.005    | 5–20         | 8.0       | 0.880          | Enemy Drone (132)    |
| `E1_droneRCS_birdV`      | 0.01     | 5–15         | 8.0       | 0.320          | Bird (101)           |

The first four variants stay within 6 pp of the 88.9% clean baseline — null results. The fifth variant succeeds, but for reasons unrelated to the ornithopter substitution.

## What the four nulls confirm

`E1_ornithopter_std` substitutes bird-style flap modulation for propeller blades while holding everything else (body RCS, v_bulk distribution) at drone-typical values. Drone-classification accuracy: 90.7%. The classifier does not notice the substitution.

`E1_low_flap` and `E1_high_flap` push flap frequency to 3 Hz and 15 Hz respectively — far from the propeller-typical micro-Doppler band — and accuracy moves only 5–6 pp.

`E1_birdRCS_droneV` further cuts body RCS from drone-scale (0.01) to bird-scale (0.005) on top of the bird-flap modulation. Per-sample normalisation (see `FINDINGS_B1.md`) absorbs the RCS change, and drone classification stays at 88%.

These four nulls jointly confirm: **the classifier is not reading micro-Doppler structure (flap vs prop), nor is it reading absolute body amplitude.** It reads bulk-Doppler peak position. Any signal whose bulk-Doppler peak sits in the drone band (5–20 m/s → ~317–1267 Hz) is classified as drone, regardless of what micro-Doppler content is overlaid.

## What the one success says

`E1_droneRCS_birdV` keeps the bird-flap micro-Doppler but restricts v_bulk to 5–15 m/s. Drone classification drops to 32%, with 101/150 samples reclassified as bird. This matches D1 closely: D1 at v_bulk = 5–10 m/s yielded 45% drone, at 5–8 m/s yielded 29%. The success is driven by the velocity restriction, not by the bird-flap content.

If the ornithopter substitution were doing the work, all four `5–20 m/s` variants would have been (partial) successes. They are not.

## Implication for the architecture

The architecture's advertised value is "physics-informed" reasoning over propeller blade-flash periodicity and temporal micro-Doppler structure. E1 confirms — alongside A1, A2, and D2 — that the classifier ignores both. The architecture's actual functional behaviour reduces to:

> Locate the bulk-Doppler peak. Read off its position. Classify by which class's training v_bulk distribution that position is most consistent with.

E1 closes a remaining gap in the attribution evidence: it shows that even an exotic (and physically distinct) micro-Doppler signature is invisible to the classifier. There is no escape hatch where a sufficiently novel micro-Doppler pattern would force the classifier to use the architectural machinery it advertises.

## Files

- `attack_e1_ornithopter.py` — implementation, including custom `generate_ornithopter_signal`
- `attack_e1_results.json` — raw results
- `run_log_a1_e1.txt` — full stdout for A1 + E1 run
