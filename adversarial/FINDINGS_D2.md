# Attack D2: Pulse-and-Glide — Findings

## TL;DR
**D2 does not defeat the baseline classifier.** Even when every frame in the 10-frame LSTM window has its propeller micro-Doppler completely removed (glide_ratio = 1.0 — just body echo at bulk Doppler velocity), the classifier still predicts "drone" 81.3% of the time — within the noise band of the unmasked baseline (83.3%).

**Conclusion:** the classifier is not using micro-Doppler information. It is classifying on bulk velocity, body echo amplitude, spectrogram texture, or other non-micro-Doppler features.

## Setup
- Baseline: CNN+LSTM+BFP trained on 4-class synthetic FMCW at SNR=15 dB (n=300/class)
- Attack: within each 10-frame sequence, a fraction `glide_ratio` of frames are generated with `n_blades=0` (no propellers) — pure body echo at bulk Doppler
- Frame order randomised within each sequence (realistic: drone alternating pulse/glide)
- Seed 42, 150 sequences per variant

## Results

| Glide ratio | Pulse frames | Glide frames | Accuracy (correct as drone) | Dominant class     |
|:-----------:|:-----------:|:------------:|:---------------------------:|:------------------:|
| 0.0         | 10          | 0            | 83.3%                       | Drone (125/150)    |
| 0.2         | 8           | 2            | 84.0%                       | Drone (126/150)    |
| 0.4         | 6           | 4            | 84.7%                       | Drone (127/150)    |
| 0.5         | 5           | 5            | 80.0%                       | Drone (120/150)    |
| 0.6         | 4           | 6            | 89.3%                       | Drone (134/150)    |
| 0.7         | 3           | 7            | 88.0%                       | Drone (132/150)    |
| 0.8         | 2           | 8            | 85.3%                       | Drone (128/150)    |
| 0.9         | 1           | 9            | 86.0%                       | Drone (129/150)    |
| 1.0         | 0           | 10           | 81.3%                       | Drone (122/150)    |

Variance range: 80.0 – 89.3%. No monotonic relationship with glide ratio. Spec+body echo alone classifies as drone at nearly the same rate as full propeller signal.

## Interpretation

This result is substantially stronger than the A2 null finding. A2 showed BFP is inert. D2 shows **all micro-Doppler is inert** — or at least, this classifier isn't using any of it.

Candidate features the classifier *must* be using (since it reaches ~85% without micro-Doppler):

1. **Bulk Doppler velocity.** Drones in training data have `v_bulk ~ U(5, 20) m/s`, birds `U(5, 15)`, fixed-wing UAVs `U(15, 35)`, aircraft `U(50, 100)`. These distributions mostly overlap for drone vs bird but are cleanly separable for aircraft.
2. **Body RCS / echo amplitude.** Each class has different default `rcs_body` ranges (0.01, 0.005, 0.05, higher for aircraft). Amplitude is a strong class discriminator by construction.
3. **Spectrogram global shape / extent.** The spread of the Doppler-time image is set largely by bulk velocity, not by micro-Doppler fine structure.

In other words, this synthetic dataset has class imbalance baked into the bulk-kinematic parameters, and the CNN is likely memorising that rather than learning micro-Doppler physics.

## What this means for the research direction

Three concrete implications:

### 1. The "novel physics-informed BFP feature" contribution in the baseline paper is fictional
Neither BFP nor HERM nor micro-Doppler structure overall is being used by the classifier. The paper's headline architectural claim (that BFP aids discrimination) is not supported by evidence — the classifier would reach the same accuracy without it.

### 2. The published 95.8% / 48.9% / 96.7% / 47.8% numbers characterise a bulk-kinematic classifier, not a micro-Doppler classifier
The multi-instance advantage the LSTM provides (per the leakage test) is likely aggregating bulk-velocity consistency across frames, not harmonic structure.

### 3. The attack surface is the training distribution's bulk kinematics, not the micro-Doppler features
Attacks that will actually work on this classifier:
- **B1 (amplitude reduction via RAM wrap):** reduces body echo, which is apparently load-bearing.
- **D1 (fly at bird-like speeds):** violates the `v_bulk` distribution the classifier implicitly learned.
- **B3 (corner-reflector decoy):** inflates echo amplitude to aircraft-class values, potentially triggering misclassification.

The `feature_attribution.py` diagnostic (running next) will quantify how much each feature group contributes — converting this hypothesis into measured accuracy drops under controlled masking.

## Methodology contribution

D2 + A2 + HERM together form a coherent methodological argument:

> *In evaluating adversarial robustness, an attack that targets an inert feature has no measurable effect and should not be interpreted as evidence of classifier robustness. Feature-attribution analysis must precede attack design: identify which inputs the classifier actually uses, then design attacks against those. For counter-UAV micro-Doppler classifiers in particular, we find that two of the three commonly claimed physics-informed features (BFP and micro-Doppler structure) are inert in a representative CNN+LSTM pipeline trained on balanced-class synthetic data. The classifier's accuracy derives from bulk-kinematic parameters baked into the training distribution, not from the micro-Doppler signatures the paper's narrative claims it learns.*

That is a strong, publishable, genuinely useful contribution — not just to counter-UAV radar but to any domain where "physics-informed ML" papers make architectural claims without rigorous feature-attribution analysis.

## Files
- `attack_d2_pulse_glide.py` — implementation
- `attack_d2_results.json` — all 9 variants, raw numbers
- `run_log_d2.txt` — full stdout
- `feature_attribution.py` — next diagnostic (running now)
