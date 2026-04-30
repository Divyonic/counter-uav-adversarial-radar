# Counter-UAV Radar &mdash; Attribution-First Adversarial Evaluation

> A case study on why null results from adversarial evaluations of counter-UAV radar classifiers can be uninterpretable without feature attribution, and what to do about it.

[![CI](https://github.com/Divyonic/counter-uav-adversarial-radar/actions/workflows/ci.yml/badge.svg)](https://github.com/Divyonic/counter-uav-adversarial-radar/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3.9%2B-blue)](requirements.txt)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Paper: PDF](https://img.shields.io/badge/paper-preprint.pdf-red)](paper/preprint.pdf)
[![Notebook: demo](https://img.shields.io/badge/demo-notebook-f37626?logo=jupyter)](notebooks/demo.ipynb)
[![Status: Working Preprint](https://img.shields.io/badge/status-working%20preprint-lightgrey)](paper/preprint.md)

**Author:** Divya Kumar Jitendra Patel, Indian Institute of Technology Madras
**Contact:** `divyakumarpatel202@gmail.com`
**Preprint:** [`paper/preprint.md`](paper/preprint.md) &middot; [`paper/preprint.pdf`](paper/preprint.pdf)
**Demo notebook:** [`notebooks/demo.ipynb`](notebooks/demo.ipynb) (runs in under a minute, no GPU)

---

## TL;DR

We train a representative counter-UAV radar classifier (CNN + LSTM + physics-informed BFP feature) and run a full threat taxonomy of seven adversarial experiments against it. Four physically motivated attacks against the architecture's advertised physics (A1, A2, D2, E1) produce **null results**. Feature-attribution experiments — both fixed-band masking and a class-conditional mask around each sample's own bulk-Doppler peak — reveal that the classifier never reads the features those attacks target; it reads bulk-Doppler peak position. The two attribution-driven attacks then split: **D1 (bird-speed flight) succeeds strongly, collapsing drone classification from 89% to 0–45%; B1 (RAM-wrap / amplitude reduction) is null because per-sample spectrogram normalisation discards absolute amplitude before the classifier sees the input**. We propose an **attribution-first workflow** as a minimum prerequisite for credible adversarial evaluation of radar ML classifiers, and demonstrate end-to-end how it would have reframed every result above.

![Feature attribution headline result](paper/figures_preprint/fig5_attr.png)

*Accuracy drop when each feature group is perturbed. Frame order and in-frame time are essentially free to scramble (grey). Spectrogram content and BFP distributional fingerprint are load-bearing (red). Attacks A2 and D2 were designed against blade-flash harmonics, which the classifier never reads.*

---

## Contents

- [The story](#the-story)
- [Findings at a glance](#findings-at-a-glance)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [How to reproduce](#how-to-reproduce)
- [Known limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

---

## The story

### Where I started

I set out to build a counter-UAV radar classifier. The setup was the one the counter-UAV ML literature has converged on over the last five or six years: FMCW radar at X-band, STFT spectrograms of the returns, a CNN on the spectrograms, an LSTM stacking ten frames together, and a hand-crafted "physics-informed" feature called Blade Flash Periodicity (BFP) concatenated at the final classifier head. Four target classes, trained on synthetic data from a physics-based simulator. The headline on a balanced test set was 95.8%.

It looked like a contribution. The architecture had a story. Every component nominally analysed a different aspect of micro-Doppler physics: the CNN read the spectrogram's time-frequency patterns, the LSTM tracked the target across frames, and BFP measured the rotor blade flash frequency that separates drones from birds. Each component had a paragraph in the baseline paper explaining what it did and why. The accuracy number sat at the top.

### The first doubt

While stress-testing the pipeline I started asking a blunt question. India already fields counter-UAV systems (SAKSHAM from BEL, Indrajaal from Grene Robotics, IDD&IS from DRDO). These systems fuse radar with RF-emission detection, EO/IR cameras, and operator judgement. They classify drones in operational conditions, not in a synthetic 4-class benchmark with hand-picked kinematic ranges. What exactly was my 95.8%-accuracy classifier adding?

The more I probed the baseline, the less convinced I was. A data-leakage diagnostic showed the LSTM's improvement over a CNN-only baseline came from within-class parameter consistency across a 10-frame window, not from real temporal tracking: the LSTM was acting as a multi-instance aggregator, learning "the last ten frames probably belong to the same object" rather than "this object's propellers spin at 5000 RPM." The BFP feature helped in some configurations and hurt in others without a clean explanation. The training distribution had class-specific kinematics baked in that an uninformed model could memorise.

I could have tuned and kept the 95.8%. Instead I pivoted.

### The pivot to adversarial robustness

The more interesting question was the opposite one: **can a motivated adversary defeat counter-UAV classifiers by making cheap physical changes to a drone?** This is an operationally relevant question. Hobbyist drones are modifiable. Radar-absorbent material costs very little. Flying slower costs nothing. A drone rebuilt with one-blade propellers is no more conspicuous to an observer than one with two. If any of these physical modifications make a published classifier misclassify, the classifier's operational value is weaker than its benchmark number suggests.

I sketched a threat taxonomy, organised by which layer of the pipeline each attack targeted. The A-series went after the BFP feature. The B-series attacked the CFAR detection stage by reducing bulk RCS (radar-absorbent wrap, for instance). The C-series was active electronic countermeasures. The D-series attacked the LSTM's multi-frame input. The E-series was an existence proof: if the attacker substitutes a flapping-wing ornithopter for a multi-rotor, no radar-only micro-Doppler system can tell it apart from a bird.

Then I picked the most physically plausible attack to run first.

### Attack A2: fewer blades

A drone with a single-blade propeller (counterweighted for balance) is a real, buildable modification. Combined with reduced RPM, the blade-flash fundamental frequency drops from around 167 Hz for a 2-blade 5000 RPM multi-rotor down to about 13 Hz, which is inside the typical bird-wingbeat band. If BFP is doing anything like the physics it claims to do, pushing the blade flash into bird range should push classifications toward bird.

I ran six variants: a clean 2-blade 5000 RPM control, then 1-blade at decreasing RPM values down to 800 RPM. Each variant was 100 sequences, same trained model, same evaluation protocol. Drone-class accuracy across all six variants stayed between 79% and 91%, with no monotonic relationship to ground-truth BFP.

**Null result.** My first read was the comfortable one: the classifier is robust to blade-count attacks. That reading turned out to be wrong, but it took me a while to see why.

The first clue came from an ablation on BFP itself. On clean drone data, the autocorrelation-based BFP extractor returned 45 Hz plus or minus 59 Hz, independent of whether the ground-truth physics was 13 Hz or 167 Hz. The feature that was supposed to measure blade-flash frequency was returning a narrow, noisy, class-correlated distribution that had almost nothing to do with the actual physics. A2 changed the physics, but the BFP extractor wasn't reading the physics in the first place.

### Attack D2: pulse-and-glide

If BFP was noise, maybe the classifier was using the CNN to read harmonic structure directly from the spectrogram. A2 left harmonic structure intact; it just moved it down in frequency. A stronger attack would remove it entirely.

D2 modelled pulse-and-glide flight. A drone alternates powered segments (rotors spinning, full propeller micro-Doppler) and gliding segments (rotors off, only bulk-Doppler body echo). For each sequence I varied the glide ratio from 0 to 100 percent, with frame order randomised to simulate realistic pulse/glide alternation. At glide ratio 1.0, every frame in every 10-frame LSTM window contained zero propeller content. Just a blob moving through the air at typical drone speed.

I expected this to work. It did not. At 100 percent glide, the classifier still labelled the sample as drone 81.3 percent of the time. Across all nine glide ratios, drone-class accuracy stayed in the 80–89 percent band.

This was a substantially stronger null than A2. A2 could be hand-waved as a noisy feature attached to a working classifier. D2 removed propeller content from the signal itself, not just from the extracted feature. If the classifier's drone decision didn't depend on propeller content at all, then what was it using?

### The attribution run

I stopped designing attacks and started measuring what the classifier actually used. Six tests, each perturbing a specific feature group on the held-out test set, each measuring the accuracy drop:

1. **Shuffle frame order within each 10-frame sequence.** Accuracy dropped by 1.6 percentage points, within noise. The LSTM does not use temporal order. This confirmed the earlier leakage-test finding: the LSTM is a multi-instance aggregator, not a temporal tracker.
2. **Zero out the central 50 percent of the time axis within each frame.** Accuracy dropped by 0.0 percentage points. Half the temporal detail inside each frame is redundant to this classifier.
3. **Zero the outer 50 percent of frequency bins.** 33 pp drop.
4. **Zero the central 25 percent of frequency bins.** 16 pp drop.
5. **Shuffle BFP vectors across samples.** 38 pp drop.
6. **Shuffle spectrograms across samples.** 41 pp drop.

Frame order: irrelevant. Time axis within a frame: mostly irrelevant. Spectrogram content: critical. BFP vector: critical.

The BFP result looked like a contradiction at first. A2 changed the physics beneath BFP and nothing happened. Attribution showed BFP was load-bearing. How? The resolution is that BFP values have class-correlated distributions: the noisy 45-Hz cluster for drones is numerically different from the noisy cluster for birds even though neither cluster measures anything meaningful. Permuting BFP across classes gives the classifier BFP values drawn from the wrong class, which confuses it. The classifier learned the distributional fingerprint of BFP noise, not the physical quantity.

The frequency-band mask results needed careful interpretation. At 9.5 GHz with PRF 8.33 kHz, each of the 128 Doppler bins is 65 Hz wide. Drones at 10–20 m/s produce bulk-Doppler peaks between 633 Hz and 1267 Hz, sitting near the edge of the "central 25 percent" band. Aircraft at 50–100 m/s sit outside the central 50 percent entirely. So the two frequency masks do not cleanly separate bulk Doppler from micro-Doppler sidebands; they separate classes with different bulk-kinematic ranges. Both masks mostly affect the band where each class's bulk-Doppler peak happens to live.

### The reveal

Piecing A2, D2, and attribution together, the simplest hypothesis that fits all three is this:

**The classifier identifies drones by the position and amplitude of the bulk-Doppler peak in the spectrogram, not by propeller harmonic structure, blade-flash periodicity, or temporal evolution across frames.**

- A2 changed the blade physics but preserved the bulk-Doppler peak. Null result predicted.
- D2 removed the propeller content but preserved the body echo at drone-typical bulk-Doppler velocity. Null result predicted.
- Spectrogram permutation breaks the bulk-Doppler peak's position within the image. Big drop, predicted.
- BFP permutation breaks the class-correlated distributional cue. Big drop, predicted.
- Frame-order permutation doesn't move any of this. No drop, predicted.

The "physics-informed CNN + LSTM + BFP" architecture, when trained on this synthetic dataset, behaves like a bulk-Doppler peak locator with a class-prior head. Minus the interpretability that a simpler model would provide.

### Closing the threat taxonomy: A1 and E1

Before designing attribution-driven attacks I wanted to make sure the original null results weren't a quirk of A2 and D2 specifically. The threat taxonomy I sketched at the start had two more attacks against advertised physics: **A1**, a pure-RPM sweep that varies BFP frequency without touching blade count; and **E1**, an "ornithopter" substitution that swaps propeller modulation for biological flapping-wing micro-Doppler while holding bulk kinematics at drone-typical values.

Both produced null results, and both produced them for the reason the attribution predicts. A1 swept RPM from 500 to 6000 with `n_blades=2` fixed, expected BFP fundamentals from 16.7 Hz to 200 Hz; drone classification stayed in the 76–93 percent band, indistinguishable from baseline. The measured BFP feature stayed at 33–44 Hz regardless of expected BFP, reproducing the BFP-noise ablation cleanly on the RPM axis alone.

E1 was the more decisive of the two. Four variants kept bulk velocity in the drone-typical range and substituted bird-style asymmetric flap micro-Doppler for propeller harmonics — drone classification stayed at 83–91 percent. Only the variant that *also* restricted velocity to the bird-overlap range collapsed accuracy, and that variant collapses to D1 (the velocity attack) rather than to E1 itself. The classifier does not notice when propellers are replaced with flapping wings, as long as the bulk-Doppler peak stays in the drone band.

Four nulls across the A-, D-, and E-series of the taxonomy. The architecture's advertised physics is not what the classifier reads, end of story.

### Sharpening the attribution: the class-conditional mask

The two fixed-band masks in the original attribution table (central 25 percent, outer 50 percent) are geometrically confounded — bulk-Doppler energy for different classes lives in different absolute frequency bins. A cleaner test masks ±N bins around *each sample's own* peak-power frequency bin. The result is striking: zeroing **just three bins (±1 bin, 2.3 percent of the frequency axis)** at each sample's peak drops accuracy by 20.8 percentage points — *more* than masking the central 25 percent of the axis with a fixed band did. Bird recall collapses to zero at this notch width.

This is the cleanest demonstration in the repo of the central attribution claim: the classifier is using bulk-Doppler peak position and post-normalisation shape, period. Three bins at the peak destroy nearly half the classifier's accuracy lead.

### The attack that works: D1, bird-speed flight

If the classifier reads bulk-Doppler peak position, the obvious physical attack is to fly the drone at a velocity whose Doppler peak overlaps an adjacent class. The training distribution has drones at 5–20 m/s and birds at 5–15 m/s, with friendly UAV at 15–35 m/s. So the drone class lives in a narrow stripe (~8–15 m/s) where it does not overlap either neighbour. Outside that stripe, the classifier should hand the prediction to whichever class the velocity is more consistent with.

I swept six velocity windows from 3–5 m/s up to 15–20 m/s, holding all other drone parameters at their training-distribution values. Drone classification collapsed in two distinct directions:

- **At v_bulk ≤ 12 m/s the drone reads as a bird.** At 5–10 m/s, drone classification drops to 45 percent; at 3–5 m/s, 78 percent of the fleet is labelled bird.
- **At v_bulk = 15–20 m/s the drone reads as a friendly UAV.** 149 of 150 samples are labelled UAV — essentially complete confusion. Friendly-coded targets are explicitly *not* engaged by counter-UAV systems.

Both misclassifications require **no hardware modification** — only a flight-controller change. There is no harmonic-content attack here, no propeller modification, no cross-section reduction. The drone simply flies outside the v_bulk band that the classifier associates with "drone." This is the attribution-driven attack working exactly as predicted, and it is the most operationally consequential finding in the project.

### The attack that doesn't work, and why that's also useful: B1

The attribution suggested two attacks: position (D1, succeeded) and amplitude (B1, this one). B1 reduces the drone's effective radar return by 0–20 dB while leaving ambient noise unchanged — the cheapest physical instantiation is a radar-absorbent material wrap. Drone classification stays in the 77–97 percent band across the entire sweep, including effective SNR of −5 dB. Null result.

The reason is upstream of the classifier. `compute_spectrogram` → `resize_spectrogram` does dB clipping to a 40 dB dynamic range and per-sample normalisation to [0, 1], which **discards absolute amplitude before the classifier sees the input**. So the classifier reads peak position and post-normalised relative shape — not absolute received power. B1 attacks a feature the classifier doesn't see, but for a different reason than A2/D2 did. Those targeted physics the architecture *describes* but doesn't compute. B1 targets a feature the architecture *would* read, but the preprocessing layer normalises away before the classifier inputs.

This refines the attribution claim: amplitude-domain physical attacks like RAM wrap defeat the system at the *detection* stage (CFAR / track confirmation), not the classifier stage. They are real attacks; they just live in a different layer of the pipeline. Both the success of D1 and the null of B1 are interpretable, and neither would have been without the prior attribution work.

### Why this matters

The two physically motivated attacks both produced null results. The initial, comfortable reading was "the classifier is robust." The correct reading is "the attacks were designed against features the classifier does not use." Those are very different conclusions. Without attribution, they are indistinguishable from the accuracy numbers alone.

This failure mode is easy to miss because null results feel like positive evidence about the defender. They are not. A null result from an attack targeting a feature the classifier ignores tells you nothing about the classifier's behaviour on attacks that target features it does use. Every adversarial evaluation in the counter-UAV ML literature that reports null results without accompanying feature-attribution analysis is exposed to this same failure mode. Some fraction of those reported robustness claims may be describing classifiers that are simply not being attacked where they live.

The methodology correction is small: run permutation importance and region masking before designing the attacks, so you know what you are aiming at. The implication is not small. Counter-UAV radar is safety-critical sensing. Systems derived from this literature are being procured and fielded. Adversarial-evaluation claims that lean on null results should come with attribution evidence that justifies the attack design.

### Where I landed

A methodology preprint, one worked example, four null results across the BFP / micro-Doppler axis, two attribution runs (fixed-band + class-conditional), one demonstrated successful attack, one workflow proposal. The argument holds up within its scope: for the specific classifier and dataset studied, adversarial evaluations against the architecture's advertised physics produce uninterpretable null results; attribution explains why; and the attack the attribution predicts succeeds.

The concrete artefacts are:

- A reproducible pipeline from FMCW signal synthesis to trained classifier ([`baseline/`](baseline/))
- Five adversarial attack implementations covering the A-, B-, D-, and E-series of the threat taxonomy ([`adversarial/attack_a1_rpm_reduction.py`](adversarial/attack_a1_rpm_reduction.py), [`attack_a2_fewer_blades.py`](adversarial/attack_a2_fewer_blades.py), [`attack_b1_ram_wrap.py`](adversarial/attack_b1_ram_wrap.py), [`attack_d1_bird_speed.py`](adversarial/attack_d1_bird_speed.py), [`attack_d2_pulse_glide.py`](adversarial/attack_d2_pulse_glide.py), [`attack_e1_ornithopter.py`](adversarial/attack_e1_ornithopter.py))
- A feature-attribution harness with both fixed-band ([`feature_attribution.py`](adversarial/feature_attribution.py)) and class-conditional ([`feature_attribution_class_conditional.py`](adversarial/feature_attribution_class_conditional.py)) tests
- A methodology proposal in the preprint ([`paper/preprint.md`](paper/preprint.md), [`paper/preprint.pdf`](paper/preprint.pdf))
- Findings write-ups for every experiment ([`FINDINGS_A1.md`](adversarial/FINDINGS_A1.md), [`FINDINGS_A2.md`](adversarial/FINDINGS_A2.md), [`FINDINGS_B1.md`](adversarial/FINDINGS_B1.md), [`FINDINGS_D1.md`](adversarial/FINDINGS_D1.md), [`FINDINGS_D2.md`](adversarial/FINDINGS_D2.md), [`FINDINGS_E1.md`](adversarial/FINDINGS_E1.md), [`FINDINGS_attribution.md`](adversarial/FINDINGS_attribution.md), [`FINDINGS_class_conditional.md`](adversarial/FINDINGS_class_conditional.md))

The natural next steps, none of which are in this repository, are: (1) real-radar replication on the Karlsson 77 GHz dataset (which Liaquat et al. 2026 already use for feature attribution with classical ML, so a direct deep-learning comparison is available), and (2) extending the attribution-first framing to adjacent "physics-informed ML" domains where the same failure mode plausibly exists.

---

## Findings at a glance

Seven experiments organised by what they target in the pipeline. Predicted-null and predicted-success outcomes are derived from the attribution analysis in §3.

| # | Experiment | What it targets | Outcome | Drone-classification accuracy |
|:-:|:-----------|:----------------|:--------|:------------------------------|
| 1 | **A2** &mdash; blade-count + RPM reduction | BFP / blade-flash physics | Null | 83–89% (clean: 88.9%) |
| 2 | **A1** &mdash; pure RPM sweep, 500–6000 RPM | BFP / blade-flash physics | Null | 76–93% |
| 3 | **D2** &mdash; pulse-and-glide flight | Propeller harmonic content | Null | 81–89% |
| 4 | **E1** &mdash; ornithopter substitution | Micro-Doppler structure | Null at drone-typical v_bulk (4 of 5 variants) | 83–91% |
| 5 | **Feature attribution** &mdash; permutation + fixed-band masking | Identifies the load-bearing feature | Bulk-Doppler peak (position + post-norm shape) | — |
| 6 | **Class-conditional mask** &mdash; ±N bins around each sample's own peak | Refinement of #5 | Strong: ±1 bin (2.3% of axis) drops accuracy 20.8 pp | 68% (clean: 88.9%) |
| 7 | **D1** &mdash; bird-speed flight | Bulk-Doppler peak position | **Strong success.** 0% drone at 15–20 m/s (UAV confusion); 21–45% at 3–10 m/s (bird confusion) | 0–45% |
| 8 | **B1** &mdash; RAM-wrap / amplitude attenuation 0–20 dB | Bulk-Doppler peak amplitude | Null. Preprocessing absorbs amplitude before the classifier | 77–97% |

Eight experiments, told in narrative order. Each builds on the previous.

### 1. Attack A2 &mdash; blade-count reduction &mdash; *null*

Reducing propeller blade count and RPM pushes the blade-flash fundamental from 167 Hz down to 13 Hz (into the bird-wingbeat band). Accuracy on the drone class stays flat across six variants.

![A2 accuracy across variants](paper/figures_preprint/fig2_a2.png)

Ablation on the BFP feature explains why: the autocorrelation-based extractor returns 45&nbsp;&plusmn;&nbsp;59 Hz regardless of whether ground-truth physics is 13 Hz or 167 Hz. The "physics-informed" feature is numerically noise.

![BFP extractor output vs ground truth](paper/figures_preprint/fig3_bfp.png)

### 2. Attack D2 &mdash; pulse-and-glide flight &mdash; *null*

Removing propeller content from the signal itself (not just from the extracted feature) by varying the glide ratio from 0% to 100%. At 100% glide, every frame in every 10-frame LSTM window contains only bulk-Doppler body echo. The classifier still labels the sample as drone 81.3% of the time.

![D2 accuracy across glide ratios](paper/figures_preprint/fig4_d2.png)

### 3. Feature attribution &mdash; what the classifier actually uses

Permutation-importance and region-masking tests resolve both nulls. The classifier identifies drones by the position and amplitude of the bulk-Doppler peak, not by harmonic structure. The LSTM is a multi-instance aggregator, not a temporal tracker. BFP is used, but as a class-correlated noise distribution.

![Feature attribution results](paper/figures_preprint/fig5_attr.png)

### 4. Attacks A1 &amp; E1 &mdash; closing the threat taxonomy &mdash; *null*

A1 (pure RPM sweep, 500–6000 RPM with `n_blades=2` fixed) and E1 (ornithopter substitution: drone-scale RCS and drone-scale velocity but bird-style flapping-wing micro-Doppler) both produce nulls. A1 keeps drone classification at 76–93% across the entire RPM sweep — the BFP extractor returns 33–44 Hz regardless of expected BFP from 16.7 to 200 Hz. E1's first four variants stay at 83–91% drone classification; the only succeeding variant restricts velocity to 5–15 m/s and is really D1 in disguise. Together they confirm: **no attack against the architecture's advertised physics moves the classifier**.

### 5. Class-conditional bulk-Doppler mask &mdash; sharpening the attribution

Masking ±N bins around *each sample's own* peak-power frequency bin disentangles bulk-Doppler peak dependence from the geometric confound in §3. The result: a 3-bin notch (±1 bin, 2.3% of the axis) drops accuracy by 20.8 pp — more than the fixed central-25% mask did with 10× more masking. Bird recall collapses to zero at this notch width.

![Class-conditional bulk-Doppler mask sweep](paper/figures_preprint/fig7_classcond.png)

### 6. Attack D1 &mdash; bird-speed flight &mdash; ***strong success***

The attribution-driven attack against bulk-Doppler peak *position*. Fly the drone at a velocity whose bulk-Doppler peak overlaps an adjacent class. Drone-classification accuracy collapses from 88.9% baseline to 0–45% across the velocity sweep — and reveals two distinct failure modes:

- At v_bulk = 15–20 m/s, drones are classified as **friendly UAV** (149/150 — essentially complete confusion). Friendly-coded targets are explicitly not engaged.
- At v_bulk = 3–10 m/s, drones are classified as **bird** (78% confusion at 3–5 m/s). A counter-UAV system that triages by class will likely not engage.

Both require no hardware modification, only a flight-controller change. This is the most operationally consequential finding in the project.

![Attack D1 results](paper/figures_preprint/fig8_d1.png)

### 7. Attack B1 &mdash; RAM-wrap / amplitude reduction &mdash; *informative null*

The attribution-driven attack against bulk-Doppler peak *amplitude*. Reduce the drone's effective radar return by 0–20 dB. Accuracy stays in the 77–97% band across the entire sweep, including −5 dB effective SNR.

The reason: `compute_spectrogram` → `resize_spectrogram` does dB clipping to 40 dB dynamic range and per-sample [0, 1] normalisation, which discards absolute amplitude before the classifier sees the input. The classifier reads peak *position* and *post-normalised* shape — not absolute power. B1 is therefore a CFAR/detection-stage attack, not a classifier-input attack on this pipeline. This sharpens the attribution claim: amplitude-domain attacks live in a different layer of the system.

![Attack B1 results](paper/figures_preprint/fig9_b1.png)

### 8. Workflow proposal

![Attribution-first workflow](paper/figures_preprint/fig6_workflow.png)

**Before designing an adversarial attack, run feature attribution.** A null result on an attack targeting features the classifier does not use is uninterpretable. The attack that targets what the classifier *does* use can succeed catastrophically with no hardware modification (D1). Both outcomes are only interpretable when the attribution work comes first.

---

## Quick start

Requires Python 3.9+ with PyTorch, NumPy, SciPy, scikit-learn, Matplotlib.

```bash
git clone https://github.com/Divyonic/counter-uav-adversarial-radar.git
cd counter-uav-adversarial-radar
pip install -r requirements.txt

# Browse the whole story in a single notebook (under a minute, no GPU):
jupyter notebook notebooks/demo.ipynb

# Or reproduce every result in the paper from scratch (~45 min CPU):
python3 baseline/train_and_evaluate.py       # baseline CNN / CNN+BFP / CNN+LSTM+BFP
python3 baseline/leakage_test.py             # diagnostic: is LSTM really temporal?
python3 adversarial/attack_a2_fewer_blades.py
python3 adversarial/attack_d2_pulse_glide.py
python3 adversarial/feature_attribution.py

# Attribution-driven attacks and the class-conditional mask
python3 adversarial/attack_a1_rpm_reduction.py
python3 adversarial/attack_b1_ram_wrap.py
python3 adversarial/attack_d1_bird_speed.py
python3 adversarial/attack_e1_ornithopter.py
python3 adversarial/feature_attribution_class_conditional.py
```

Each script is self-contained. Random seeds are pinned. Re-running produces byte-identical results (up to CPU-specific floating-point variation).

### Tests

```bash
pip install pytest
pytest tests/ -v
```

21 tests covering the simulator shapes and determinism, model forward passes, the BFP-is-noise regression guard, and results-JSON integrity. CI runs on every push across Python 3.9-3.12.

---

## Repository layout

```
counter-uav-adversarial-radar/
├── paper/
│   ├── preprint.md                  Main paper, Markdown with embedded charts
│   ├── preprint.pdf                 Rendered PDF, 8 pages, arXiv-ready
│   └── figures_preprint/            Figure sources (PNG + Mermaid .mmd)
├── notebooks/
│   ├── demo.ipynb                   Interactive walk-through, plots inline
│   └── build_demo.py                Programmatic source for demo.ipynb
├── baseline/
│   ├── fmcw_simulation.py           FMCW radar signal generator + spectrogram + BFP
│   ├── model.py                     CNN / CNN+BFP / CNN+LSTM+BFP architectures
│   ├── train_and_evaluate.py        Training + evaluation pipeline
│   ├── leakage_test.py              Diagnostic: randomise per-frame params
│   ├── herm_extractor.py            Alternative HERM-based feature (null result)
│   └── results/                     Baseline experiment JSON outputs
├── adversarial/
│   ├── attack_a1_rpm_reduction.py   Attack A1: RPM-only sweep (predicted null)
│   ├── attack_a2_fewer_blades.py    Attack A2: variable blade count x RPM
│   ├── attack_b1_ram_wrap.py        Attack B1: bulk-amplitude reduction (RAM)
│   ├── attack_d1_bird_speed.py      Attack D1: bird-speed flight (succeeds)
│   ├── attack_d2_pulse_glide.py     Attack D2: variable pulse-and-glide ratio
│   ├── attack_e1_ornithopter.py    Attack E1: ornithopter substitution
│   ├── feature_attribution.py                 Permutation + fixed-band masking
│   ├── feature_attribution_class_conditional.py  Per-sample peak-bin masking
│   ├── FINDINGS_A1.md, FINDINGS_A2.md          Predicted-null A-series
│   ├── FINDINGS_B1.md                          B1 null + preprocessing critique
│   ├── FINDINGS_D1.md                          D1 successful attack
│   ├── FINDINGS_D2.md                          D2 null result
│   ├── FINDINGS_E1.md                          E1 ornithopter
│   ├── FINDINGS_attribution.md                 Fixed-band attribution
│   ├── FINDINGS_class_conditional.md           Class-conditional refinement
│   ├── results/*.json               Raw experiment outputs
│   └── run_log_*.txt                Full stdout logs
├── tests/                           pytest suite, runs in ~2 seconds
│   ├── test_fmcw_simulation.py      Simulator shapes, determinism, BFP regression
│   ├── test_model.py                Classifier forward passes, parameter counts
│   └── test_results_integrity.py    Guards against silent findings drift
└── .github/workflows/ci.yml         CI: pytest + ruff, Python 3.9-3.12
```

---

## How to reproduce

Each experiment is a single self-contained script; no arguments required. Expected runtimes are CPU, M-series Mac.

| Script                                   | Runtime | Output                                             |
|:-----------------------------------------|:-------:|:---------------------------------------------------|
| `baseline/train_and_evaluate.py`         | 5-15 min | `baseline/results/experiment_results.json`         |
| `baseline/leakage_test.py`               | 3-10 min | `baseline/results/leakage_test_results.json`       |
| `adversarial/attack_a1_rpm_reduction.py` | 8-12 min | `adversarial/results/attack_a1_results.json`       |
| `adversarial/attack_a2_fewer_blades.py`  | 5-10 min | `adversarial/results/attack_a2_results.json`       |
| `adversarial/attack_b1_ram_wrap.py`      | 8-12 min | `adversarial/results/attack_b1_results.json`       |
| `adversarial/attack_d1_bird_speed.py`    | 8-12 min | `adversarial/results/attack_d1_results.json`       |
| `adversarial/attack_d2_pulse_glide.py`   | 10-15 min | `adversarial/results/attack_d2_results.json`      |
| `adversarial/attack_e1_ornithopter.py`   | 8-12 min | `adversarial/results/attack_e1_results.json`       |
| `adversarial/feature_attribution.py`     | 10-15 min | `adversarial/results/feature_attribution_results.json` |
| `adversarial/feature_attribution_class_conditional.py` | 12-18 min | `adversarial/results/feature_attribution_class_conditional_results.json` |

To regenerate the preprint PDF (requires Node.js, one-time `npx` downloads):

```bash
cd paper/figures_preprint
npx --yes md-to-pdf preprint_pdf.md
mv preprint_pdf.pdf ../preprint.pdf
```

---

## Known limitations

1. **Synthetic data only.** The entire evaluation uses physics-based simulated FMCW returns. Extension to real radar data (Karlsson 77 GHz on Zenodo, DIAT-&mu;SAT on IEEE DataPort) is proposed in the preprint but not yet attempted.
2. **Single classifier architecture.** The critique applies directly to the CNN + LSTM + BFP baseline; generalisation to other counter-UAV ML architectures is suggested but not proved.
3. **D1's "drone &rarr; friendly UAV" failure mode is training-distribution-specific.** The 99% confusion at v_bulk = 15–20 m/s exists because the training distribution overlaps drone (5–20 m/s) with friendly UAV (15–35 m/s). A different training distribution would close that window, but it would not close the bird window — birds and slow drones overlap in any realistic distribution.
4. **B1 attacks the wrong layer.** B1's null result is informative (per-sample dB clipping + [0,1] normalisation discards absolute amplitude), but it does not test classifier robustness against amplitude attacks combined with the upstream CFAR detection stage. A complete amplitude-attack evaluation should integrate detection-time effects, which are out of scope here.
5. **Preprint is one revision behind the code.** [`paper/preprint.md`](paper/preprint.md) covers A2, D2, and the original fixed-band attribution but predates the B1, D1, A1, E1, and class-conditional results in this repo. The findings write-ups in [`adversarial/`](adversarial/) are the current source of truth.

---

## Citation

If this work is useful, please cite:

**Plain text**

```
Divya Kumar Jitendra Patel (2026). Feature-Attribution-First Adversarial
Evaluation of a Physics-Informed Counter-UAV Radar Classifier. Working
preprint, Indian Institute of Technology Madras.
https://github.com/Divyonic/counter-uav-adversarial-radar
```

**BibTeX**

```bibtex
@misc{patel2026attribution,
  author       = {Divya Kumar Jitendra Patel},
  title        = {Feature-Attribution-First Adversarial Evaluation of a
                  Physics-Informed Counter-UAV Radar Classifier},
  year         = {2026},
  institution  = {Indian Institute of Technology Madras},
  note         = {Working preprint},
  howpublished = {\url{https://github.com/Divyonic/counter-uav-adversarial-radar}}
}
```

Machine-readable metadata is also available in [`CITATION.cff`](CITATION.cff).

---

## License

Released under the [MIT License](LICENSE). Copyright &copy; 2026 Divya Kumar Jitendra Patel.

---

## Contact

Divya Kumar Jitendra Patel, Indian Institute of Technology Madras
`divyakumarpatel202@gmail.com`
