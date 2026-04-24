---
title: "Feature-Attribution-First Adversarial Evaluation of a Physics-Informed Counter-UAV Radar Classifier"
stylesheet: https://cdn.jsdelivr.net/npm/github-markdown-css@5.5.1/github-markdown-light.min.css
body_class: markdown-body
css: |-
    .page-break { page-break-after: always; }
    .markdown-body { font-size: 11px; max-width: 100%; padding: 0; }
    .markdown-body img { max-width: 100%; }
    .markdown-body h1 { font-size: 22px; }
    .markdown-body h2 { font-size: 16px; margin-top: 22px; }
    .markdown-body h3 { font-size: 13px; }
    .markdown-body table { font-size: 10px; }
pdf_options:
    format: A4
    margin: 18mm
    printBackground: true
---

# Feature-Attribution-First Adversarial Evaluation of a Physics-Informed Counter-UAV Radar Classifier

**Divya Kumar Jitendra Patel**
Indian Institute of Technology, Madras
`divyakumarpatel202@gmail.com`

*April 2026 — working preprint.*

---

## Abstract

We report a case study on the adversarial evaluation of a representative counter-UAV radar classifier — a CNN + LSTM architecture augmented with a hand-crafted Blade Flash Periodicity (BFP) feature, trained on synthetic 9.5 GHz FMCW micro-Doppler data across four target classes (multi-rotor drone, bird, fixed-wing UAV, manned aircraft). Two physically motivated attacks — blade-count reduction (A2) and pulse-and-glide flight (D2) — produce null results on the baseline model; accuracy does not drop. Permutation-importance and frequency-band masking show *why*: the classifier is not reading blade-flash harmonics or propeller micro-Doppler structure at all. It locates the bulk-Doppler peak and classifies from its position and amplitude, with the LSTM acting as a multi-instance aggregator rather than a temporal tracker. The BFP feature is used, but as a class-correlated noise distribution rather than as a physics measurement. We argue that *adversarial evaluation without feature attribution is unreliable*: attacks targeting features the classifier does not use produce null results that are easy to misread as robustness. We propose an attribution-first workflow as a prerequisite for credible adversarial evaluation of radar ML, and release all code and data to support replication.

---

## 1. Introduction

Counter-UAV radar classifiers are moving from research prototypes into operational deployment. Vendors claim robustness on the basis of adversarial evaluation — typically a small set of physically motivated attacks run against the classifier, with accuracy reported as the outcome. A null result (accuracy holds) is read as evidence of robustness.

This paper documents a case where that reading is wrong. We train a baseline classifier that matches the architectural pattern commonly published in the counter-UAV ML literature (CNN + LSTM + physics-informed hand-crafted feature), stress it with two physically motivated attacks designed around the propeller micro-Doppler physics the architecture claims to analyse, and find that neither attack moves accuracy. The reason is not robustness: feature-attribution measurements show the classifier never reads the features the attacks are designed to disturb.

Our contribution is methodological. We show, through one worked example, that:

1. A CNN + LSTM + BFP classifier trained on balanced synthetic FMCW data can achieve high accuracy without using any of the physics features it is architecturally claimed to use.
2. Adversarial evaluations that target those features produce null results that characterise the *attack design*, not the *classifier's robustness*.
3. Permutation-importance and region-masking tests resolve the ambiguity. Prefixing adversarial evaluation with attribution analysis would have reframed both attacks before they were run.

We do not claim this failure mode is universal across the counter-UAV literature. We claim it is plausible enough that attribution-first evaluation should be a prerequisite to any adversarial robustness claim.

## 2. Baseline classifier

### 2.1 Synthetic dataset

We generate 9.5 GHz FMCW radar returns for four target classes using a physics-based simulator (code in `baseline/fmcw_simulation.py`). Parameters follow published counter-UAV simulation studies: PRF 8.33 kHz, range resolution 0.5 m, chirp duration 120 µs. Each sample is a 10-frame sequence of 128 × 128 micro-Doppler spectrograms with the associated BFP feature vector. Training set: 300 samples per class at SNR 15 dB.

Class kinematic distributions:

| Class           | `v_bulk` (m/s) | `rcs_body` (m²)  | Notes                                |
|:----------------|:--------------:|:----------------:|:-------------------------------------|
| Multi-rotor     | 5 – 20         | 0.01             | 2-blade props, 4000–6000 RPM         |
| Bird            | 5 – 15         | 0.005            | 2–12 Hz flap, 0.2–0.8 m wingspan     |
| Fixed-wing UAV  | 15 – 35        | 0.05             | 2-blade props, 2500–4500 RPM         |
| Manned aircraft | 50 – 100       | 1–10             | Engine modulation, no rotor harmonic |

### 2.2 Architecture

A 4-block CNN extracts per-frame spectrogram features (~0.45M parameters). A 2-layer LSTM aggregates across the 10 frames. The final hidden state is concatenated with a 3-dimensional BFP feature vector computed via autocorrelation of the temporal envelope (code in `baseline/model.py`). A softmax head produces class probabilities.

![Figure 1: baseline classifier pipeline](fig1_pipeline.png)

**Figure 1.** Baseline classifier pipeline. The spectrogram stream feeds a CNN + LSTM; the BFP physics feature is concatenated at the final classifier head.

### 2.3 Clean-data performance

On held-out synthetic data the full model reaches 88.9% accuracy (the original paper cites 95.8% on a larger training set; we use a smaller set here for faster iteration). Per-class confusion aligns with class-kinematic separability: manned aircraft identified near-perfectly, multi-rotor vs bird hardest.

## 3. Adversarial attacks

We run two attacks. Both were designed against the physics the architecture claims to analyse. Both produce null results. For attack implementations see `adversarial/attack_a2_fewer_blades.py` and `adversarial/attack_d2_pulse_glide.py`.

### 3.1 A2 — blade-count reduction

Drones with single-blade propellers (with counterweights) are physically realistic and reduce the blade-flash fundamental frequency by a factor of two. Combined with reduced RPM, the blade-flash fundamental can be pushed into the bird-wingbeat band (5–20 Hz), which should destroy BFP as a discriminator. Six variants test the gradient from "clean 2-blade control" to "1-blade, 800 RPM, BFP matches typical bird flap."

Result: accuracy on the drone class stays between 83.3% and 89.3% across all variants — indistinguishable from the unmasked baseline. Pushing BFP into the bird range does not push predictions toward bird.

![Figure 2: Attack A2 accuracy across variants](fig2_a2.png)

**Figure 2.** A2 results. Blade-count reduction combined with RPM reduction was designed to push blade-flash frequency from 167 Hz down into the bird-wingbeat range (13–20 Hz). Accuracy does not drop.

An ablation on the BFP feature itself explains this. On clean drone data, the autocorrelation-based BFP extractor returns 45 ± 59 Hz regardless of whether the physical ground truth is 13 Hz or 167 Hz. The feature is numerically noise. Classification accuracy is unchanged by A2 because the *measured* BFP was already unrelated to the physics the attack modifies.

![Figure 3: BFP measurement vs ground truth](fig3_bfp.png)

**Figure 3.** The measured BFP feature is numerically decoupled from the physics it claims to measure. Green bars show the physical blade-flash frequency computed from ground-truth blade count and RPM; grey bars show the mean output of the BFP extractor across 100 samples at each configuration. Extractor output remains in a narrow ~30–45 Hz band regardless of whether ground truth is 13 Hz or 167 Hz.

### 3.2 D2 — pulse-and-glide

Drones that alternate powered and unpowered flight segments present a classifier with input sequences in which a fraction of frames contain body-echo only (no propeller content). We vary the "glide ratio" from 0 (all frames have propeller content) to 1.0 (every frame in the 10-frame LSTM window is glide-only). Frame order is randomised within each sequence.

Result: accuracy on the drone class stays in the 80.0%–89.3% band across all glide ratios. Even when every frame in every sequence contains zero propeller content — only body echo at bulk Doppler — the classifier still labels the sample as drone 81.3% of the time.

![Figure 4: Attack D2 accuracy across glide ratios](fig4_d2.png)

**Figure 4.** D2 results. Across nine glide ratios, including the extreme case where every frame in every LSTM window contains zero propeller content, classifier accuracy remains in the baseline band.

This is a stronger outcome than A2. A2 could be explained by a noisy BFP extractor still driving a real feature. D2 removes propeller content from the signal itself, not just from the extracted feature. The classifier's drone decision clearly does not depend on propeller content at any level.

## 4. Feature attribution

To resolve what the classifier *is* using, we run permutation-importance and region-masking tests on the held-out test set. Code in `adversarial/feature_attribution.py`. Results in `adversarial/feature_attribution_results.json`.

| Perturbation                                  | Accuracy | Drop (pp) |
|:----------------------------------------------|:--------:|:---------:|
| *(clean baseline)*                            | 88.9%    | —         |
| Spectrogram permutation across samples        | 47.7%    | +41.2     |
| BFP permutation across samples                | 51.0%    | +37.9     |
| Frequency mask: outer 50% of bins             | 55.6%    | +33.3     |
| Frequency mask: central 25% of bins           | 72.9%    | +16.0     |
| Temporal mask: central 50% of time bins       | 88.9%    | +0.0      |
| Frame-order permutation within each sequence  | 90.5%    | −1.6      |

![Figure 5: Feature attribution results](fig5_attr.png)

**Figure 5.** Feature attribution results, sorted by impact. Grey bars indicate tests that leave accuracy unchanged — frame order and within-frame time axis. Coloured bars indicate load-bearing features: frequency-band content and BFP distributional fingerprint. Note that the two frequency-mask tests are geometrically confounded because bulk-Doppler energy for different classes lives in different Doppler bands (see text).

Three things stand out. First, frame order does not matter (−1.6 pp; within run-to-run noise). The LSTM functions as a multi-instance aggregator, not a temporal tracker — consistent with a separate data-leakage test we ran early in the project. Second, half the time axis can be zeroed with zero accuracy impact. The within-frame temporal structure — which is where blade-flash periodicity lives — is redundant. Third, BFP permutation causes a 38 pp drop, which at first appears to contradict A2. It does not. BFP values have class-correlated *distributions* (a noisy 45 Hz cluster for drones, a different noisy cluster for birds, and so on); shuffling the vectors across classes hands the classifier BFP values drawn from the wrong class. The classifier learns the distributional fingerprint of BFP noise, not the physical quantity BFP is supposed to measure.

The frequency-band masks need care. At 9.5 GHz with PRF 8.33 kHz, each of the 128 Doppler bins is ~65 Hz wide. Drones at 10–20 m/s have bulk-Doppler peaks between 633 and 1267 Hz, which puts them at the *edge* of the central 25% band. Aircraft at 50–100 m/s have bulk-Doppler peaks outside the central 50% region entirely. So the two masks do not cleanly separate "bulk Doppler" from "micro-Doppler sidebands"; they separate classes with different bulk-kinematic ranges.

Reconciled with D2, the simplest account that fits all three pieces of evidence is: the classifier identifies drones by the position and amplitude of the bulk-Doppler peak. D2 preserves this peak (it only removes propeller content around it), so D2 cannot defeat the classifier. Masking that peak band (or permuting spectrograms entirely) does defeat it. BFP is used as a class-correlated noise proxy that co-varies with bulk kinematics in the training distribution, which is why permuting BFP across classes hurts accuracy while changing BFP physics does not.

## 5. Discussion: why evaluation needs attribution first

The paper's central observation is that both attacks were designed around the same incorrect premise — that the classifier's architecture reflects the features it uses. The architecture advertises blade-flash periodicity analysis, temporal tracking across frames, and spectrogram micro-Doppler analysis. A feature-attribution measurement beforehand would have shown that the LSTM is not temporal, the in-frame time axis is redundant, the BFP feature is noise, and the classifier's accuracy reduces to a bulk-Doppler peak locator.

Two specific failure modes follow. First, a null result on A2 was initially easy to read as "the classifier is robust to blade-count manipulation." The correct reading is "the attack modified a feature the classifier does not use." Second, a null result on D2 initially suggested that micro-Doppler as a whole is inert inside the classifier. Attribution refines that: the classifier does use the *frequency band* in which drone propeller energy lives, but it uses the band's overall position and shape — not the harmonic structure that D2 disturbs.

For adversarial evaluation of radar classifiers, we therefore propose the following order of operations:

1. Run permutation importance across each input modality and per-sample region masks across each input axis. Record which perturbations damage accuracy.
2. Inspect high-impact regions for their physical meaning. Separate bulk-kinematic from micro-kinematic dependence where possible (class-conditional masking around each sample's own bulk-Doppler peak is a cleaner version of our frequency-mask test).
3. Design attacks against features the classifier actually uses, not against features the architecture implies it uses.
4. Report attack success alongside the attribution results that justify its design. A null result is uninterpretable in isolation.

This is a lightweight addition to an existing evaluation. The cost (one run of permutation importance per architecture) is small relative to the cost of a full adversarial evaluation, and the resulting attacks are substantially more informative.

![Figure 6: attribution-first vs traditional workflow](fig6_workflow.png)

**Figure 6.** The attribution-first workflow (solid arrows) contrasted with the traditional approach (dotted arrows). A null result from a traditional evaluation cannot distinguish classifier robustness from an attack targeting features the classifier does not use. Attribution-first evaluation produces attacks whose outcomes are interpretable regardless of whether they succeed or fail.

## 6. Limitations

The critique is based on a single classifier architecture and a single synthetic dataset. We make no claim that every published counter-UAV classifier exhibits this failure mode. We claim only that the failure mode is possible for architectures of this general shape, and that the field's common practice of reporting adversarial-attack success rates without accompanying feature attribution cannot distinguish this failure mode from genuine robustness.

We do not validate on real radar data. The Karlsson 77 GHz dataset (Zenodo 5511912) and DIAT-µSAT (IEEE DataPort 10.21227/1x2q-8v62) are both suitable targets for replication. Liaquat et al. 2026 (arXiv 2604.12567) have already performed feature attribution on the Karlsson dataset using classical ML; extending their work to a deep-learning classifier is the natural next step and is not attempted here.

We do not demonstrate a successful attack in this preprint. Attribution predicts that attacks against bulk-Doppler position (reduced airspeed) or bulk-echo amplitude (radar-absorbent material) should succeed. These are straightforward to simulate but outside the scope of this work.

## 7. Related work

Adversarial robustness of micro-Doppler classifiers has been studied directly by Czerkawski et al. 2024 (arXiv 2402.13651), who report that CNN classifiers on spectrograms are highly susceptible to gradient-based attacks and propose cadence-velocity-diagram representations plus adversarial training as defences. Their attack is gradient-based on the spectrogram input, rather than on a physical parameter of the target. Liaquat et al. 2026 (arXiv 2604.12567) evaluate the noise-robustness of hand-crafted features with SVM/RF classifiers on the Karlsson 77 GHz FMCW dataset and run permutation importance; deep-learning classifiers and physical attacks are not in scope.

Signal-domain physical attacks exist in adjacent radar modalities. Peng et al. 2022 (arXiv 2209.04779) introduce SMGAA, which generates adversarial scatterers for SAR automatic target recognition using an attributed scattering-centre model. Lemeire et al. 2025 (arXiv 2511.03192) demonstrate aspect-angle-invariant corner-reflector placement against SAR ATR with 80% fooling. Gazit et al. 2025 (arXiv 2512.20712) present the first physical over-the-air attack against RF-based drone detectors, transmitting universal I/Q perturbations. None of these target radar micro-Doppler classifiers; the intersection of (signal-domain attack) × (radar micro-Doppler) × (real data) is empty in the current literature.

The broader shortcut-learning literature (Geirhos et al. 2020) frames our observation in general terms: a classifier solving a task via features that co-vary with the label in the training distribution, rather than features that generalise, is a shortcut-learner. Our contribution is to show that this failure mode survives an adversarial evaluation procedure that does not check for it.

## 8. Conclusion

We trained a physics-informed counter-UAV classifier, attacked its claimed physics (propeller blade count, pulse-and-glide flight), found null results, and then measured what the classifier was actually using. The attacks were targeting features the classifier does not read. We propose attribution-first evaluation as a lightweight prerequisite for credible adversarial robustness claims in radar ML.

## Reproducibility

All code, datasets, trained model weights, experiment logs, and intermediate results are archived at [github.com/Divyonic/counter-uav-adversarial-radar](https://github.com/Divyonic/counter-uav-adversarial-radar).

Each experiment script is self-contained and seeded. Running the following reproduces every number in this paper, in order:

```
python3 baseline/train_and_evaluate.py
python3 baseline/leakage_test.py
python3 adversarial/attack_a2_fewer_blades.py
python3 adversarial/attack_d2_pulse_glide.py
python3 adversarial/feature_attribution.py
```

## References

- Czerkawski, M., Clemente, C., Michie, C., and Tachtatzis, C. *Robustness of Deep Neural Networks for Micro-Doppler Radar Classification.* arXiv:2402.13651, 2024.
- Liaquat, S. et al. *Feature-Level Robustness of Physics-Guided Micro-Doppler Descriptors for Classification of Drones and Birds.* arXiv:2604.12567, 2026.
- Peng, B. et al. *Scattering Model Guided Adversarial Examples for SAR Target Recognition: Attack and Defense.* arXiv:2209.04779, 2022.
- Lemeire, I. et al. *SAAIPAA: Optimizing Aspect-Angle-Invariant Physical Adversarial Attacks on SAR Target Recognition.* arXiv:2511.03192, 2025.
- Gazit, O., Itzhakev, Y., Elovici, Y., and Shabtai, A. *Real-World Adversarial Attacks on RF-Based Drone Detectors.* arXiv:2512.20712, 2025.
- Abdulatif, S., Armanious, K., Aziz, F., Schneider, U., and Yang, B. *Towards Adversarial Denoising of Radar Micro-Doppler Signatures.* arXiv:1811.04678, 2018.
- Kokalj-Filipovic, S. and Miller, R. *Adversarial Examples in RF Deep Learning: Detection of the Attack and its Physical Robustness.* arXiv:1902.06044, 2019.
- Karlsson, A. *Radar Measurements on Drones, Birds and Humans with a 77 GHz FMCW Sensor.* Zenodo record 5511912, 2021.
- Geirhos, R. et al. *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence, 2020.
