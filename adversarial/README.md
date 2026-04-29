# Adversarial experiments

This directory contains the adversarial experiments that drive the preprint's methodology argument. Each experiment is self-contained, each re-trains the baseline classifier from scratch using [`../baseline/`](../baseline/), and each writes both a raw results JSON and a findings write-up.

For the full narrative see [`../paper/preprint.md`](../paper/preprint.md). For the attribution-first workflow argument see the top-level [README](../README.md).

The experiments are organised by the layer of the pipeline they target. The original threat taxonomy in the project's narrative used letter-prefixes: A-series attacks the BFP feature, B-series attacks the bulk-RCS / detection layer, D-series attacks the LSTM's multi-frame input via velocity or temporal manipulation, E-series tests the existence proof of biological-signature substitution.

---

## Predicted-null experiments (BFP / micro-Doppler attacks)

### A2 &mdash; blade-count reduction (joint with RPM)

| Item | Value |
|:-----|:------|
| **Script** | [`attack_a2_fewer_blades.py`](attack_a2_fewer_blades.py) |
| **Write-up** | [`FINDINGS_A2.md`](FINDINGS_A2.md) |
| **Raw results** | [`attack_a2_results.json`](attack_a2_results.json) |
| **Outcome** | Null. Accuracy stays in the 83-89% band across six variants. |

Six variants spanning the gradient from "clean 2-blade control" to "1-blade at 800 RPM, BFP at typical bird-flap frequency." Hypothesis was that pushing the blade-flash fundamental into the bird-wingbeat band would destroy BFP as a discriminator. Ablation on the BFP feature itself shows the autocorrelation extractor returns 45 &plusmn; 59 Hz on clean drone data regardless of physical ground truth (13–167 Hz). The "physics-informed feature" is numerically noise; any attack that modifies only the physics underneath it has no measurable effect.

### A1 &mdash; pure RPM reduction

| Item | Value |
|:-----|:------|
| **Script** | [`attack_a1_rpm_reduction.py`](attack_a1_rpm_reduction.py) |
| **Write-up** | [`FINDINGS_A1.md`](FINDINGS_A1.md) |
| **Raw results** | [`attack_a1_results.json`](attack_a1_results.json) |
| **Outcome** | Null. Accuracy 76–93% across RPM 500–6000. |

A-series completion test: A2 covered blade-count + RPM jointly. A1 isolates the RPM axis (n_blades=2 fixed, RPM swept 500–6000, expected BFP 16.7–200 Hz). Measured BFP stays at 33–44 Hz regardless of expected BFP, reproducing A2's BFP-noise ablation cleanly on the RPM axis alone.

### D2 &mdash; pulse-and-glide flight

| Item | Value |
|:-----|:------|
| **Script** | [`attack_d2_pulse_glide.py`](attack_d2_pulse_glide.py) |
| **Write-up** | [`FINDINGS_D2.md`](FINDINGS_D2.md) |
| **Raw results** | [`attack_d2_results.json`](attack_d2_results.json) |
| **Outcome** | Null. At 100% glide (no propeller content anywhere in the sequence), the classifier still labels it drone 81.3% of the time. |

Nine variants, glide ratio from 0 (every frame has propeller content) to 1.0 (every frame in every 10-frame LSTM window is glide-only, body-echo only). Frame order randomised within each sequence to model realistic pulse-glide alternation. This is a stronger null than A2: A2 could be explained by a noisy BFP extractor still driving a real feature downstream. D2 removes propeller content from the signal itself, not just from the extracted feature. The classifier's drone decision does not depend on propeller content at any level.

### E1 &mdash; ornithopter substitution

| Item | Value |
|:-----|:------|
| **Script** | [`attack_e1_ornithopter.py`](attack_e1_ornithopter.py) |
| **Write-up** | [`FINDINGS_E1.md`](FINDINGS_E1.md) |
| **Raw results** | [`attack_e1_results.json`](attack_e1_results.json) |
| **Outcome** | Null at drone-typical v_bulk (4 variants, 83–91% drone classification). The single succeeding variant collapses to D1 (it restricts v_bulk to bird-overlap range). |

Custom signal generator combining drone-scale body RCS and drone-scale bulk velocity with bird-style asymmetric flapping-wing micro-Doppler. Closes the existence-proof gap in the attribution evidence: a physically distinct micro-Doppler signature is invisible to the classifier when bulk-Doppler position is held in the drone band.

---

## Predicted-success experiments

### D1 &mdash; bird-speed flight

| Item | Value |
|:-----|:------|
| **Script** | [`attack_d1_bird_speed.py`](attack_d1_bird_speed.py) |
| **Write-up** | [`FINDINGS_D1.md`](FINDINGS_D1.md) |
| **Raw results** | [`attack_d1_results.json`](attack_d1_results.json) |
| **Outcome** | **Strong success.** Drone-classification accuracy collapses to 0–45% across the velocity sweep. |

Six velocity windows from 15–20 m/s down to 3–5 m/s. Two distinct failure modes appear: at low speeds (≤ 12 m/s) the drone reads as bird (78% confusion at v=3–5 m/s); at high speeds (≥ 15 m/s) the drone reads as friendly UAV (99% confusion at v=15–20 m/s). The classifier's "drone" decision occupies a narrow window in v_bulk, and any drone flown outside that window is reclassified. This is the first attack in the methodology study that succeeds, and it succeeds in exactly the way the attribution analysis predicted.

### B1 &mdash; RAM-wrap / bulk-amplitude reduction

| Item | Value |
|:-----|:------|
| **Script** | [`attack_b1_ram_wrap.py`](attack_b1_ram_wrap.py) |
| **Write-up** | [`FINDINGS_B1.md`](FINDINGS_B1.md) |
| **Raw results** | [`attack_b1_results.json`](attack_b1_results.json) |
| **Outcome** | Null, but instructive. Accuracy stays 77–97% across drops 0–20 dB. |

Reducing the drone's effective radar return by 0–20 dB does not move the classifier. The reason is preprocessing: `compute_spectrogram` → `resize_spectrogram` does dB clipping to a 40 dB dynamic range and per-sample [0, 1] normalisation, which discards absolute amplitude before the classifier sees the input. This sharpens the attribution claim — the classifier reads bulk-Doppler peak *position* and *post-normalised relative shape*, not absolute amplitude. B1 is therefore a CFAR/detection-stage attack rather than a classifier-input attack on this pipeline.

### Class-conditional bulk-Doppler mask

| Item | Value |
|:-----|:------|
| **Script** | [`feature_attribution_class_conditional.py`](feature_attribution_class_conditional.py) |
| **Write-up** | [`FINDINGS_class_conditional.md`](FINDINGS_class_conditional.md) |
| **Raw results** | [`feature_attribution_class_conditional_results.json`](feature_attribution_class_conditional_results.json) |
| **Outcome** | **Strong success.** Masking ±1 freq bin (2.3% of axis) at each sample's own peak drops accuracy by 20.8 pp — more than the original fixed central 25% mask (16.0 pp). |

Disentangles the bulk-Doppler peak from the rest of the spectrogram by masking ±N bins around *each sample's own* peak-power frequency bin, instead of a fixed band. The class-conditional version is roughly 10× more efficient per masked bin than the fixed-band masks. Bird recall drops to 0 even at ±1 bin — birds depend on their bulk-Doppler peak more than any other class.

---

## Feature attribution (original fixed-band tests)

| Item | Value |
|:-----|:------|
| **Script** | [`feature_attribution.py`](feature_attribution.py) |
| **Write-up** | [`FINDINGS_attribution.md`](FINDINGS_attribution.md) |
| **Raw results** | [`feature_attribution_results.json`](feature_attribution_results.json) |
| **Run log** | [`run_log_feat_attr.txt`](run_log_feat_attr.txt) |

Six perturbation tests on the trained classifier: spectrogram permutation across samples, BFP permutation across samples, frame-order permutation within each sequence, frequency-band masks (central 25% and outer 50%), and a temporal mask (central 50% of time bins). The class-conditional mask above refines the frequency-band tests by removing the per-class-confound.

---

## Reading order

For a new reader:

1. Skim the preprint: [`../paper/preprint.md`](../paper/preprint.md).
2. Read the four predicted-null write-ups in order: [`FINDINGS_A2.md`](FINDINGS_A2.md), [`FINDINGS_A1.md`](FINDINGS_A1.md), [`FINDINGS_D2.md`](FINDINGS_D2.md), [`FINDINGS_E1.md`](FINDINGS_E1.md).
3. Read [`FINDINGS_attribution.md`](FINDINGS_attribution.md) and [`FINDINGS_class_conditional.md`](FINDINGS_class_conditional.md), which together resolve the nulls and identify the load-bearing feature.
4. Read [`FINDINGS_D1.md`](FINDINGS_D1.md) and [`FINDINGS_B1.md`](FINDINGS_B1.md) — the attribution-driven attacks. D1 succeeds; B1 fails for reasons that further refine the attribution.
5. Run any of the scripts to reproduce the numbers.

## Logs

- [`run_log_d2.txt`](run_log_d2.txt) — original D2 run
- [`run_log_feat_attr.txt`](run_log_feat_attr.txt) — original fixed-band attribution run
- [`run_log_b1_d1_classcond.txt`](run_log_b1_d1_classcond.txt) — B1 + D1 + class-conditional mask
- [`run_log_a1_e1.txt`](run_log_a1_e1.txt) — A1 + E1
