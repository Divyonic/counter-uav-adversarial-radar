# Counter-UAV Radar &mdash; Attribution-First Adversarial Evaluation

> A case study on why null results from adversarial evaluations of counter-UAV radar classifiers can be uninterpretable without feature attribution, and what to do about it.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3](https://img.shields.io/badge/python-3.9%2B-blue)](requirements.txt)
[![Framework: PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)](https://pytorch.org/)
[![Paper: PDF](https://img.shields.io/badge/paper-preprint.pdf-red)](paper/preprint.pdf)
[![Status: Working Preprint](https://img.shields.io/badge/status-working%20preprint-lightgrey)](paper/preprint.md)

**Author:** Divya Kumar Jitendra Patel, Indian Institute of Technology Madras
**Contact:** `divyakumarpatel202@gmail.com`
**Preprint:** [`paper/preprint.md`](paper/preprint.md) &middot; [`paper/preprint.pdf`](paper/preprint.pdf)

---

## TL;DR

We train a representative counter-UAV radar classifier (CNN + LSTM + physics-informed BFP feature), run two physically motivated adversarial attacks, and measure null results. Feature-attribution experiments then reveal that the classifier never uses the features the attacks target. The null results therefore say nothing about robustness. We propose an **attribution-first workflow** as a minimum prerequisite for credible adversarial evaluation of radar ML classifiers.

![Feature attribution headline result](paper/figures_preprint/fig5_attr.png)

*Accuracy drop when each feature group is perturbed. Frame order and in-frame time are essentially free to scramble (grey). Spectrogram content and BFP distributional fingerprint are load-bearing (red). Attacks A2 and D2 were designed against blade-flash harmonics, which the classifier never reads.*

---

## Contents

- [Findings at a glance](#findings-at-a-glance)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [How to reproduce](#how-to-reproduce)
- [Known limitations](#known-limitations)
- [Citation](#citation)
- [License](#license)

---

## Findings at a glance

Three experiments, run in sequence. Each builds on the previous.

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

### 4. Workflow proposal

![Attribution-first workflow](paper/figures_preprint/fig6_workflow.png)

**Before designing an adversarial attack, run feature attribution.** A null result on an attack targeting features the classifier does not use is uninterpretable.

---

## Quick start

Requires Python 3.9+ with PyTorch, NumPy, SciPy, scikit-learn, Matplotlib.

```bash
git clone https://github.com/Divyonic/counter-uav-adversarial-radar.git
cd counter-uav-adversarial-radar
pip install -r requirements.txt

# Reproduce every result in the paper, in order (~45 min CPU):
python3 baseline/train_and_evaluate.py       # baseline CNN / CNN+BFP / CNN+LSTM+BFP
python3 baseline/leakage_test.py             # diagnostic: is LSTM really temporal?
python3 adversarial/attack_a2_fewer_blades.py
python3 adversarial/attack_d2_pulse_glide.py
python3 adversarial/feature_attribution.py
```

Each script is self-contained. Random seeds are pinned. Re-running produces byte-identical results (up to CPU-specific floating-point variation).

---

## Repository layout

```
counter-uav-adversarial-radar/
├── paper/
│   ├── preprint.md                  Main paper, Markdown with embedded charts
│   ├── preprint.pdf                 Rendered PDF, 8 pages, arXiv-ready
│   └── figures_preprint/            Figure sources (PNG + Mermaid .mmd)
├── baseline/
│   ├── fmcw_simulation.py           FMCW radar signal generator + spectrogram + BFP
│   ├── model.py                     CNN / CNN+BFP / CNN+LSTM+BFP architectures
│   ├── train_and_evaluate.py        Training + evaluation pipeline
│   ├── leakage_test.py              Diagnostic: randomise per-frame params
│   ├── herm_extractor.py            Alternative HERM-based feature (null result)
│   └── results/                     Baseline experiment JSON outputs
└── adversarial/
    ├── attack_a2_fewer_blades.py    Attack A2: variable blade count x RPM
    ├── attack_d2_pulse_glide.py     Attack D2: variable pulse-and-glide ratio
    ├── feature_attribution.py       Permutation importance + region masking
    ├── FINDINGS_A2.md               Results write-up: A2 null result
    ├── FINDINGS_D2.md               Results write-up: D2 null result
    ├── FINDINGS_attribution.md      Results write-up: what the classifier uses
    ├── *_results.json               Raw experiment outputs
    └── run_log_*.txt                Full stdout logs
```

---

## How to reproduce

Each experiment is a single self-contained script; no arguments required. Expected runtimes are CPU, M-series Mac.

| Script                                   | Runtime | Output                                             |
|:-----------------------------------------|:-------:|:---------------------------------------------------|
| `baseline/train_and_evaluate.py`         | 5-15 min | `baseline/results/experiment_results.json`         |
| `baseline/leakage_test.py`               | 3-10 min | `baseline/results/leakage_test_results.json`       |
| `adversarial/attack_a2_fewer_blades.py`  | 5-10 min | `adversarial/attack_a2_results.json`               |
| `adversarial/attack_d2_pulse_glide.py`   | 10-15 min | `adversarial/attack_d2_results.json`              |
| `adversarial/feature_attribution.py`     | 10-15 min | `adversarial/feature_attribution_results.json`    |

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
3. **Frequency-band mask tests are geometrically confounded.** Bulk-Doppler energy for different classes lives in different Doppler bands, so the "bulk" and "micro" masks cannot cleanly separate the two. A class-conditional mask around each sample's own bulk-Doppler peak would be cleaner; proposed but not run.
4. **No demonstration of a successful attack.** Attribution predicts that attacks targeting bulk-Doppler position (reduced airspeed) or bulk-echo amplitude (radar-absorbent material) should succeed, but these are not run here.

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
