# Counter-UAV Radar: From Classification to Adversarial Robustness

**Author:** Divya Kumar Jitendra Patel (IITM)

This repository documents a research journey in two stages:

1. **Baseline (`/baseline`, `/paper`):** An end-to-end FMCW radar + CNN + LSTM pipeline for multi-class drone/bird/UAV/aircraft classification, targeting counter-UAV applications for Indian defence.
2. **Adversarial gap (`/adversarial`):** A shift in research focus after discovering that physics-informed classifiers can be defeated by cheap physical modifications to the drone. This stage builds a threat taxonomy and robustness benchmark for counter-UAV radar.

---

## The story

I set out to build a classifier that distinguishes multi-rotor drones from birds, fixed-wing UAVs, and manned aircraft, using micro-Doppler signatures from FMCW radar. The final pipeline — documented in `paper/counter_uav_fmcw_cnn_lstm.md` — integrates:

- FMCW chirp generation and beat-signal synthesis
- 2D-FFT Range-Doppler processing
- STFT micro-Doppler spectrograms
- CA-CFAR adaptive detection
- A Blade Flash Periodicity (BFP) physics-informed feature
- CNN + LSTM classifier with temporal tracking

On synthetic data, the full system reaches ~96% classification accuracy with sub-20 ms inference latency on CPU.

While stress-testing this classifier I discovered that its physics-based assumptions can be defeated by cheap, physically realistic modifications to the target drone:

- Reducing propeller blade count (1-blade props with counterweights)
- Slowing rotor RPM to push blade-flash frequency into the bird range
- Wrapping blades in dielectric material to reduce flash amplitude
- Substituting a flapping-wing airframe (ornithopter) for a multi-rotor

These attacks cost between ₹0 and ₹50,000 to implement and break the classifier's assumptions in different ways. Since every published counter-UAV classifier and every fielded Indian counter-UAS system (SAKSHAM, IDD&IS, Indrajaal) relies on similar assumptions, this suggests a systematic vulnerability that has not been publicly characterised.

The new research direction is to build a systematic threat taxonomy for physical adversarial attacks against micro-Doppler classifiers, quantify each attack's effectiveness, and evaluate defences.

---

## Repository layout

```
counter-uav-adversarial-radar/
├── paper/
│   ├── counter_uav_fmcw_cnn_lstm.md   Baseline research paper (technical report)
│   └── figures/                        Paper figures
├── baseline/
│   ├── fmcw_simulation.py              FMCW radar signal generator + spectrogram
│   ├── model.py                        CNN / CNN+BFP / CNN+LSTM+BFP architectures
│   ├── train_and_evaluate.py           End-to-end training and evaluation
│   ├── leakage_test.py                 Diagnostic: tests if LSTM gain is a data-leakage artefact
│   ├── generate_figures.py             Figure rendering from results JSON
│   ├── results/                        Experiment output JSONs
│   └── figures/                        Rendered figures
└── adversarial/
    └── attack_a2_fewer_blades.py       Attack A2: 1-blade drones at variable RPM
```

---

## Status

- [x] Baseline classifier pipeline implemented and trained
- [x] Measured accuracy, FAR, latency, SNR robustness on synthetic data
- [x] Baseline research paper written
- [x] Data-leakage diagnostic script written
- [x] First adversarial attack script (A2: fewer blades) implemented
- [ ] A2 evaluation complete (run in progress)
- [ ] Attack A1: lower RPM
- [ ] Attack A3: low-RCS blades (dielectric wrap)
- [ ] Attack B3: corner reflector decoy
- [ ] Attack D2: pulse-and-glide flight pattern
- [ ] Attack E1: ornithopter micro-Doppler substitution
- [ ] Adversarial training defence evaluation
- [ ] Threat-cost taxonomy paper

---

## Known limitations of the baseline paper

The baseline paper (`paper/counter_uav_fmcw_cnn_lstm.md`) is a work-in-progress preprint. Known issues that are actively being investigated:

1. **Data leakage risk.** CNN-only accuracy sits near chance (~49%) while CNN+LSTM reaches ~96% — a +47 percentage point jump that suggests the LSTM may be exploiting within-sequence parameter consistency rather than learning real temporal structure. `baseline/leakage_test.py` tests this hypothesis by randomising per-frame parameters.
2. **Synthetic data only.** The entire evaluation uses physics-based synthetic FMCW returns. Field validation with real radar hardware is required before any operational claim.
3. **Small dataset.** 1,200 total samples across 4 classes is undersized relative to the 483K-parameter model. Overfitting risk.
4. **BFP feature contradiction.** Current results show BFP helps in one configuration and hurts in another; the feature needs further study.

These limitations motivated the pivot to adversarial robustness research: rather than claiming the classifier is good, the goal becomes characterising the conditions under which it fails.

---

## How to run

Requires Python 3 with PyTorch, NumPy, SciPy, scikit-learn, Matplotlib.

```bash
cd baseline
python3 train_and_evaluate.py        # Train and evaluate baseline CNN / CNN+BFP / CNN+LSTM+BFP
python3 leakage_test.py              # Diagnostic: randomise per-frame params, retrain, compare
python3 generate_improved_figures.py # Render figures from results/

cd ../adversarial
python3 attack_a2_fewer_blades.py    # Attack A2: 1-blade drones at variable RPM
```

Each script is self-contained. Random seeds are pinned for reproducibility.

---

## License

MIT — see `LICENSE`.

---

## Contact

Divya Kumar Jitendra Patel
divyakumarpatel202@gmail.com
