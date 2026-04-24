# Counter-UAV Radar: Attribution-First Adversarial Evaluation

**Author:** Divya Kumar Patel, Indian Institute of Technology, Madras
**Contact:** `divyakumarpatel202@gmail.com`

This repository documents a single case study on how counter-UAV radar ML classifiers should be evaluated for adversarial robustness. We start from a representative CNN + LSTM + physics-informed classifier, run two physically motivated attacks, find null results, and then show through feature attribution that the attacks were targeting features the classifier never used.

The central finding is a methodological one: *adversarial evaluation without feature attribution is unreliable.* Null results are uninterpretable in isolation, they can mean the classifier is robust, or they can mean the attack disturbs features the classifier ignores. The two are indistinguishable without attribution, and this repository shows why.

---

## The short version

1. We build a CNN + LSTM + BFP classifier for 4-class FMCW radar target classification (drone, bird, fixed-wing UAV, manned aircraft). On synthetic data it reaches ~89% accuracy.
2. We attack it two ways. **A2** reduces the propeller blade count (pushes blade-flash frequency toward the bird range). **D2** forces the drone into pulse-and-glide flight (removes propeller content from most frames). Both attacks keep accuracy within the baseline band.
3. We then run permutation-importance and region-masking tests on the classifier. Results: the LSTM ignores frame order, half the time axis is redundant, the BFP feature is class-correlated noise rather than a physics measurement, and the classifier's accuracy reduces to a bulk-Doppler peak locator.
4. The null results on A2 and D2 now have a clear explanation: both attacks modified features the classifier does not use. They are evidence about the attacks, not about the classifier's robustness.
5. We propose an attribution-first workflow (attribution → attack design → evaluation) as the minimum credible protocol for adversarial evaluation of radar ML.

The full write-up is in [`paper/preprint.md`](paper/preprint.md).

---

## Repository layout

```
counter-uav-adversarial-radar/
├── paper/
│   ├── preprint.md                       Main paper, attribution-first methodology argument
│   ├── preprint.pdf                      Rendered PDF of the preprint
│   └── figures_preprint/                 Figure sources (Mermaid .mmd + PNG) for the PDF
├── baseline/
│   ├── fmcw_simulation.py                FMCW radar signal generator, spectrogram, BFP extractor
│   ├── model.py                          CNN, CNN+BFP, CNN+LSTM+BFP architectures
│   ├── train_and_evaluate.py             Training and evaluation pipeline
│   ├── leakage_test.py                   Diagnostic: randomises per-frame params to test LSTM is temporal
│   ├── herm_extractor.py                 Alternative HERM-based feature extractor (null result)
│   └── results/                          Baseline experiment JSON outputs
└── adversarial/
    ├── README.md                         Experiment-by-experiment catalogue
    ├── attack_a2_fewer_blades.py         Attack A2: variable blade count × RPM
    ├── attack_d2_pulse_glide.py          Attack D2: variable pulse-and-glide ratio
    ├── feature_attribution.py            Permutation importance + region masking
    ├── FINDINGS_A2.md                    Results write-up: A2 null result
    ├── FINDINGS_D2.md                    Results write-up: D2 null result
    ├── FINDINGS_attribution.md           Results write-up: what the classifier actually uses
    └── *_results.json, run_log_*.txt     Raw experiment outputs
```

---

## How to reproduce

Python 3, PyTorch, NumPy, SciPy, scikit-learn, Matplotlib. All scripts self-contained, random seeds pinned.

```bash
# 1. Baseline classifier
cd baseline
python3 train_and_evaluate.py       # trains CNN / CNN+BFP / CNN+LSTM+BFP, prints accuracy
python3 leakage_test.py             # tests whether LSTM gain is genuine temporal structure

# 2. Adversarial attacks (both produce null results)
cd ../adversarial
python3 attack_a2_fewer_blades.py   # A2 across 6 blade-count × RPM variants
python3 attack_d2_pulse_glide.py    # D2 across 9 glide-ratio variants

# 3. Feature attribution (explains the null results)
python3 feature_attribution.py      # permutation importance + region masks
```

Each run re-trains the baseline model from scratch; there is no pre-trained checkpoint commit to rely on. Expected total runtime on a modern CPU: ~45 minutes.

---

## Status

| Component                              | State                                                      |
|:---------------------------------------|:-----------------------------------------------------------|
| Baseline classifier                    | Implemented, trained, results in `baseline/results/`       |
| Data-leakage diagnostic                | Run, shows LSTM gain is multi-instance not temporal        |
| HERM-line alternative feature          | Implemented, underperforms BFP, negative result           |
| Attack A2 (fewer blades)               | Run across 2 seeds, null result documented                 |
| Attack D2 (pulse-and-glide)            | Run across 9 glide ratios, null result documented          |
| Feature attribution                    | Run, 6 tests, results reconcile A2 and D2                  |
| Preprint                               | Written, `paper/preprint.md`                              |
| Real-data validation (Karlsson 77 GHz) | Not attempted, flagged as next step                       |
| Successful attack (B1 amplitude / D1 bird-speed) | Not attempted, predicted by attribution           |

The repository is archived in its current state. The preprint is a short working paper, not a peer-reviewed publication. Readers interested in real-data validation or a constructive adversarial attack should consider those open directions.

---

## Known limitations

- **Synthetic data only.** The entire evaluation uses physics-based simulated FMCW returns. The critique is pointed at a failure mode that can occur under balanced-class synthetic training; it is not yet validated on real radar data.
- **Single classifier architecture.** Findings apply directly to the CNN + LSTM + BFP baseline here. Generalisation to other published counter-UAV ML architectures is suggested but not proved.
- **Frequency-band mask tests are geometrically confounded.** At 9.5 GHz with the PRF and bin layout used, bulk-Doppler energy for different classes lives in different Doppler bands, so the "bulk" and "micro" masks cannot cleanly isolate the two. A class-conditional mask around each sample's own bulk-Doppler peak would be cleaner.
- **No demonstration of a successful attack.** Attribution predicts attacks on bulk-Doppler position (reduced airspeed) or bulk-echo amplitude (radar-absorbent material) would succeed. These are proposed but not run.

---

## Citation

If this work is useful, please cite:

```
Divya Kumar Patel (2026). Feature-Attribution-First Adversarial Evaluation
of a Physics-Informed Counter-UAV Radar Classifier. Working preprint,
Indian Institute of Technology Madras.
https://github.com/Divyonic/counter-uav-adversarial-radar
```

---

## License

MIT, see [`LICENSE`](LICENSE).

## Contact

Divya Kumar Patel
`divyakumarpatel202@gmail.com`
