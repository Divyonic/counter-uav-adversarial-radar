# Baseline counter-UAV classifier

End-to-end FMCW radar + CNN + LSTM pipeline for a 4-class classification task (drone / bird / fixed-wing UAV / manned aircraft). This is the classifier that the adversarial experiments in [`../adversarial/`](../adversarial/) probe.

For the methodology context see the top-level [README](../README.md) and [`../paper/preprint.md`](../paper/preprint.md).

---

## Files

| File                          | Purpose                                                           |
|:------------------------------|:------------------------------------------------------------------|
| [`fmcw_simulation.py`](fmcw_simulation.py)       | FMCW signal synthesis, 2D-FFT, STFT, CFAR, BFP extractor |
| [`model.py`](model.py)                           | CNN, CNN+BFP, CNN+LSTM+BFP PyTorch architectures          |
| [`train_and_evaluate.py`](train_and_evaluate.py) | Training and evaluation pipeline, writes `results/`       |
| [`leakage_test.py`](leakage_test.py)             | Diagnostic: retrains with per-frame randomised parameters |
| [`herm_extractor.py`](herm_extractor.py)         | Alternative HERM-based feature extractor (null result)    |
| [`results/`](results/)                           | Baseline experiment JSON outputs                          |

---

## Run

```bash
python3 train_and_evaluate.py     # 5-15 min on CPU
python3 leakage_test.py           # 3-10 min on CPU
```

Each script writes results into [`results/`](results/). Random seeds are pinned for reproducibility.

---

## Radar parameters

| Parameter            | Value             |
|:---------------------|:------------------|
| Centre frequency     | 9.5 GHz (X-band)  |
| Bandwidth            | 400 MHz           |
| Chirp duration       | 100 &mu;s         |
| PRF                  | 8333 Hz           |
| Samples per chirp    | 256               |
| Chirps per frame     | 512               |
| Range resolution     | 0.375 m           |
| Doppler resolution   | 0.51 m/s          |

## Model parameters

| Component | Details                              |
|:----------|:-------------------------------------|
| CNN       | 4 conv blocks, ~420K parameters      |
| LSTM      | 2 layers (64 &rarr; 32 units), ~63K |
| **Total** | **~483K parameters**                 |

---

## Known issues

See the top-level [README](../README.md#known-limitations) for the methodological limitations that motivated the adversarial-robustness direction. In brief:

- LSTM gain over the CNN baseline comes from multi-instance aggregation over within-class parameter consistency, not from temporal structure. Confirmed by `leakage_test.py`.
- The BFP feature's autocorrelation-based extractor returns a narrow ~30-45 Hz band regardless of the physical ground truth (13-167 Hz). Confirmed in [`../adversarial/FINDINGS_A2.md`](../adversarial/FINDINGS_A2.md).
- Performance is characterised on physics-based synthetic returns only; no real-radar validation.
