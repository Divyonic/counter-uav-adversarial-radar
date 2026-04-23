# Baseline Counter-UAV Classifier

End-to-end FMCW radar + CNN + LSTM pipeline for drone / bird / fixed-wing UAV / manned aircraft classification.

## Files

| File                          | Purpose                                                        |
|-------------------------------|----------------------------------------------------------------|
| `fmcw_simulation.py`          | Synthetic FMCW signal generation, 2D-FFT, STFT, CFAR, BFP      |
| `model.py`                    | CNN, CNN+BFP, CNN+LSTM+BFP architectures (PyTorch)             |
| `train_and_evaluate.py`       | Full training + evaluation pipeline; writes `results/`          |
| `leakage_test.py`             | Diagnostic: retrains with per-frame randomised parameters      |
| `generate_figures.py`         | Renders figures from `results/experiment_results.json`          |
| `generate_improved_figures.py`| Updated figure rendering                                       |

## Run

```bash
python3 train_and_evaluate.py         # ~5-15 min on CPU
python3 leakage_test.py               # ~3-10 min on CPU
python3 generate_improved_figures.py
```

## FMCW radar parameters

- Centre frequency: 9.5 GHz (X-band)
- Bandwidth: 400 MHz
- Chirp duration: 100 μs
- PRF: 8333 Hz
- 256 samples × 512 chirps per frame
- Range resolution: 0.375 m
- Doppler resolution: 0.51 m/s

## Model parameters

- CNN: 4 conv blocks, ~420K params
- LSTM: 2 layers (64 → 32 units), ~63K params
- Total: ~483K parameters

## Known issues

See top-level `README.md` for a list of known methodological limitations in the baseline paper, including the suspected data-leakage issue that motivated the adversarial-robustness research direction.
