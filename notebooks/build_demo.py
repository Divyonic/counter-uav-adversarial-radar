"""Build `demo.ipynb` programmatically via nbformat."""

import nbformat as nbf

nb = nbf.v4.new_notebook()

nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

cells = []


def md(src: str):
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str):
    cells.append(nbf.v4.new_code_cell(src))


md("""# Counter-UAV Adversarial Evaluation: Interactive Walk-through

This notebook walks through the paper's case study end-to-end in one browsable document. It renders live radar-signal simulations, shows the measured-vs-ground-truth BFP failure, then loads the cached experimental results and renders the attack and attribution charts.

Runs in under a minute on any laptop. No GPU required.

**Reading order**

1. Setup
2. A quick look at the simulated radar data
3. The BFP-is-noise diagnostic
4. Baseline classifier performance
5. Attack A2 (null result)
6. Attack D2 (null result)
7. Feature attribution (the reveal)
8. What it means

---
""")

md("""## 1. Setup""")

code("""import os, sys, json
import numpy as np
import matplotlib.pyplot as plt

# Notebook is in `notebooks/`; code is one level up.
HERE = os.path.abspath(".")
REPO = os.path.abspath("..")
sys.path.insert(0, os.path.join(REPO, "baseline"))

np.random.seed(42)
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#eef0f3",
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
})
print("repo:", REPO)""")

md("""## 2. A quick look at the simulated radar data

The simulator generates FMCW radar returns for four target classes. Let's look at one example of each and the micro-Doppler spectrograms the CNN would see.""")

code("""from fmcw_simulation import (
    generate_drone_signal,
    generate_bird_signal,
    generate_friendly_uav_signal,
    generate_aircraft_signal,
    compute_spectrogram,
)

classes = {
    "Drone (2-blade, 5000 RPM)": lambda: generate_drone_signal(
        R0=1000, v_bulk=12, snr_db=20, n_blades=2, n_props=4,
        rpm=5000, blade_len=0.12, tilt_angle=45),
    "Bird (8 Hz flap)": lambda: generate_bird_signal(
        R0=800, v_bulk=10, snr_db=20, flap_freq=8, wingspan=0.5),
    "Fixed-wing UAV": lambda: generate_friendly_uav_signal(
        R0=900, v_bulk=25, snr_db=20, n_blades=2, rpm=3500, blade_len=0.2),
    "Aircraft": lambda: generate_aircraft_signal(
        R0=3000, v_bulk=80, snr_db=20),
}

fig, axes = plt.subplots(1, 4, figsize=(16, 3.4))
for ax, (name, fn) in zip(axes, classes.items()):
    beat = fn()
    spec, _, _ = compute_spectrogram(beat)
    ax.imshow(10 * np.log10(spec + 1e-9), origin="lower",
              aspect="auto", cmap="magma")
    ax.set_title(name, fontsize=11)
    ax.set_xlabel("time bin")
    ax.set_ylabel("Doppler bin")
    ax.grid(False)
plt.suptitle("Micro-Doppler spectrograms (128 × 128, SNR 20 dB)",
             y=1.02, fontsize=12)
plt.tight_layout()
plt.show()""")

md("""The drone has fine horizontal structure from the rotating blades, the bird a slower periodic flap, the fixed-wing a weaker propeller track, and the aircraft almost no micro-Doppler on top of its strong bulk-Doppler line. These are the patterns the classifier is supposed to be reading.""")

md("""## 3. The BFP-is-noise diagnostic

The "physics-informed" Blade Flash Periodicity (BFP) feature is supposed to measure the blade-flash fundamental frequency, which depends on rotor RPM and blade count. Let's vary those parameters over a wide range and see what the BFP extractor actually returns.""")

code("""from fmcw_simulation import extract_bfp_features

scenarios = [
    ("2-blade 5000 RPM",   dict(n_blades=2, rpm=5000, blade_len=0.12)),
    ("1-blade 5000 RPM",   dict(n_blades=1, rpm=5000, blade_len=0.12)),
    ("1-blade 3000 RPM",   dict(n_blades=1, rpm=3000, blade_len=0.12)),
    ("1-blade 2000 RPM",   dict(n_blades=1, rpm=2000, blade_len=0.12)),
    ("1-blade 1200 RPM",   dict(n_blades=1, rpm=1200, blade_len=0.12)),
    ("1-blade 800 RPM",    dict(n_blades=1, rpm=800,  blade_len=0.12)),
]

ground_truth, measured = [], []
for label, params in scenarios:
    gt = params["n_blades"] * params["rpm"] / 60.0
    bfps = []
    for _ in range(20):
        beat = generate_drone_signal(
            R0=np.random.uniform(500, 2000),
            v_bulk=np.random.uniform(5, 20),
            snr_db=15, n_props=4, tilt_angle=45, **params)
        spec, f, t = compute_spectrogram(beat)
        fs = len(t) / (t[-1] - t[0])
        bfps.append(extract_bfp_features(spec, fs)[0])
    bfps = np.array(bfps)
    ground_truth.append(gt)
    measured.append(bfps.mean())
    print(f"{label:20s}  ground-truth BFP = {gt:6.1f} Hz   "
          f"measured = {bfps.mean():5.1f} ± {bfps.std():5.1f} Hz")""")

code("""labels = [s[0] for s in scenarios]
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(x - 0.18, ground_truth, width=0.36, color="#16a34a",
       label="Ground truth", zorder=2)
ax.bar(x + 0.18, measured, width=0.36, color="#64748b",
       label="Measured (mean)", zorder=2)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=25, ha="right")
ax.set_ylabel("Blade-flash frequency (Hz)")
ax.set_title("BFP extractor returns noise regardless of ground truth")
ax.legend(frameon=False, loc="upper right")
ax.set_ylim(0, 200)
plt.tight_layout()
plt.show()""")

md("""The ground-truth BFP spans 13 Hz (single blade at 800 RPM) through 167 Hz (2 blades at 5000 RPM), a 13x range. The measured BFP clusters in a narrow 30 to 45 Hz band *regardless* of the physical ground truth. The feature that is supposed to measure blade-flash frequency does not.

This is the first clue that **a feature's presence in the architecture does not imply the classifier uses it physically**.""")

md("""## 4. Baseline classifier performance

Rather than retrain from scratch inside this notebook (which takes 5 to 15 minutes), we load the results already committed in the repo.""")

code("""with open(os.path.join(REPO, "baseline/results/experiment_results.json")) as f:
    baseline = json.load(f)

for k, v in baseline.items():
    print(f"{k}: {v}")""")

md("""## 5. Attack A2 — fewer blades (null result)

Six variants spanning the gradient from clean 2-blade to 1-blade at 800 RPM (BFP at typical bird-flap frequency). Hypothesis: pushing BFP into the bird range should collapse drone-class accuracy.""")

code("""with open(os.path.join(REPO, "adversarial/attack_a2_results.json")) as f:
    a2 = json.load(f)

variants = [a["attack_name"] for a in a2["attacks"]]
accs = [a["accuracy_as_drone"] * 100 for a in a2["attacks"]]
exp_bfp = [a["expected_bfp_hz"] for a in a2["attacks"]]
meas_bfp = [a["measured_bfp_hz_mean"] for a in a2["attacks"]]

for v, acc, ebfp, mbfp in zip(variants, accs, exp_bfp, meas_bfp):
    print(f"{v:25s}  acc={acc:5.1f}%   "
          f"expected BFP={ebfp:6.1f} Hz   measured={mbfp:5.1f} Hz")""")

code("""fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar(range(len(variants)), accs, color="#2563eb")
ax.axhspan(79, 91, alpha=0.1, color="#fb923c",
           label="baseline band")
ax.axhline(a2["baseline_clean_test_accuracy"] * 100,
           color="#f97316", linestyle="--", alpha=0.6,
           label="clean baseline")
ax.set_xticks(range(len(variants)))
ax.set_xticklabels([v.replace("_", " ") for v in variants],
                   rotation=25, ha="right")
ax.set_ylim(0, 100)
ax.set_ylabel("Drone-class accuracy (%)")
ax.set_title("A2: accuracy flat across six variants")
ax.legend(frameon=False, loc="lower right")
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width()/2, a + 1, f"{a:.0f}",
            ha="center", fontsize=10)
plt.tight_layout()
plt.show()""")

md("""Accuracy stays in an 79-91% band regardless of how aggressively we push the blade flash into bird range. **Null result.** The obvious reading is "classifier is robust to blade-count attacks." That reading is about to turn out to be wrong.""")

md("""## 6. Attack D2 — pulse-and-glide (null result, stronger)

Instead of changing blade physics, this attack removes propeller content from the signal itself. Glide ratio is the fraction of frames in each 10-frame LSTM window that contain only body echo (no propellers spinning). At glide ratio 1.0, the classifier sees zero propeller content anywhere in the input.""")

code("""with open(os.path.join(REPO, "adversarial/attack_d2_results.json")) as f:
    d2 = json.load(f)

ratios = [a["glide_ratio"] for a in d2["attacks"]]
accs = [a["accuracy_as_drone"] * 100 for a in d2["attacks"]]
for r, a in zip(ratios, accs):
    print(f"glide ratio {int(r*100):3d}%  →  drone accuracy {a:5.1f}%")""")

code("""fig, ax = plt.subplots(figsize=(9, 4))
bars = ax.bar([f"{int(r*100)}%" for r in ratios], accs, color="#dc2626")
ax.axhspan(80, 89.3, alpha=0.1, color="#fb923c",
           label="baseline band")
ax.set_ylim(0, 100)
ax.set_xlabel("Glide ratio")
ax.set_ylabel("Drone-class accuracy (%)")
ax.set_title("D2: accuracy flat across glide ratios (100% glide → 81% drone)")
ax.legend(frameon=False, loc="lower right")
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width()/2, a + 1, f"{a:.0f}",
            ha="center", fontsize=10)
plt.tight_layout()
plt.show()""")

md("""At 100% glide, with literally no propeller content in any frame, the classifier still returns drone 81% of the time. This is much stronger than A2: we aren't just modifying the physics underneath a feature, we are deleting the signal the classifier is supposedly reading.

If both attacks fail, and both attacks targeted what the architecture is described as analysing, then the architecture is not doing what it says on the tin. Time to measure what it actually uses.""")

md("""## 7. Feature attribution — the reveal

Six perturbation tests on the trained classifier. Each test perturbs one feature group and measures the accuracy drop. Large drop = important feature. Small drop = the classifier does not use that feature.""")

code("""with open(os.path.join(REPO, "adversarial/feature_attribution_results.json")) as f:
    attr = json.load(f)

baseline_acc = attr["baseline_clean_accuracy"]
tests = attr["tests"]
rows = []
for key, data in tests.items():
    rows.append((data["test"], data["mean_acc"],
                 data["accuracy_drop_pp"]))
rows.sort(key=lambda r: r[2])

print(f"Clean baseline accuracy: {baseline_acc:.4f}\\n")
print(f"{'Test':<48} {'Masked acc':<12} {'Drop (pp)':<10}")
print("-" * 70)
for name, acc, drop in rows:
    print(f"{name[:47]:<48} {acc:<12.4f} {drop:+.2f}")""")

code("""labels = [r[0] for r in rows]
drops = [r[2] for r in rows]
colors = ["#cbd5e1" if d < 5 else
          "#f59e0b" if d < 25 else
          "#ef4444" if d < 40 else
          "#991b1b" for d in drops]

fig, ax = plt.subplots(figsize=(9, 4.4))
y = np.arange(len(labels))
bars = ax.barh(y, drops, color=colors)
ax.axvline(0, color="#666", linewidth=1.5)
ax.set_yticks(y)
ax.set_yticklabels([l[:42] for l in labels])
ax.set_xlabel("Accuracy drop (percentage points)")
ax.set_title("What the classifier actually uses")
ax.set_xlim(-5, 50)
for b, d in zip(bars, drops):
    ax.text(d + (1 if d >= 0 else -1), b.get_y() + b.get_height()/2,
            f"{d:+.1f}", va="center",
            ha="left" if d >= 0 else "right", fontsize=10)
plt.tight_layout()
plt.show()""")

md("""Read top-to-bottom:

- **Frame-order shuffle (−1.6 pp)** the LSTM doesn't use temporal order
- **Temporal mask (0.0 pp)** half the time axis is redundant
- **Central 25% frequency mask (+16 pp)** low-Doppler matters but not hugely
- **Outer 50% frequency mask (+33 pp)** mid-to-high Doppler band is load-bearing
- **BFP permutation (+38 pp)** BFP vector is load-bearing, but as a class-correlated distribution (see section 3: its values are noise)
- **Spectrogram permutation (+41 pp)** spectrogram content is the single biggest contributor

The picture resolving A2 and D2: the classifier identifies drones by **the position and amplitude of the bulk-Doppler peak in the spectrogram**, not by blade-flash harmonics or temporal evolution. A2 changed the blade physics while leaving the bulk-Doppler peak intact. D2 removed the blades but kept the body echo at drone-typical bulk-Doppler velocity. Neither attack touched what the classifier was actually reading.""")

md("""## 8. Class-conditional bulk-Doppler mask

The fixed-band mask tests above are geometrically confounded — the bulk-Doppler peak sits at different absolute frequency bins for different classes. A cleaner test masks ±N bins around *each sample's own* peak-power frequency bin.""")

code("""with open(os.path.join(REPO, "adversarial/feature_attribution_class_conditional_results.json")) as f:
    cc = json.load(f)

baseline_cc = cc["baseline_clean_accuracy"]
print(f"Clean baseline: {baseline_cc:.4f}")
print(f"{'Half-width':<12} {'Masked acc':<12} {'Drop (pp)':<10}")
print("-" * 36)
for r in cc["tests"]:
    print(f"±{r['half_width_bins']:<11} {r['mean_acc']:<12.4f} +{r['accuracy_drop_pp']:.2f}")""")

code("""hws = [r["half_width_bins"] for r in cc["tests"]]
drops = [r["accuracy_drop_pp"] for r in cc["tests"]]

fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(hws, drops, marker="o", markersize=12, linewidth=2.5,
         color="#dc2626", markerfacecolor="white", markeredgewidth=2)
ax.axhline(16.0, color="#3b82f6", linestyle=":", linewidth=2,
            label="Fixed central 25% mask (+16.0 pp)")
ax.axhline(33.3, color="#9333ea", linestyle=":", linewidth=2,
            label="Fixed outer 50% mask (+33.3 pp)")
ax.set_xticks(hws)
ax.set_xlabel("Mask half-width (bins around each sample's own peak)")
ax.set_ylabel("Accuracy drop (pp)")
ax.set_title("Class-conditional mask is more efficient per masked bin")
ax.legend(loc="lower right", frameon=False)
for hw, d in zip(hws, drops):
    ax.annotate(f"+{d:.1f}", (hw, d), textcoords="offset points",
                  xytext=(0, 10), ha="center", fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""Masking just **±1 frequency bin (2.3% of the axis) at each sample's peak drops accuracy by 20.8 pp** — more than the fixed central-25% mask did with 10× more masking. This is the cleanest demonstration in the repo that the load-bearing feature is the bulk-Doppler peak position+shape.""")

md("""## 9. Attack D1 — bird-speed flight (the attack that succeeds)

If the classifier reads bulk-Doppler peak position, then a drone flown slowly enough that its bulk-Doppler peak overlaps the bird-class velocity distribution should be classified as a bird. A drone flown at the upper end of the drone training distribution (where it overlaps the friendly UAV class) should be classified as friendly UAV.""")

code("""with open(os.path.join(REPO, "adversarial/attack_d1_results.json")) as f:
    d1 = json.load(f)

print(f"{'Velocity (m/s)':<18} {'Drone acc':<12} {'Dominant class'}")
print("-" * 60)
for r in d1["attacks"]:
    cd = r["class_distribution"]
    dominant = max(cd.items(), key=lambda kv: kv[1])
    label = f"{r['v_lo_mps']:.0f}-{r['v_hi_mps']:.0f}"
    print(f"{label:<18} {r['accuracy_as_drone']:<12.3f} "
          f"{dominant[0]} ({dominant[1]}/{r['n_samples']})")""")

code("""labels = [f"{r['v_lo_mps']:.0f}-{r['v_hi_mps']:.0f}" for r in d1["attacks"]]
drone_pcts = {cls: [] for cls in
              ["Enemy Drone", "Bird", "Friendly UAV", "Manned Aircraft"]}
for r in d1["attacks"]:
    n = r["n_samples"]
    for cls in drone_pcts:
        drone_pcts[cls].append(r["class_distribution"][cls] / n * 100)

class_colours = {"Enemy Drone": "#dc2626", "Bird": "#16a34a",
                  "Friendly UAV": "#2563eb", "Manned Aircraft": "#9333ea"}

fig, ax = plt.subplots(figsize=(10, 4.6))
bottoms = np.zeros(len(labels))
for cls, pcts in drone_pcts.items():
    pcts_arr = np.array(pcts)
    ax.bar(labels, pcts_arr, bottom=bottoms, label=cls,
            color=class_colours[cls], edgecolor="white", linewidth=0.8)
    bottoms = bottoms + pcts_arr
ax.set_ylim(0, 100)
ax.set_xlabel("Drone bulk velocity v_bulk (m/s)")
ax.set_ylabel("Class-prediction share (%)")
ax.set_title("D1: drone classification collapses outside the 8-15 m/s window")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4,
           frameon=False)
plt.tight_layout()
plt.show()""")

md("""Two distinct failure modes appear:

- **Slow drones (≤ 12 m/s) read as bird.** At v_bulk = 5–10 m/s, drone classification drops to 45%; at 3–5 m/s, 78% of the fleet reads as bird.
- **Fast drones (15–20 m/s) read as friendly UAV.** 149/150 — essentially complete confusion. Friendly-coded targets are explicitly *not* engaged by counter-UAV systems.

Both misclassifications require **no hardware modification**: only a flight-controller change. This is the attribution-predicted attack, and it works exactly as predicted.""")

md("""## 10. Attack B1 — RAM-wrap (the informative null)

If amplitude were also load-bearing, attenuating the drone's radar return should also defeat the classifier. We sweep 0–20 dB of attenuation. It does not.""")

code("""with open(os.path.join(REPO, "adversarial/attack_b1_results.json")) as f:
    b1 = json.load(f)

print(f"{'RCS drop':<10} {'Eff SNR':<10} {'Drone acc'}")
print("-" * 30)
for r in b1["attacks"]:
    print(f"{r['rcs_drop_db']:<10.0f} {r['effective_snr_db']:<10.0f} "
          f"{r['accuracy_as_drone']:.3f}")""")

code("""drops = [r["rcs_drop_db"] for r in b1["attacks"]]
accs = [r["accuracy_as_drone"] * 100 for r in b1["attacks"]]
baseline_b1 = b1["baseline_clean_test_accuracy"] * 100

fig, ax = plt.subplots(figsize=(9, 4.4))
ax.axhspan(baseline_b1 - 5, baseline_b1 + 5, color="#10b981", alpha=0.15,
            label="Baseline ±5 pp")
ax.plot(drops, accs, marker="o", markersize=12, linewidth=2.5,
         color="#dc2626", markerfacecolor="white", markeredgewidth=2)
ax.set_xticks(drops)
ax.set_ylim(0, 100)
ax.set_xlabel("Bulk-amplitude attenuation (dB)")
ax.set_ylabel("Drone-class accuracy (%)")
ax.set_title("B1: amplitude attacks bounce off the preprocessing layer")
ax.legend(loc="lower left", frameon=False)
for d, a in zip(drops, accs):
    ax.annotate(f"{a:.0f}%", (d, a), textcoords="offset points",
                  xytext=(0, 10), ha="center", fontweight="bold")
plt.tight_layout()
plt.show()""")

md("""Drone classification stays in the 77–97% band across the entire 0–20 dB sweep, including effective SNR of −5 dB.

The reason is the preprocessing pipeline. `compute_spectrogram` → `resize_spectrogram` does dB clipping to a 40 dB dynamic range and per-sample [0, 1] normalisation, which **discards absolute amplitude before the classifier sees the input**. So the classifier reads bulk-Doppler peak *position* and *post-normalised relative shape* — not absolute amplitude. B1 is therefore a CFAR/detection-stage attack rather than a classifier-input attack.

This sharpens the attribution claim. The methodology argument now has both a successful attack (D1) and an informative null (B1), each interpretable only because the attribution work came first.""")

md("""## 11. What it means

The complete picture:

- Six attacks from the threat taxonomy, plus two attribution runs.
- Four attacks against the architecture's advertised physics (A1, A2, D2, E1) all null. Easy to misread as robustness.
- Attribution shows what the classifier actually reads: bulk-Doppler peak position+shape. The class-conditional mask makes this surgical.
- D1 (bird-speed flight, attribution-driven) **succeeds completely** — drone classification collapses to 0–45% with no hardware change.
- B1 (RAM-wrap, attribution-driven) is null because preprocessing absorbs amplitude. This is interpretable only with the attribution context.

**Without the attribution run, the four pre-attribution nulls would have read as robustness.** They are not. The attribution-driven attack against what the classifier actually uses is catastrophic.

The methodology correction is small: **run permutation importance and region masking before designing the attacks, so you know what you are aiming at.** The implication is not: counter-UAV radar is safety-critical sensing, and the published robustness claims that lean on null results deserve a much closer audit.

For the full argument see [`../paper/preprint.pdf`](../paper/preprint.pdf) or [`../paper/preprint.md`](../paper/preprint.md). For the concrete experiment code that generated these JSON files, see [`../adversarial/`](../adversarial/).

---

*Divya Kumar Jitendra Patel, IIT Madras, April 2026 (revision 2).*
""")

nb.cells = cells

import os
out = os.path.join(os.path.dirname(__file__), "demo.ipynb")
with open(out, "w") as f:
    nbf.write(nb, f)
print(f"wrote {out} with {len(cells)} cells")
