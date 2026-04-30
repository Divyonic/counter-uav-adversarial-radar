"""
Generate publication-quality figures for the new experiments (B1, D1,
class-conditional mask) added in the second methodology revision.

Reads the JSON outputs in adversarial/results/ and produces PNG figures
sized 1800 x 960 (matching the existing fig2..fig5 layout).
"""

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = ROOT / "adversarial"
OUT_DIR = Path(__file__).resolve().parent

CLASS_COLOURS = {
    "Enemy Drone": "#dc2626",     # red
    "Bird": "#16a34a",             # green
    "Friendly UAV": "#2563eb",    # blue
    "Manned Aircraft": "#9333ea",  # purple
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 14,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
})


def fig_d1():
    with open(RESULTS_DIR / "attack_d1_results.json") as f:
        data = json.load(f)
    baseline = data["baseline_clean_test_accuracy"] * 100

    rows = data["attacks"]
    labels = [f"{r['v_lo_mps']:.0f}–{r['v_hi_mps']:.0f}" for r in rows]
    drone_acc = [r["accuracy_as_drone"] * 100 for r in rows]
    n = len(rows)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 9.6), dpi=100,
                                    gridspec_kw={"width_ratios": [1.1, 1.0]})

    # Left panel: drone accuracy vs velocity window
    bar_x = np.arange(n)
    colours = ["#dc2626" if a >= 50 else "#fb923c" if a >= 30 else "#16a34a"
                for a in drone_acc]
    ax1.bar(bar_x, drone_acc, color=colours, width=0.7, edgecolor="white",
             linewidth=1.2)
    ax1.axhline(baseline, color="#475569", linestyle="--", linewidth=1.5,
                 label=f"Clean baseline ({baseline:.1f}%)")
    ax1.set_xticks(bar_x)
    ax1.set_xticklabels(labels)
    ax1.set_xlabel("Drone bulk velocity v_bulk (m/s)")
    ax1.set_ylabel("Drone-class accuracy (%)")
    ax1.set_title("D1: Drone accuracy collapses at every off-band velocity")
    ax1.set_ylim(0, 100)
    ax1.legend(loc="upper right")
    for i, a in enumerate(drone_acc):
        ax1.text(i, a + 2, f"{a:.0f}%", ha="center", va="bottom",
                  fontsize=11, fontweight="bold")

    # Right panel: stacked confusion-class distribution
    classes = ["Enemy Drone", "Bird", "Friendly UAV", "Manned Aircraft"]
    bottoms = np.zeros(n)
    for cls in classes:
        counts = np.array([r["class_distribution"][cls] for r in rows])
        pcts = counts / np.array([r["n_samples"] for r in rows]) * 100
        ax2.bar(bar_x, pcts, bottom=bottoms, color=CLASS_COLOURS[cls],
                 width=0.7, label=cls, edgecolor="white", linewidth=0.8)
        bottoms = bottoms + pcts
    ax2.set_xticks(bar_x)
    ax2.set_xticklabels(labels)
    ax2.set_xlabel("Drone bulk velocity v_bulk (m/s)")
    ax2.set_ylabel("Class-prediction share (%)")
    ax2.set_title("D1: At v=15–20 m/s, drone reads as friendly UAV")
    ax2.set_ylim(0, 100)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, -0.18), ncol=4,
                frameon=False)

    plt.tight_layout()
    out = OUT_DIR / "fig8_d1.png"
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out}")


def fig_b1():
    with open(RESULTS_DIR / "attack_b1_results.json") as f:
        data = json.load(f)
    baseline = data["baseline_clean_test_accuracy"] * 100

    rows = data["attacks"]
    drops = np.array([r["rcs_drop_db"] for r in rows])
    eff_snr = np.array([r["effective_snr_db"] for r in rows])
    drone_acc = np.array([r["accuracy_as_drone"] * 100 for r in rows])

    fig, ax = plt.subplots(figsize=(18.0, 9.6), dpi=100)

    # Shaded baseline band ±5 pp
    ax.axhspan(baseline - 5, baseline + 5, color="#10b981", alpha=0.15,
                label=f"Baseline ±5 pp band")
    ax.axhline(baseline, color="#475569", linestyle="--", linewidth=1.5,
                label=f"Clean baseline ({baseline:.1f}%)")

    ax.plot(drops, drone_acc, marker="o", markersize=12,
             linewidth=3, color="#dc2626",
             markerfacecolor="white", markeredgewidth=2.5,
             markeredgecolor="#dc2626")

    for d, a, snr in zip(drops, drone_acc, eff_snr):
        ax.annotate(f"{a:.0f}%\nSNR={snr:.0f} dB",
                     (d, a), textcoords="offset points",
                     xytext=(0, 14), ha="center", fontsize=10)

    ax.set_xlabel("Bulk-amplitude attenuation (dB)")
    ax.set_ylabel("Drone-class accuracy (%)")
    ax.set_title("B1: Per-sample normalisation absorbs amplitude attacks "
                  "across the full 0–20 dB sweep", pad=15)
    ax.set_ylim(0, 100)
    ax.set_xticks(drops)
    ax.legend(loc="lower left", framealpha=0.9)

    plt.tight_layout()
    out = OUT_DIR / "fig9_b1.png"
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out}")


def fig_classcond():
    with open(RESULTS_DIR / "feature_attribution_class_conditional_results.json") as f:
        data = json.load(f)
    baseline = data["baseline_clean_accuracy"] * 100

    rows = data["tests"]
    half_widths = np.array([r["half_width_bins"] for r in rows])
    masked_acc = np.array([r["mean_acc"] * 100 for r in rows])
    drops_pp = np.array([r["accuracy_drop_pp"] for r in rows])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.0, 9.6), dpi=100)

    # Left: drop in accuracy as half-width grows
    ax1.plot(half_widths, drops_pp, marker="o", markersize=14,
             linewidth=3, color="#dc2626",
             markerfacecolor="white", markeredgewidth=2.5,
             markeredgecolor="#dc2626", label="Class-conditional mask")

    # Reference points from fixed-band masks
    ax1.axhline(16.0, color="#3b82f6", linestyle=":", linewidth=2,
                 label="Fixed central 25% mask (+16.0 pp)")
    ax1.axhline(33.3, color="#9333ea", linestyle=":", linewidth=2,
                 label="Fixed outer 50% mask (+33.3 pp)")

    for hw, d in zip(half_widths, drops_pp):
        ax1.annotate(f"+{d:.1f} pp", (hw, d), textcoords="offset points",
                      xytext=(0, 14), ha="center", fontsize=11,
                      fontweight="bold")

    ax1.set_xlabel("Mask half-width (bins around each sample's own peak)")
    ax1.set_ylabel("Accuracy drop (pp)")
    ax1.set_title("Class-conditional mask is more efficient per masked bin",
                   pad=15)
    ax1.set_xticks(half_widths)
    ax1.set_ylim(0, max(drops_pp) + 10)
    ax1.legend(loc="lower right", framealpha=0.9)

    # Right: per-class recall sweep
    classes = ["Enemy Drone", "Bird", "Friendly UAV", "Manned Aircraft"]
    width = 0.18
    x = np.arange(len(half_widths))
    for i, cls in enumerate(classes):
        recalls = []
        for r in rows:
            info = r["per_class"].get(cls)
            recalls.append(info["recall"] * 100 if info else 0)
        ax2.bar(x + (i - 1.5) * width, recalls, width,
                 color=CLASS_COLOURS[cls], label=cls,
                 edgecolor="white", linewidth=0.6)

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"±{h}" for h in half_widths])
    ax2.set_xlabel("Mask half-width")
    ax2.set_ylabel("Per-class recall (%)")
    ax2.set_title("Bird recall collapses at +-1 bin; other classes degrade gradually",
                   pad=15)
    ax2.set_ylim(0, 110)
    ax2.legend(loc="lower left", framealpha=0.9, fontsize=11)

    plt.tight_layout()
    out = OUT_DIR / "fig7_classcond.png"
    plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="white")
    plt.close()
    print(f"Wrote {out}")


def main():
    fig_d1()
    fig_b1()
    fig_classcond()


if __name__ == "__main__":
    main()
