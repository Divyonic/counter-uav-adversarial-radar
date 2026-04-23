"""
Generate improved Figures 1 (system architecture), 4 (CFAR), and 5 (model architecture).
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import os, sys

sys.path.insert(0, os.path.dirname(__file__))
from fmcw_simulation import (generate_drone_signal, compute_range_doppler_map,
                               apply_cfar, RadarParams)

FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'font.family': 'sans-serif',
    'figure.facecolor': 'white',
})


def fig1_system_architecture():
    """Rendered system architecture block diagram."""
    fig, ax = plt.subplots(figsize=(16, 7))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis('off')

    # Color scheme
    c_radar = '#2980b9'
    c_sigproc = '#27ae60'
    c_ml = '#8e44ad'
    c_decision = '#c0392b'
    c_output = '#f39c12'

    box_h = 2.2
    box_y = 3.0
    text_kw = dict(ha='center', va='center', fontweight='bold', color='white')

    # Main processing blocks
    blocks = [
        (0.5,  box_y, 2.8, box_h, c_radar,   'FMCW Radar\nFront End',
         'X-band (9.5 GHz)\n400 MHz BW\n256 chirps/frame', 10),
        (4.3,  box_y, 3.0, box_h, c_sigproc,  'Signal\nProcessing',
         '2D-FFT → RD Map\nSTFT → Spectrogram\nCA-CFAR Detection\nBFP Extraction', 9),
        (8.3,  box_y, 3.0, box_h, c_ml,       'CNN + LSTM\nClassifier',
         'CNN: 420K params\nLSTM: 63K params\nT=10 frame sequence\nBFP concatenation', 9),
        (12.3, box_y, 3.0, box_h, c_decision,  'Decision &\nAlert Module',
         'Target classification\nThreat assessment\nTrajectory estimate\nC2 output', 9),
    ]

    for x, y, w, h, color, title, detail, dsize in blocks:
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                              facecolor=color, edgecolor='white', linewidth=2, alpha=0.9)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.68, title, **text_kw, fontsize=13)
        ax.text(x + w/2, y + h*0.25, detail,
                ha='center', va='center', fontsize=dsize, color='#ecf0f1', linespacing=1.3)

    # Arrows between blocks
    arrow_kw = dict(arrowstyle='->', color='#2c3e50', lw=2.5, mutation_scale=20)
    for x1, x2 in [(3.3, 4.3), (7.3, 8.3), (11.3, 12.3)]:
        ax.annotate('', xy=(x2, box_y + box_h/2), xytext=(x1, box_y + box_h/2),
                    arrowprops=arrow_kw)

    # Output labels below
    outputs = [
        (1.9,  'Raw IF\nSignal'),
        (5.8,  'RD Map +\nSpectrogram +\nCFAR + BFP'),
        (9.8,  'Classification\n+ Trajectory'),
        (13.8, 'Threat Level\n+ Response'),
    ]
    for x, label in outputs:
        ax.annotate('', xy=(x, box_y - 0.6), xytext=(x, box_y),
                    arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.5))
        ax.text(x, box_y - 1.2, label, ha='center', va='center', fontsize=9,
                color='#2c3e50', fontstyle='italic')

    # Title
    ax.text(8, 6.5, 'Counter-UAV System Architecture', ha='center', va='center',
            fontsize=16, fontweight='bold', color='#2c3e50')

    # Stage labels
    for i, (x, label) in enumerate([(1.9, 'Stage 1'), (5.8, 'Stage 2'), (9.8, 'Stage 3'), (13.8, 'Stage 4')]):
        ax.text(x, box_y + box_h + 0.3, label, ha='center', va='center',
                fontsize=10, color='#7f8c8d', fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'system_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 1: System architecture")


def fig4_cfar_improved():
    """Improved CFAR detection figure with visible target and proper axes."""
    np.random.seed(123)

    # Generate drone at a range that maps to a visible bin
    # Use a closer range and higher SNR for clear visibility
    beat = generate_drone_signal(R0=300, v_bulk=10, snr_db=25)
    rd_map = compute_range_doppler_map(beat)

    # Also generate noise-only for false alarm comparison
    Nc, Ns = RadarParams.Nc, RadarParams.Ns
    noise_beat = np.sqrt(0.5) * (np.random.randn(Nc, Ns) + 1j * np.random.randn(Nc, Ns))
    noise_rd = compute_range_doppler_map(noise_beat)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    rd_db = 10 * np.log10(rd_map + 1e-10)
    # Find peak to center the view
    peak_idx = np.unravel_index(np.argmax(rd_map), rd_map.shape)

    # Show a region around the target
    r_center = peak_idx[1]
    d_center = peak_idx[0]
    r_win = 60
    d_win = 40

    r_lo = max(0, r_center - r_win)
    r_hi = min(Ns, r_center + r_win)
    d_lo = max(0, d_center - d_win)
    d_hi = min(Nc, d_center + d_win)

    range_axis = np.arange(r_lo, r_hi) * RadarParams.delta_R
    vel_axis = (np.arange(d_lo, d_hi) - Nc//2) * RadarParams.delta_v

    rd_region = rd_db[d_lo:d_hi, r_lo:r_hi]
    rd_power_region = rd_map[d_lo:d_hi, r_lo:r_hi]

    # Panel 1: Range-Doppler map with target
    im = axes[0].pcolormesh(range_axis, vel_axis, rd_region, shading='auto', cmap='jet')
    axes[0].set_title('Range-Doppler Map\n(Drone at R=300m, v=10 m/s)', fontweight='bold')
    axes[0].set_xlabel('Range (m)')
    axes[0].set_ylabel('Velocity (m/s)')
    plt.colorbar(im, ax=axes[0], label='Power (dB)', shrink=0.8)
    # Mark target
    axes[0].plot(peak_idx[1] * RadarParams.delta_R, (peak_idx[0] - Nc//2) * RadarParams.delta_v,
                 'wo', markersize=12, markerfacecolor='none', markeredgewidth=2)

    # Panel 2: Fixed threshold detection
    threshold = np.mean(rd_power_region) + 3 * np.std(rd_power_region)
    fixed_det = rd_power_region > threshold
    n_fixed = np.sum(fixed_det)

    det_display = np.zeros_like(rd_region)
    det_display[fixed_det] = 1.0
    # Also show the RD map faintly
    bg = (rd_region - rd_region.min()) / (rd_region.max() - rd_region.min())

    axes[1].pcolormesh(range_axis, vel_axis, bg, shading='auto', cmap='Greys', alpha=0.3)
    # Overlay detections as red dots
    det_r, det_d = np.where(fixed_det)
    if len(det_r) > 0:
        axes[1].scatter(range_axis[0] + det_d * RadarParams.delta_R,
                       vel_axis[0] + det_r * RadarParams.delta_v,
                       c='red', s=8, alpha=0.8, label='False alarm')
    # True target
    if d_lo <= peak_idx[0] < d_hi and r_lo <= peak_idx[1] < r_hi:
        axes[1].plot(peak_idx[1] * RadarParams.delta_R, (peak_idx[0] - Nc//2) * RadarParams.delta_v,
                     'g*', markersize=15, label='True target')
    axes[1].set_title(f'Fixed Threshold (μ+3σ)\n{n_fixed} detections', fontweight='bold')
    axes[1].set_xlabel('Range (m)')
    axes[1].set_ylabel('Velocity (m/s)')
    axes[1].legend(loc='upper right', fontsize=9)

    # Panel 3: CFAR detection
    detections, alpha = apply_cfar(rd_map, n_train=16, n_guard=4, pfa=1e-4)
    cfar_region = detections[d_lo:d_hi, r_lo:r_hi]
    n_cfar = np.sum(cfar_region)

    axes[2].pcolormesh(range_axis, vel_axis, bg, shading='auto', cmap='Greys', alpha=0.3)
    det_r, det_d = np.where(cfar_region)
    if len(det_r) > 0:
        axes[2].scatter(range_axis[0] + det_d * RadarParams.delta_R,
                       vel_axis[0] + det_r * RadarParams.delta_v,
                       c='lime', s=30, marker='D', edgecolors='darkgreen',
                       linewidth=1, label='CFAR detection')
    axes[2].set_title(f'CA-CFAR (Pfa=10⁻⁴)\n{n_cfar} detections', fontweight='bold')
    axes[2].set_xlabel('Range (m)')
    axes[2].set_ylabel('Velocity (m/s)')
    axes[2].legend(loc='upper right', fontsize=9)

    plt.suptitle('CFAR vs Fixed Threshold Detection — Drone Target at SNR = 25 dB',
                 fontweight='bold', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cfar_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 4: CFAR detection (improved)")


def fig5_model_architecture():
    """Rendered CNN+LSTM+BFP model architecture diagram."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')

    def draw_box(x, y, w, h, color, label, detail='', fontsize=10):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                              facecolor=color, edgecolor='#2c3e50', linewidth=1.5, alpha=0.85)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, label, ha='center', va='center',
                fontsize=fontsize, fontweight='bold', color='white')
        if detail:
            ax.text(x + w/2, y + h*0.3, detail, ha='center', va='center',
                    fontsize=8, color='#ecf0f1')

    def arrow_down(x, y1, y2, label=''):
        ax.annotate('', xy=(x, y2), xytext=(x, y1),
                    arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=2))
        if label:
            ax.text(x + 0.15, (y1+y2)/2, label, fontsize=8, color='#7f8c8d',
                    va='center', ha='left', fontstyle='italic')

    # Title
    ax.text(7, 9.7, 'CNN + LSTM + BFP Model Architecture', ha='center',
            fontsize=16, fontweight='bold', color='#2c3e50')

    # Input
    draw_box(2, 8.7, 5, 0.7, '#95a5a6', 'Input: Spectrogram Sequence',
             '{S₁, S₂, ..., S₁₀}  —  T=10 frames × 128×128', fontsize=11)

    # Dashed box around CNN (shared weights)
    shared_box = FancyBboxPatch((1.5, 5.0), 6, 3.3, boxstyle="round,pad=0.15",
                                 facecolor='none', edgecolor='#3498db',
                                 linewidth=2, linestyle='--', alpha=0.7)
    ax.add_patch(shared_box)
    ax.text(4.5, 8.1, 'CNN Feature Extractor (shared weights across T frames)',
            ha='center', fontsize=10, color='#3498db', fontweight='bold')

    arrow_down(4.5, 8.7, 7.8)

    # CNN layers
    cnn_layers = [
        (1.8, 7.1, 2.5, 0.55, '#3498db', 'Conv1+BN+Pool', '3×3, 32 → 64×64'),
        (4.7, 7.1, 2.5, 0.55, '#2980b9', 'Conv2+BN+Pool', '3×3, 64 → 32×32'),
        (1.8, 6.2, 2.5, 0.55, '#2471a3', 'Conv3+BN+Pool', '3×3, 128 → 16×16'),
        (4.7, 6.2, 2.5, 0.55, '#1a5276', 'Conv4+BN+Pool', '3×3, 256 → 8×8'),
    ]
    for x, y, w, h, c, l, d in cnn_layers:
        draw_box(x, y, w, h, c, l, d, fontsize=9)

    # Arrows between CNN layers
    ax.annotate('', xy=(4.7, 7.35), xytext=(4.3, 7.35),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
    ax.annotate('', xy=(1.8, 6.75), xytext=(3.05, 7.1),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))
    ax.annotate('', xy=(4.7, 6.45), xytext=(4.3, 6.45),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    # GAP + FC
    draw_box(2.5, 5.2, 4, 0.6, '#1b4f72', 'GAP → FC(128) + Dropout(0.5)',
             '~420K params total', fontsize=10)
    ax.annotate('', xy=(4.5, 5.8), xytext=(5.95, 6.2),
                arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=1.5))

    arrow_down(4.5, 5.2, 4.6, '128-dim')

    # BFP feature box (right side)
    draw_box(8.5, 5.8, 3.5, 1.5, '#e67e22', 'BFP Feature\nExtraction',
             'Autocorrelation →\nf_BFP, C_BFP, Δf_BFP\n(3-dim vector)', fontsize=10)

    # BFP arrow to concatenation
    ax.annotate('', xy=(8.0, 4.35), xytext=(8.5, 5.8),
                arrowprops=dict(arrowstyle='->', color='#e67e22', lw=2))

    # Concatenation
    draw_box(3.0, 3.9, 5.5, 0.6, '#16a085', 'Feature Concatenation',
             'CNN(128) ⊕ BFP(3) = 131-dim per frame', fontsize=10)
    arrow_down(4.5, 4.6, 4.5)

    arrow_down(5.75, 3.9, 3.3, '131 × T')

    # LSTM
    draw_box(2.5, 2.2, 5.5, 1.0, '#8e44ad', 'LSTM Temporal Tracker',
             'LSTM₁(64) → LSTM₂(32) — ~63K params', fontsize=11)

    arrow_down(5.25, 2.2, 1.5, '32-dim')

    # Softmax output
    draw_box(3.0, 0.6, 4.5, 0.8, '#c0392b', 'Softmax Classification',
             'P(drone) | P(bird) | P(FW-UAV) | P(aircraft)', fontsize=10)

    # Total params annotation
    ax.text(12.5, 2.5, 'Total:\n~484K\nparams', ha='center', va='center',
            fontsize=12, fontweight='bold', color='#2c3e50',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#f0f0f0', edgecolor='#bbb'))

    # Latency annotation
    ax.text(12.5, 1.2, 'Inference:\n14.7 ms\n(CPU)', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#27ae60',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#eafaf1', edgecolor='#27ae60'))

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'model_architecture.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Figure 5: Model architecture")


if __name__ == '__main__':
    print("Generating improved figures...")
    fig1_system_architecture()
    fig4_cfar_improved()
    fig5_model_architecture()
    print("Done!")
