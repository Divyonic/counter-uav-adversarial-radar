"""
Generate all figures for the Counter-UAV paper.
Uses matplotlib to produce publication-quality charts.
Run AFTER train_and_evaluate.py has completed.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from fmcw_simulation import (generate_drone_signal, generate_bird_signal,
                               generate_aircraft_signal, generate_friendly_uav_signal,
                               compute_range_doppler_map, compute_spectrogram,
                               apply_cfar, extract_bfp_features, RadarParams)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results')
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 150,
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
})


def fig1_range_doppler_maps():
    """Generate example Range-Doppler maps for each target class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    configs = [
        ('Enemy Drone', generate_drone_signal, {'R0': 800, 'v_bulk': 12, 'snr_db': 20}),
        ('Bird', generate_bird_signal, {'R0': 600, 'v_bulk': 8, 'snr_db': 20}),
        ('Friendly UAV', generate_friendly_uav_signal, {'R0': 500, 'v_bulk': 6, 'snr_db': 20}),
        ('Manned Aircraft', generate_aircraft_signal, {'R0': 2000, 'v_bulk': 80, 'snr_db': 20}),
    ]
    
    for ax, (name, gen_func, params) in zip(axes.flat, configs):
        beat = gen_func(**params)
        rd_map = compute_range_doppler_map(beat)
        rd_db = 10 * np.log10(rd_map + 1e-10)
        
        # Axes
        range_axis = np.arange(RadarParams.Ns) * RadarParams.delta_R
        vel_axis = np.linspace(-RadarParams.v_max, RadarParams.v_max, RadarParams.Nc)
        
        im = ax.pcolormesh(range_axis[:RadarParams.Ns//4], vel_axis, 
                           rd_db[:, :RadarParams.Ns//4],
                           shading='auto', cmap='jet')
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Range (m)')
        ax.set_ylabel('Velocity (m/s)')
        plt.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.suptitle('Range-Doppler Maps for Different Target Classes', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'range_doppler_maps.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Range-Doppler maps")


def fig2_spectrograms():
    """Generate example micro-Doppler spectrograms for each class."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    configs = [
        ('Enemy Drone', generate_drone_signal, {'R0': 800, 'v_bulk': 12, 'snr_db': 20}),
        ('Bird', generate_bird_signal, {'R0': 600, 'v_bulk': 8, 'snr_db': 20}),
        ('Friendly UAV', generate_friendly_uav_signal, {'R0': 500, 'v_bulk': 6, 'snr_db': 20}),
        ('Manned Aircraft', generate_aircraft_signal, {'R0': 2000, 'v_bulk': 80, 'snr_db': 20}),
    ]
    
    for ax, (name, gen_func, params) in zip(axes.flat, configs):
        beat = gen_func(**params)
        spec, f, t = compute_spectrogram(beat)
        spec_db = 10 * np.log10(spec + 1e-10)
        
        im = ax.pcolormesh(t * 1000, f, spec_db, shading='auto', cmap='viridis')
        ax.set_title(f'{name} - Micro-Doppler Spectrogram', fontweight='bold')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Doppler Frequency (Hz)')
        plt.colorbar(im, ax=ax, label='Power (dB)')
    
    plt.suptitle('Micro-Doppler Spectrograms', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'spectrograms.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Micro-Doppler spectrograms")


def fig3_cfar_detection():
    """Show CFAR detection on a Range-Doppler map."""
    beat = generate_drone_signal(R0=800, v_bulk=10, snr_db=15)
    rd_map = compute_range_doppler_map(beat)
    detections, alpha = apply_cfar(rd_map)
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    rd_db = 10 * np.log10(rd_map + 1e-10)
    range_axis = np.arange(RadarParams.Ns) * RadarParams.delta_R
    vel_axis = np.linspace(-RadarParams.v_max, RadarParams.v_max, RadarParams.Nc)
    
    # Raw RD map
    im = axes[0].pcolormesh(range_axis[:RadarParams.Ns//4], vel_axis, 
                            rd_db[:, :RadarParams.Ns//4], shading='auto', cmap='jet')
    axes[0].set_title('Range-Doppler Map', fontweight='bold')
    axes[0].set_xlabel('Range (m)')
    axes[0].set_ylabel('Velocity (m/s)')
    plt.colorbar(im, ax=axes[0])
    
    # Fixed threshold
    threshold = np.mean(rd_map) + 3 * np.std(rd_map)
    fixed_det = rd_map > threshold
    axes[1].pcolormesh(range_axis[:RadarParams.Ns//4], vel_axis,
                       fixed_det[:, :RadarParams.Ns//4].astype(float),
                       shading='auto', cmap='RdYlGn_r')
    axes[1].set_title(f'Fixed Threshold\n({np.sum(fixed_det)} detections)', fontweight='bold')
    axes[1].set_xlabel('Range (m)')
    axes[1].set_ylabel('Velocity (m/s)')
    
    # CFAR
    axes[2].pcolormesh(range_axis[:RadarParams.Ns//4], vel_axis,
                       detections[:, :RadarParams.Ns//4].astype(float),
                       shading='auto', cmap='RdYlGn_r')
    axes[2].set_title(f'CA-CFAR Detection\n({np.sum(detections)} detections)', fontweight='bold')
    axes[2].set_xlabel('Range (m)')
    axes[2].set_ylabel('Velocity (m/s)')
    
    plt.suptitle('CFAR vs Fixed Threshold Detection', fontweight='bold', fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'cfar_detection.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ CFAR detection comparison")


def fig4_confusion_matrix(results):
    """Plot confusion matrix from experiment results."""
    cm = np.array(results['cnn_lstm_bfp']['confusion_matrix'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Normalize
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
    
    im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=100)
    
    class_names = ['Enemy\nDrone', 'Bird', 'Friendly\nUAV', 'Manned\nAircraft']
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    
    # Add text annotations
    for i in range(4):
        for j in range(4):
            color = 'white' if cm_norm[i, j] > 50 else 'black'
            ax.text(j, i, f'{cm_norm[i,j]:.1f}%\n({cm[i,j]})', 
                    ha='center', va='center', color=color, fontsize=10)
    
    ax.set_xlabel('Predicted Class', fontweight='bold')
    ax.set_ylabel('True Class', fontweight='bold')
    ax.set_title('Confusion Matrix, CNN+LSTM+BFP Model', fontweight='bold', fontsize=12)
    plt.colorbar(im, label='Classification Rate (%)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Confusion matrix")


def fig5_snr_accuracy(results):
    """Plot accuracy vs SNR for different model configurations."""
    snr_data = results['snr_sweep']
    snr_levels = sorted([int(k) for k in snr_data.keys()])
    
    cnn_acc = [snr_data[str(s)]['cnn_only'] * 100 for s in snr_levels]
    bfp_acc = [snr_data[str(s)]['cnn_bfp'] * 100 for s in snr_levels]
    lstm_acc = [snr_data[str(s)]['cnn_lstm_bfp'] * 100 for s in snr_levels]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(snr_levels, cnn_acc, 'o-', label='CNN Only', linewidth=2, markersize=8, color='#e74c3c')
    ax.plot(snr_levels, bfp_acc, 's-', label='CNN + BFP', linewidth=2, markersize=8, color='#f39c12')
    ax.plot(snr_levels, lstm_acc, 'D-', label='CNN + LSTM + BFP (Full)', linewidth=2, markersize=8, color='#2ecc71')
    
    ax.set_xlabel('Signal-to-Noise Ratio (dB)', fontweight='bold')
    ax.set_ylabel('Classification Accuracy (%)', fontweight='bold')
    ax.set_title('Classification Accuracy vs SNR', fontweight='bold', fontsize=13)
    ax.legend(loc='lower right', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-1, 26)
    ax.set_ylim(0, 105)
    ax.axhline(y=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'snr_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ SNR vs accuracy")


def fig6_ablation_bar(results):
    """Ablation study bar chart."""
    configs = {
        'CNN Only': results['cnn_only'],
        'CNN + BFP': results['cnn_bfp'],
        'CNN + LSTM + BFP': results['cnn_lstm_bfp'],
    }
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    names = list(configs.keys())
    accs = [v['accuracy'] * 100 for v in configs.values()]
    fars = [v['far'] * 100 for v in configs.values()]
    lats = [v['latency']['mean_ms'] for v in configs.values()]
    
    colors = ['#e74c3c', '#f39c12', '#2ecc71']
    
    # Accuracy
    bars = axes[0].bar(names, accs, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Accuracy (%)')
    axes[0].set_title('Classification Accuracy', fontweight='bold')
    for bar, val in zip(bars, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    axes[0].set_ylim(0, 110)
    
    # FAR
    bars = axes[1].bar(names, fars, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('False Alarm Rate (%)')
    axes[1].set_title('False Alarm Rate', fontweight='bold')
    for bar, val in zip(bars, fars):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                     f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Latency
    bars = axes[2].bar(names, lats, color=colors, edgecolor='black', linewidth=0.5)
    axes[2].set_ylabel('Inference Latency (ms)')
    axes[2].set_title('Inference Latency (CPU)', fontweight='bold')
    for bar, val in zip(bars, lats):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{val:.1f}ms', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Ablation Study Results', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'ablation_study.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Ablation study")


def fig7_per_class_metrics(results):
    """Per-class precision, recall, F1 bar chart."""
    full_results = results['cnn_lstm_bfp']
    
    class_names = ['Enemy Drone', 'Bird', 'Friendly UAV', 'Manned Aircraft']
    
    x = np.arange(len(class_names))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.bar(x - width, full_results['precision'], width, label='Precision', color='#3498db', edgecolor='black', linewidth=0.5)
    ax.bar(x, full_results['recall'], width, label='Recall', color='#e74c3c', edgecolor='black', linewidth=0.5)
    ax.bar(x + width, full_results['f1'], width, label='F1-Score', color='#2ecc71', edgecolor='black', linewidth=0.5)
    
    ax.set_xlabel('Target Class', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title('Per-Class Performance, CNN+LSTM+BFP Model', fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(class_names)
    ax.legend()
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.2, axis='y')
    
    # Add value labels
    for i, (p, r, f) in enumerate(zip(full_results['precision'], full_results['recall'], full_results['f1'])):
        ax.text(i - width, p + 0.02, f'{p:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i, r + 0.02, f'{r:.3f}', ha='center', va='bottom', fontsize=8)
        ax.text(i + width, f + 0.02, f'{f:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'per_class_metrics.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Per-class metrics")


def generate_all_figures():
    """Generate all figures."""
    print("Generating figures...")
    
    # Signal processing figures (no experiment data needed)
    fig1_range_doppler_maps()
    fig2_spectrograms()
    fig3_cfar_detection()
    
    # Load experiment results
    results_path = os.path.join(RESULTS_DIR, 'experiment_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)
        
        fig4_confusion_matrix(results)
        fig5_snr_accuracy(results)
        fig6_ablation_bar(results)
        fig7_per_class_metrics(results)
    else:
        print("  ⚠ No experiment results found, skipping result-dependent figures")
        print(f"    Run train_and_evaluate.py first, then re-run this script")
    
    print(f"\nAll figures saved to {FIGURES_DIR}/")


if __name__ == '__main__':
    generate_all_figures()
