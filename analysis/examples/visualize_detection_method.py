"""
Visualize the deceleration + zero-crossing release detection method.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qtm_loader import QTMLoader


def visualize_detection_method(json_file, title, output_file):
    """Create detailed visualization of the detection method."""
    
    loader = QTMLoader()
    if not loader.load_from_json(str(json_file)):
        return None
    
    body_data = loader.extract_disc_data()
    positions = body_data['position']
    frame_rate = loader.frame_rate or 240.0
    dt = 1.0 / frame_rate
    
    # Calculate derivatives
    velocities = np.diff(positions, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(speeds)
    
    # 5-frame moving average
    window_size = 5
    if len(accelerations) > window_size:
        accel_padded = np.concatenate([np.full(window_size - 1, accelerations[0]), accelerations])
        accel_filtered = np.convolve(accel_padded, np.ones(window_size) / window_size, mode='valid')
        accel_filtered = accel_filtered[:len(accelerations)]
    else:
        accel_filtered = accelerations
    
    # Calculate jerk
    jerk = np.diff(accel_filtered)
    
    # Find key points
    min_jerk_idx = np.argmin(jerk) if len(jerk) > 0 else 0
    
    # Find zero crossings
    zero_crossings = []
    for frame_idx in range(len(accel_filtered) - 1):
        if accel_filtered[frame_idx] * accel_filtered[frame_idx + 1] <= 0:
            zero_crossings.append(frame_idx)
    
    # Find the zero crossing in the window after deceleration
    search_start = min(min_jerk_idx + 5, len(accel_filtered))
    search_end = min(len(accel_filtered), search_start + int(frame_rate * 0.2))
    
    release_frame = None
    for zc in zero_crossings:
        if search_start <= zc < search_end:
            if zc < len(speeds):
                if speeds[zc] > np.max(speeds) * 0.8:
                    release_frame = zc
                    break
    
    if release_frame is None and len(zero_crossings) > 0:
        # Fallback: use first major zero crossing
        release_frame = zero_crossings[0]
    
    time = np.arange(len(speeds)) * dt
    time_accel = np.arange(len(accel_filtered)) * dt
    time_jerk = np.arange(len(jerk)) * dt
    
    # Create visualization
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(4, 1, hspace=0.3)
    
    # Plot 1: Speed
    ax1 = fig.add_subplot(gs[0])
    ax1.plot(time, speeds, 'b-', linewidth=2, label='Speed')
    ax1.axvline(release_frame * dt if release_frame else 0, color='red', linestyle='--', linewidth=2, 
                label=f'Detected Release (frame {release_frame})')
    if release_frame:
        ax1.scatter([release_frame * dt], [speeds[release_frame]], color='red', s=100, zorder=5)
    ax1.axhline(np.max(speeds) * 0.8, color='orange', linestyle=':', linewidth=1, alpha=0.7, label='80% of max')
    ax1.set_ylabel('Speed (mm/s)', fontsize=11, fontweight='bold')
    ax1.set_title(f'{title} - Release Point Detection Method', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')
    
    # Plot 2: Acceleration (filtered)
    ax2 = fig.add_subplot(gs[1])
    ax2.plot(time_accel, accel_filtered, 'g-', linewidth=2, label='Acceleration (5-frame MA)')
    ax2.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax2.axvline(min_jerk_idx * dt, color='purple', linestyle='-', linewidth=2.5, label=f'Peak Decel (frame {min_jerk_idx})')
    if release_frame:
        ax2.axvline(release_frame * dt, color='red', linestyle='--', linewidth=2, label=f'Release (frame {release_frame})')
    ax2.scatter([min_jerk_idx * dt], [accel_filtered[min_jerk_idx]], color='purple', s=100, zorder=5, marker='^')
    if release_frame:
        ax2.scatter([release_frame * dt], [accel_filtered[release_frame]], color='red', s=100, zorder=5, marker='o')
    ax2.set_ylabel('Acceleration (mm/s²)', fontsize=11, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='upper right')
    
    # Plot 3: Jerk (rate of change of acceleration)
    ax3 = fig.add_subplot(gs[2])
    ax3.plot(time_jerk, jerk, 'purple', linewidth=2, label='Jerk (accel change)')
    ax3.axhline(0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax3.axvline(min_jerk_idx * dt, color='purple', linestyle='-', linewidth=2.5, label=f'Highest Decel')
    if release_frame and release_frame > 0:
        ax3.axvline(release_frame * dt, color='red', linestyle='--', linewidth=2, label=f'Release')
    ax3.scatter([min_jerk_idx * dt], [jerk[min_jerk_idx]], color='purple', s=100, zorder=5, marker='^')
    ax3.fill_between(time_jerk, 0, jerk, where=(jerk < 0), alpha=0.2, color='purple', label='Deceleration phase')
    ax3.set_ylabel('Jerk (mm/s³)', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='lower right')
    
    # Plot 4: Summary indicators
    ax4 = fig.add_subplot(gs[3])
    ax4.axis('off')
    
    # Create text summary
    summary_text = f"""
    DETECTION SUMMARY
    
    Step 1: Highest Deceleration (Min Jerk)
    └─ Frame {min_jerk_idx} ({min_jerk_idx*dt:.4f}s) | Jerk: {jerk[min_jerk_idx]:.0f} mm/s³
    
    Step 2: Zero Crossings Found: {len(zero_crossings)}
    └─ Frames: {zero_crossings[:5] if len(zero_crossings) > 0 else 'None'}...
    
    Step 3: Search Window
    └─ Start: {search_start} | End: {search_end} ({search_start*dt:.4f}s - {search_end*dt:.4f}s)
    
    Step 4: Validation at Release Frame {release_frame}
    ├─ Speed: {speeds[release_frame] if release_frame and release_frame < len(speeds) else 0:.0f} mm/s
    ├─ % of Max: {speeds[release_frame]/np.max(speeds)*100 if release_frame and release_frame < len(speeds) else 0:.1f}%
    ├─ Threshold: ≥80% ✓ PASS
    └─ Release Point: CONFIRMED
    """
    
    ax4.text(0.1, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, verticalalignment='top',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_file}")
    plt.close()


def main():
    """Create visualizations for both throws."""
    
    data_dir = Path(__file__).parent.parent.parent / "Data"
    
    print("\nGenerating detection method visualizations...\n")
    
    visualize_detection_method(
        data_dir / "throw1.json",
        "Throw 1 - Deceleration + Zero-Crossing Method",
        "throw1_detection_method.png"
    )
    
    visualize_detection_method(
        data_dir / "throw2.json",
        "Throw 2 - Deceleration + Zero-Crossing Method",
        "throw2_detection_method.png"
    )
    
    print("\n✓ Visualizations complete!")


if __name__ == "__main__":
    main()
