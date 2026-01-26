"""
Test the new deceleration + zero-crossing release detection method.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer


def test_detection(json_file, title):
    """Test the new detection method."""
    
    loader = QTMLoader()
    if not loader.load_from_json(str(json_file)):
        print(f"Failed to load {json_file}")
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
    
    # Use the analyzer's detection method
    analyzer = DiscAnalyzer(frame_rate=frame_rate)
    release_frame = analyzer._detect_release_frame(speeds)
    
    # Calculate jerk for analysis
    jerk = np.diff(accel_filtered)
    min_jerk_idx = np.argmin(jerk) if len(jerk) > 0 else 0
    
    # Find zero crossing
    zero_crossing_frames = []
    for frame_idx in range(len(accel_filtered) - 1):
        if accel_filtered[frame_idx] * accel_filtered[frame_idx + 1] <= 0:
            zero_crossing_frames.append(frame_idx)
    
    time = np.arange(len(speeds)) * dt
    
    print(f"\n{title}")
    print("=" * 80)
    print(f"Total frames: {len(speeds)} ({time[-1]:.4f}s)")
    print(f"Frame rate: {frame_rate:.1f} Hz")
    print()
    print("Detection Analysis:")
    print(f"  Highest deceleration (min jerk): Frame {min_jerk_idx} ({min_jerk_idx*dt:.4f}s)")
    print(f"  Jerk value: {jerk[min_jerk_idx] if min_jerk_idx < len(jerk) else 0:.0f} mm/s³")
    print()
    print(f"  Zero crossings found: {len(zero_crossing_frames)}")
    for i, zc in enumerate(zero_crossing_frames[:3]):
        accel_at_zc = accel_filtered[zc] if zc < len(accel_filtered) else 0
        print(f"    {i+1}. Frame {zc} ({zc*dt:.4f}s) - Accel: {accel_at_zc:.0f} mm/s²")
    print()
    print(f"  DETECTED RELEASE FRAME: {release_frame} ({release_frame*dt:.4f}s)")
    print(f"  Speed at release: {speeds[release_frame]:.0f} mm/s")
    print(f"  Max speed: {np.max(speeds):.0f} mm/s")
    print(f"  Ratio: {speeds[release_frame] / np.max(speeds) * 100:.1f}% of max")
    print()
    
    # Check against your manual review for throw2
    if "throw2" in str(json_file).lower():
        print(f"  Manual review frames: 156-157")
        print(f"  Algorithm vs manual: {abs(release_frame - 156.5):.1f} frames difference")
        if abs(release_frame - 156.5) < 5:
            print(f"  ✓ MATCH!")
        else:
            print(f"  ⚠ Difference > 5 frames")
    
    return {
        'release_frame': release_frame,
        'release_time': release_frame * dt,
        'release_speed': speeds[release_frame],
        'peak_speed': np.max(speeds),
        'min_jerk_frame': min_jerk_idx,
        'zero_crossings': zero_crossing_frames,
        'speeds': speeds,
        'accelerations': accel_filtered,
        'jerk': jerk,
        'time': time,
        'dt': dt,
    }


def main():
    """Test both throws."""
    
    data_dir = Path(__file__).parent.parent.parent / "Data"
    
    print("\n" + "=" * 80)
    print("DECELERATION + ZERO-CROSSING RELEASE DETECTION")
    print("=" * 80)
    
    results1 = test_detection(data_dir / "throw1.json", "THROW 1")
    results2 = test_detection(data_dir / "throw2.json", "THROW 2")
    
    if results1 and results2:
        print("\n" + "=" * 80)
        print("COMPARISON")
        print("=" * 80)
        print(f"\nThrow 1: Frame {results1['release_frame']} - Speed {results1['release_speed']:.0f} mm/s ({results1['release_speed']/results1['peak_speed']*100:.1f}% of max)")
        print(f"Throw 2: Frame {results2['release_frame']} - Speed {results2['release_speed']:.0f} mm/s ({results2['release_speed']/results2['peak_speed']*100:.1f}% of max)")


if __name__ == "__main__":
    main()
