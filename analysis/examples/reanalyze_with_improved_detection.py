"""
Re-analyze both throws with improved release detection.
Compare metrics before and after the fix.
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer
from src.visualization.visualizer_6dof import DiscVisualizer6DOF


def analyze_throw(json_file, title):
    """Analyze a single throw with improved detection."""
    
    loader = QTMLoader()
    if not loader.load_from_json(str(json_file)):
        print(f"Failed to load {json_file}")
        return None
    
    body_data = loader.extract_disc_data()
    
    analyzer = DiscAnalyzer(frame_rate=loader.frame_rate)
    results = analyzer.analyze_disc_trajectory(body_data)
    
    return results, body_data, analyzer.frame_rate


def print_results(title, results):
    """Print detailed results."""
    print(f"\n{title}")
    print("=" * 70)
    
    if results is None:
        print("Failed to analyze")
        return
    
    print(f"Duration: {results['early_phase_duration']:.4f}s ({results['total_frames_analyzed']} frames)")
    print()
    print("6 KEY DISC PARAMETERS AT RELEASE:")
    print(f"  1. Disc Speed:        {results['disc_speed']:>10.0f} mm/s")
    print(f"  2. Spin Rate (RPM):   {results['spin']:>10.1f} RPM" if results['spin'] else f"  2. Spin Rate (RPM):   N/A")
    print(f"  3. Hyzer Angle:       {results['hyzer_angle']:>10.1f}°")
    print(f"  4. Launch Angle:      {results['launch_angle']:>10.1f}°")
    print(f"  5. Nose Angle:        {results['nose_angle']:>10.1f}°")
    print(f"  6. Wobble Amplitude:  {results['wobble_amplitude']:>10.1f}°")
    print()
    print("EARLY PHASE DYNAMICS:")
    print(f"  Acceleration Duration: {results['acceleration_phase_duration']:.4f}s")
    print(f"  Acceleration Distance: {results['acceleration_distance']:.0f} mm")
    print(f"  Max Acceleration:      {results['max_acceleration']:.0f} mm/s²")
    print(f"  Average Acceleration:  {results['average_acceleration']:.0f} mm/s²")
    print()
    print("ANGULAR VELOCITIES AT RELEASE:")
    print(f"  Yaw (Z-axis):   {results['angular_velocity_yaw']:>8.2f} rad/s")
    print(f"  Pitch (Y-axis): {results['angular_velocity_pitch']:>8.2f} rad/s")
    print(f"  Roll (X-axis):  {results['angular_velocity_roll']:>8.2f} rad/s")


def main():
    """Analyze both throws with improved detection."""
    
    data_dir = Path(__file__).parent.parent.parent / "Data"
    
    print("\n" + "=" * 70)
    print("RE-ANALYSIS WITH IMPROVED RELEASE POINT DETECTION")
    print("=" * 70)
    
    # Analyze Throw 1
    results1, body1, frame_rate1 = analyze_throw(data_dir / "throw1.json", "Throw 1")
    print_results("THROW 1 - WITH IMPROVED DETECTION", results1)
    
    # Analyze Throw 2
    results2, body2, frame_rate2 = analyze_throw(data_dir / "throw2.json", "Throw 2")
    print_results("THROW 2 - WITH IMPROVED DETECTION", results2)
    
    # Comparison
    print("\n" + "=" * 70)
    print("THROW COMPARISON")
    print("=" * 70)
    
    if results1 and results2:
        print(f"\n{'Metric':<30} {'Throw 1':>15} {'Throw 2':>15} {'Difference':<15}")
        print("-" * 75)
        
        metrics = [
            ('Duration', 'early_phase_duration', 's'),
            ('Disc Speed', 'disc_speed', 'mm/s'),
            ('Spin Rate', 'spin', 'RPM'),
            ('Hyzer Angle', 'hyzer_angle', '°'),
            ('Launch Angle', 'launch_angle', '°'),
            ('Nose Angle', 'nose_angle', '°'),
            ('Wobble Amplitude', 'wobble_amplitude', '°'),
            ('Max Acceleration', 'max_acceleration', 'mm/s²'),
        ]
        
        for label, key, unit in metrics:
            v1 = results1.get(key)
            v2 = results2.get(key)
            
            if v1 is not None and v2 is not None:
                diff = v2 - v1
                pct_diff = (diff / v1 * 100) if v1 != 0 else 0
                
                if unit == 'RPM':
                    print(f"{label:<30} {v1:>12.1f} {v2:>12.1f} {diff:+10.1f} ({pct_diff:+6.1f}%)")
                elif unit == 'mm/s':
                    print(f"{label:<30} {v1:>12.0f} {v2:>12.0f} {diff:+10.0f} ({pct_diff:+6.1f}%)")
                elif unit == '°':
                    print(f"{label:<30} {v1:>12.1f} {v2:>12.1f} {diff:+10.1f} ({pct_diff:+6.1f}%)")
                else:
                    print(f"{label:<30} {v1:>12.4f} {v2:>12.4f} {diff:+10.4f}")
    
    # Generate visualizations
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    visualizer = DiscVisualizer6DOF()
    
    if results1 and body1 is not None:
        print("\nThrow 1:")
        try:
            fig = visualizer.plot_6dof_motion_analysis(body1, title="Throw 1 - Motion Analysis (Improved Detection)")
            import matplotlib.pyplot as plt
            plt.savefig("throw1_improved_motion_analysis.png", dpi=150, bbox_inches='tight')
            print("  ✓ Generated throw1_improved_motion_analysis.png")
            plt.close(fig)
        except Exception as e:
            print(f"  Error: {e}")
    
    if results2 and body2 is not None:
        print("\nThrow 2:")
        try:
            fig = visualizer.plot_6dof_motion_analysis(body2, title="Throw 2 - Motion Analysis (Improved Detection)")
            import matplotlib.pyplot as plt
            plt.savefig("throw2_improved_motion_analysis.png", dpi=150, bbox_inches='tight')
            print("  ✓ Generated throw2_improved_motion_analysis.png")
            plt.close(fig)
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "=" * 70 + "\n")


if __name__ == "__main__":
    main()
