"""Example script to load and analyze throw data directly from QTM instance."""

import sys
import numpy as np
from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer
from src.visualization.visualizer import ThrowVisualizer
import matplotlib.pyplot as plt


def analyze_throw_from_qtm(measurement_file: str):
    """
    Load a throw measurement from QTM and analyze it.
    
    Args:
        measurement_file: Path to .qtm measurement file (e.g., 'Throw1.qtm')
    """
    print("=" * 70)
    print("DISC GOLF THROW ANALYSIS FROM QTM")
    print("=" * 70)
    
    # Initialize loader
    loader = QTMLoader()
    
    # Connect to QTM
    print("\n1. Connecting to QTM...")
    if not loader.connect_to_qtm():
        print("   ERROR: QTM not available. Ensure QTM is installed and running.")
        return False
    
    print("   [OK] Connected to QTM")
    
    # Load measurement
    print(f"\n2. Loading measurement: {measurement_file}")
    if not loader.load_measurement(measurement_file):
        print("   ERROR: Could not load measurement file.")
        return False
    
    print("   [OK] Measurement loaded")
    
    # Extract 6DOF body data
    print("\n3. Extracting 6DOF rigid body data...")
    body_data = loader.extract_disc_data()
    
    if body_data is None:
        print("   ERROR: Could not extract body data.")
        return False
    
    num_frames = len(body_data['position'])
    frame_rate = 240.0
    duration = num_frames / frame_rate
    
    print(f"   [OK] Extracted {num_frames} frames ({duration:.3f} seconds)")
    
    # Analyze disc flight
    print("\n4. Analyzing flight dynamics...")
    analyzer = DiscAnalyzer(frame_rate=frame_rate)
    results = analyzer.analyze_disc_trajectory(body_data)
    
    # Display results
    print("\n" + "=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    
    print("\nFLIGHT CHARACTERISTICS:")
    print(f"  • Duration: {results['flight_duration']:.3f} seconds")
    print(f"  • Distance: {results['flight_distance']:.1f} mm ({results['flight_distance']/1000:.2f}m)")
    print(f"  • Max Height: {results['max_height']:.1f} mm")
    print(f"  • Flight Angle: {results['flight_angle']:.2f}°")
    print(f"  • Glide Ratio: {results['glide_ratio']:.2f}")
    
    print("\nVELOCITY PROFILE:")
    print(f"  • Release Velocity: {results['release_velocity']:.1f} mm/s ({results['release_velocity']/1000:.2f} m/s)")
    print(f"  • Peak Velocity: {results['peak_velocity']:.1f} mm/s")
    print(f"  • Average Velocity: {results['average_velocity']:.1f} mm/s")
    
    print("\n6 KEY RELEASE PARAMETERS:")
    if results['disc_speed'] is not None:
        print(f"  1. Disc Speed: {results['disc_speed']:.1f} mm/s ({results['disc_speed']/1000:.2f} m/s)")
    if results['spin'] is not None:
        print(f"  2. Spin Rate: {results['spin']:.0f} RPM")
    if results['hyzer_angle'] is not None:
        print(f"  3. Hyzer Angle: {results['hyzer_angle']:.2f}°")
    if results['launch_angle'] is not None:
        print(f"  4. Launch Angle: {results['launch_angle']:.2f}°")
    if results['nose_angle'] is not None:
        print(f"  5. Nose Angle: {results['nose_angle']:.2f}°")
    if results['wobble_amplitude'] is not None:
        print(f"  6. Wobble Amplitude (Roll Range): {results['wobble_amplitude']:.2f}°")
    
    print("\nDISC DYNAMICS:")
    if results['spin_rate']:
        print(f"  • Spin Rate: {results['spin_rate']:.0f} RPM")
    if results['angular_velocity_yaw'] is not None:
        print(f"  • Angular Velocity (Yaw): {results['angular_velocity_yaw']:.2f} rad/s")
    if results['stability_index']:
        stability = "Excellent" if results['stability_index'] > 2 else (
                    "Good" if results['stability_index'] > 1 else "Variable")
        print(f"  • Stability Index: {results['stability_index']:.2f} ({stability})")
    if results['wobble_frequency']:
        print(f"  • Wobble Frequency: {results['wobble_frequency']:.1f} Hz")
    
    print("\nRELEASE CHARACTERISTICS:")
    print(f"  • Release Height: {results['release_z_position']:.0f} mm")
    if results['release_tilt_angle']:
        print(f"  • Release Tilt Angle: {results['release_tilt_angle']:.1f}°")
    
    # Create visualizations
    print("\n5. Creating visualizations...")
    try:
        visualizer = ThrowVisualizer()
        
        # 3D trajectory plot
        fig1 = visualizer.plot_disc_trajectory_3d(body_data)
        plt.show()
        
        # Stability metrics
        fig2 = visualizer.plot_disc_stability_metrics(body_data)
        plt.show()
        
        # Analysis summary
        fig3 = visualizer.plot_analysis_summary(body_data, results)
        plt.show()
        
        print("   [OK] Visualizations created and displayed")
    except Exception as e:
        print(f"   [WARNING] Visualization error: {e}")
    
    return True


if __name__ == "__main__":
    # Analyze different throws
    throws = [
        "c:\\Users\\Contemplas\\Documents\\DiscGolf\\Data\\Static_Disc.qtm",
        "c:\\Users\\Contemplas\\Documents\\DiscGolf\\Data\\Throw1.qtm",
        "c:\\Users\\Contemplas\\Documents\\DiscGolf\\Data\\throw2.qtm",
    ]
    
    print("Available throws to analyze:")
    for i, throw in enumerate(throws, 1):
        print(f"  {i}. {throw.split(chr(92))[-1]}")
    
    # Analyze first available throw
    for throw_file in throws:
        print(f"\nAttempting to analyze: {throw_file.split(chr(92))[-1]}")
        if analyze_throw_from_qtm(throw_file):
            break
