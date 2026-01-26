"""Example script for disc golf flight analysis workflow."""

import sys
import numpy as np
from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer
from src.visualization.visualizer import ThrowVisualizer
import matplotlib.pyplot as plt


def example_disc_analysis():
    """
    Example: Load QTM 6DOF rigid body data and perform comprehensive flight analysis.
    """
    print("=" * 70)
    print("DISC GOLF FLIGHT ANALYSIS - 6DOF Rigid Body Example")
    print("=" * 70)
    
    # Initialize loader
    loader = QTMLoader()
    
    # Connect to QTM
    print("\n1. Connecting to QTM...")
    if not loader.connect_to_qtm():
        print("   WARNING: QTM API not available. Using synthetic data for demo.")
        # Generate synthetic data
        body_data = generate_demo_body_data()
    else:
        # Load measurement from QTM file
        measurement_path = "c:\\Users\\Contemplas\\Documents\\DiscGolf\\Data\\Throw1.qtm"
        print(f"2. Loading measurement: {measurement_path}")
        if loader.load_measurement(measurement_path):
            print("   [OK] Measurement loaded")
            
            # Extract 6DOF rigid body data
            print("\n3. Extracting disc 6DOF rigid body data from 'Discgolf_Midrange_Hex'...")
            body_data = loader.extract_disc_data()
            if body_data is not None:
                num_frames = len(body_data['position'])
                print(f"   [OK] Loaded 6DOF body data ({num_frames} frames)")
                frame_rate = 240.0
                duration = num_frames / frame_rate
                print(f"       Flight duration: {duration:.3f} seconds")
            else:
                print("   [FAILED] Failed to load body data, using synthetic")
                body_data = generate_demo_body_data()
        else:
            print("   [FAILED] Failed to load measurement, using synthetic")
            body_data = generate_demo_body_data()
    
    # Analyze disc flight
    print("\n4. Analyzing disc flight dynamics from 6DOF data...")
    analyzer = DiscAnalyzer(frame_rate=240.0)
    disc_analysis = analyzer.analyze_disc_trajectory(body_data)
    
    # Print comprehensive results
    print("\n" + "="*70)
    print("DISC FLIGHT ANALYSIS RESULTS")
    print("="*70)
    
    print("\nFLIGHT CHARACTERISTICS:")
    print(f"  • Duration: {disc_analysis['flight_duration']:.3f} seconds")
    print(f"  • Distance: {disc_analysis['flight_distance']:.1f} mm ({disc_analysis['flight_distance']/1000:.2f}m)")
    print(f"  • Max Height: {disc_analysis['max_height']:.1f} mm")
    print(f"  • Flight Angle: {disc_analysis['flight_angle']:.2f}°")
    print(f"  • Glide Ratio: {disc_analysis['glide_ratio']:.2f}")
    
    print("\nVELOCITY PROFILE:")
    print(f"  • Release Velocity: {disc_analysis['release_velocity']:.1f} mm/s")
    print(f"  • Peak Velocity: {disc_analysis['peak_velocity']:.1f} mm/s")
    print(f"  • Average Velocity: {disc_analysis['average_velocity']:.1f} mm/s")
    
    print("\n6 KEY RELEASE PARAMETERS:")
    if disc_analysis['disc_speed'] is not None:
        print(f"  1. Disc Speed: {disc_analysis['disc_speed']:.1f} mm/s")
    if disc_analysis['spin'] is not None:
        print(f"  2. Spin Rate: {disc_analysis['spin']:.0f} RPM")
    if disc_analysis['hyzer_angle'] is not None:
        print(f"  3. Hyzer Angle: {disc_analysis['hyzer_angle']:.2f}°")
    if disc_analysis['launch_angle'] is not None:
        print(f"  4. Launch Angle: {disc_analysis['launch_angle']:.2f}°")
    if disc_analysis['nose_angle'] is not None:
        print(f"  5. Nose Angle: {disc_analysis['nose_angle']:.2f}°")
    if disc_analysis['wobble_amplitude'] is not None:
        print(f"  6. Wobble Amplitude: {disc_analysis['wobble_amplitude']:.2f}°")
    
    print("\nDISC DYNAMICS:")
    if disc_analysis['spin_rate']:
        print(f"  • Spin Rate (legacy): {disc_analysis['spin_rate']:.0f} RPM")
        print(f"  • Angular Velocity (Yaw): {disc_analysis['angular_velocity_yaw']:.2f} rad/s")
    
    if disc_analysis['stability_index']:
        stability = "Excellent" if disc_analysis['stability_index'] > 2 else (
                    "Good" if disc_analysis['stability_index'] > 1 else "Variable")
        print(f"  • Stability Index: {disc_analysis['stability_index']:.2f} ({stability})")
    
    if disc_analysis['wobble_frequency']:
        print(f"  • Wobble Frequency: {disc_analysis['wobble_frequency']:.1f} Hz")
    
    print("\nRELEASE CHARACTERISTICS:")
    print(f"  • Release Height: {disc_analysis['release_z_position']:.0f} mm")
    if disc_analysis['release_tilt_angle']:
        print(f"  • Release Tilt Angle: {disc_analysis['release_tilt_angle']:.1f}°")
    
    # Visualize results
    print("\n5. Creating visualizations...")
    visualizer = ThrowVisualizer()
    
    # Create a figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))
    
    # Extract position and rotation from body data
    positions = body_data['position']
    rotations = body_data['rotation']
    
    # 3D trajectory (body center path)
    ax = fig.add_subplot(2, 3, 1, projection='3d')
    valid_mask = ~np.isnan(positions).any(axis=1)
    valid_pos = positions[valid_mask]
    if len(valid_pos) > 0:
        ax.plot(valid_pos[:, 0], valid_pos[:, 1], valid_pos[:, 2],
               'b-', linewidth=2, label='Body Path')
        ax.scatter(*valid_pos[0], s=100, c='green', marker='o', label='Release')
        ax.scatter(*valid_pos[-1], s=100, c='red', marker='x', label='Landing')
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_title('3D Flight Trajectory (6DOF Body)')
    ax.legend()
    
    # Height vs Time
    ax = fig.add_subplot(2, 3, 2)
    if len(valid_pos) > 0:
        time = np.arange(len(valid_pos)) / 240.0
        ax.plot(time, valid_pos[:, 2], linewidth=2, color='blue')
        ax.fill_between(time, valid_pos[:, 2].min(), valid_pos[:, 2], alpha=0.3)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Height (mm)')
        ax.set_title('Height Profile')
        ax.grid(True, alpha=0.3)
    
    # Distance vs Time
    ax = fig.add_subplot(2, 3, 3)
    if len(valid_pos) > 0:
        time = np.arange(len(valid_pos)) / 240.0
        distances = np.linalg.norm(valid_pos[:, :2] - valid_pos[0, :2], axis=1)
        ax.plot(time, distances, linewidth=2, color='red')
        ax.fill_between(time, 0, distances, alpha=0.3, color='red')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Distance (mm)')
        ax.set_title('Distance Profile')
        ax.grid(True, alpha=0.3)
    
    # Velocity profile
    ax = fig.add_subplot(2, 3, 4)
    dt = 1.0 / 240.0
    if len(valid_pos) > 1:
        velocities = np.diff(valid_pos, axis=0) / dt
        speeds = np.linalg.norm(velocities, axis=1)
        time = np.arange(len(speeds)) / 240.0
        ax.plot(time, speeds, linewidth=2, color='green')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Speed (mm/s)')
    ax.set_title('Velocity Profile')
    ax.grid(True, alpha=0.3)
    
    # Angular motion (yaw/pitch/roll from rotations)
    ax = fig.add_subplot(2, 3, 5)
    time_rot = np.arange(len(rotations)) / 240.0
    yaws = []
    pitches = []
    
    for R in rotations:
        sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
        yaw = np.arctan2(R[1, 0], R[0, 0])
        pitch = np.arctan2(-R[2, 0], sy)
        yaws.append(np.degrees(yaw))
        pitches.append(np.degrees(pitch))
    
    ax.plot(time_rot, yaws, linewidth=2, label='Yaw (Spin)', color='blue')
    ax.plot(time_rot, pitches, linewidth=2, label='Pitch (Wobble)', color='red')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (degrees)')
    ax.set_title('Disc Orientation')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Summary metrics
    ax = fig.add_subplot(2, 3, 6)
    metrics_text = "FLIGHT SUMMARY\n" + "="*25 + "\n"
    metrics_text += f"Distance: {disc_analysis['flight_distance']/1000:.2f}m\n"
    metrics_text += f"Height: {disc_analysis['max_height']/1000:.2f}m\n"
    metrics_text += f"Duration: {disc_analysis['flight_duration']:.2f}s\n"
    metrics_text += f"Release Vel: {disc_analysis['release_velocity']:.0f}mm/s\n"
    metrics_text += f"Angle: {disc_analysis['flight_angle']:.1f}°\n"
    if disc_analysis['stability_index']:
        metrics_text += f"Stability: {disc_analysis['stability_index']:.2f}\n"
    if disc_analysis['spin_rate']:
        metrics_text += f"Spin: {disc_analysis['spin_rate']:.0f} RPM\n"
    
    ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes,
           fontsize=10, verticalalignment='top', family='monospace',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("   [OK] Visualizations created and displayed")



def generate_demo_body_data():
    """Generate synthetic 6DOF rigid body data for demonstration."""
    print("   Generating synthetic 6DOF body data...")
    
    # Flight parameters
    num_frames = 150  # ~0.625 seconds at 240 Hz
    t = np.linspace(0, 0.625, num_frames)
    
    # Parabolic trajectory (center of mass)
    x = 6000 * t  # Forward distance  
    y = 400 * np.sin(2 * np.pi * t / 0.625)  # Wobble
    z = 1500 + 4500 * t - 3600 * t**2  # Vertical arc
    
    positions = np.column_stack([x, y, z])
    
    # Generate rotation matrices representing disc orientation
    # Primarily spinning around Z-axis with slight pitch and roll
    rotations = []
    for i, frame_t in enumerate(t):
        # Yaw (spin around Z) - main component
        yaw = 2 * np.pi * 200 * frame_t  # 200 RPM spin
        
        # Pitch (nose angle) - slight wobble
        pitch = 0.1 * np.sin(2 * np.pi * frame_t / 0.625)
        
        # Roll (banking) - slight variation
        roll = 0.05 * np.cos(2 * np.pi * frame_t / 0.625)
        
        # Build rotation matrix from Euler angles (Z-Y-X convention)
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        
        R = Rz @ Ry @ Rx
        rotations.append(R)
    
    rotations = np.array(rotations)
    
    # Residual values (model fit quality)
    residuals = np.random.normal(0.5, 0.1, num_frames)
    residuals = np.clip(residuals, 0, 2)
    
    body_data = {
        'position': positions,
        'rotation': rotations,
        'residual': residuals
    }
    
    return body_data



if __name__ == "__main__":
    example_disc_analysis()
