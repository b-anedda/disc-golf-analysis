"""Visualization module for disc golf 6DOF analysis - direct rigid body tracking."""

from typing import Dict, Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


class DiscVisualizer6DOF:
    """Create visualizations for 6DOF disc dynamics analysis."""
    
    @staticmethod
    def plot_3d_trajectory_with_orientation(
        body_data: Dict[str, np.ndarray],
        max_duration: float = 1.0,
        frame_rate: float = 240.0,
        title: str = "Disc 6DOF Trajectory - Early Phase"
    ) -> Figure:
        """
        Create 3D visualization of disc position with orientation indicators.
        
        Args:
            body_data: Dictionary with 'position' (num_frames, 3) and 'rotation' (num_frames, 3, 3)
            max_duration: Maximum time to display (seconds)
            frame_rate: Capture frame rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        positions = body_data['position']
        rotations = body_data['rotation']
        
        # Limit to early phase
        max_frames = int(max_duration * frame_rate)
        positions = positions[:max_frames]
        rotations = rotations[:max_frames]
        
        # Filter NaN values
        valid_mask = ~np.isnan(positions).any(axis=1)
        positions = positions[valid_mask]
        rotations = rotations[valid_mask]
        
        if len(positions) == 0:
            return fig
        
        # Plot trajectory
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                'b-', linewidth=2, label='Disc Center Path')
        
        # Mark start and end
        ax.scatter(*positions[0], s=200, marker='o', c='green', edgecolors='darkgreen', 
                   linewidth=2, label='Start')
        ax.scatter(*positions[-1], s=200, marker='X', c='red', edgecolors='darkred', 
                   linewidth=2, label='End')
        
        # Plot orientation vectors at regular intervals
        interval = max(1, len(positions) // 10)  # Show ~10 orientation arrows
        for i in range(0, len(positions), interval):
            R = rotations[i]
            pos = positions[i]
            
            # Disc normal (Z-axis) - blue
            normal = R @ np.array([0, 0, 1]) * 100
            ax.quiver(pos[0], pos[1], pos[2], normal[0], normal[1], normal[2],
                     color='blue', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.6)
            
            # Disc forward (X-axis) - red
            forward = R @ np.array([1, 0, 0]) * 80
            ax.quiver(pos[0], pos[1], pos[2], forward[0], forward[1], forward[2],
                     color='red', arrow_length_ratio=0.3, linewidth=1.5, alpha=0.6)
        
        ax.set_xlabel('X (mm)', fontsize=11)
        ax.set_ylabel('Y (mm)', fontsize=11)
        ax.set_zlabel('Z (mm)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        return fig
    
    @staticmethod
    def plot_6dof_motion_analysis(
        body_data: Dict[str, np.ndarray],
        frame_rate: float = 240.0,
        title: str = "6DOF Motion Analysis"
    ) -> Figure:
        """
        Create 4-panel analysis of disc motion parameters.
        
        Args:
            body_data: Dictionary with 'position' and 'rotation' data
            frame_rate: Capture frame rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        positions = body_data['position']
        rotations = body_data['rotation']
        
        # Filter NaN
        valid_mask = ~np.isnan(positions).any(axis=1)
        positions = positions[valid_mask]
        rotations = rotations[valid_mask]
        
        if len(positions) < 2:
            return fig
        
        dt = 1.0 / frame_rate
        time = np.arange(len(positions)) * dt
        
        # (1) Position over time
        ax = axes[0, 0]
        ax.plot(time, positions[:, 0], label='X', linewidth=2, color='red', alpha=0.7)
        ax.plot(time, positions[:, 1], label='Y', linewidth=2, color='green', alpha=0.7)
        ax.plot(time, positions[:, 2], label='Z', linewidth=2, color='blue', alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Position (mm)', fontsize=10)
        ax.set_title('Position Components', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # (2) Velocity magnitude
        ax = axes[0, 1]
        velocities = np.diff(positions, axis=0) / dt
        speeds = np.linalg.norm(velocities, axis=1)
        ax.plot(time[:-1], speeds, linewidth=2, color='darkblue')
        ax.fill_between(time[:-1], 0, speeds, alpha=0.3, color='cyan')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Speed (mm/s)', fontsize=10)
        ax.set_title('Velocity Magnitude', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # (3) Distance from origin
        ax = axes[1, 0]
        distances = np.linalg.norm(positions - positions[0], axis=1)
        ax.plot(time, distances, linewidth=2, color='darkgreen')
        ax.fill_between(time, 0, distances, alpha=0.3, color='lightgreen')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Distance from Start (mm)', fontsize=10)
        ax.set_title('Displacement', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # (4) Euler angles from rotation matrices
        ax = axes[1, 1]
        euler_angles = []
        for R in rotations:
            # Extract Euler angles (YXZ convention)
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            euler_angles.append([yaw, pitch, roll])
        
        euler_angles = np.array(euler_angles)
        ax.plot(time, euler_angles[:, 0], label='Yaw', linewidth=2, color='red', alpha=0.7)
        ax.plot(time, euler_angles[:, 1], label='Pitch', linewidth=2, color='green', alpha=0.7)
        ax.plot(time, euler_angles[:, 2], label='Roll', linewidth=2, color='blue', alpha=0.7)
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Angle (degrees)', fontsize=10)
        ax.set_title('Euler Angles', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_angular_velocity_profile(
        body_data: Dict[str, np.ndarray],
        frame_rate: float = 240.0,
        title: str = "Angular Velocity Profile"
    ) -> Figure:
        """
        Plot angular velocity components derived from rotation matrix changes.
        
        Args:
            body_data: Dictionary with 'rotation' data
            frame_rate: Capture frame rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        rotations = body_data['rotation']
        dt = 1.0 / frame_rate
        time = np.arange(len(rotations)) * dt
        
        # Calculate angular velocities
        angular_velocities = []
        for i in range(1, len(rotations)):
            R_dot = (rotations[i] - rotations[i-1]) / dt
            # Skew-symmetric part: [ω]× = R_dot * R^T
            skew = R_dot @ rotations[i].T
            
            # Extract components
            wx = skew[2, 1]
            wy = skew[0, 2]
            wz = skew[1, 0]
            
            angular_velocities.append([wz, wy, wx])  # yaw (rad/s), pitch (rad/s), roll (rad/s)
        
        angular_velocities = np.array(angular_velocities)
        
        # (1) Yaw (Z-axis spin)
        ax = axes[0, 0]
        yaw_rpm = angular_velocities[:, 0] * 60.0 / (2.0 * np.pi)
        ax.plot(time[:-1], yaw_rpm, linewidth=2, color='blue')
        ax.fill_between(time[:-1], 0, yaw_rpm, alpha=0.3, color='cyan')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Spin (RPM)', fontsize=10)
        ax.set_title('Yaw (Z-axis Spin)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # (2) Pitch (Y-axis)
        ax = axes[0, 1]
        ax.plot(time[:-1], angular_velocities[:, 1], linewidth=2, color='green')
        ax.fill_between(time[:-1], 0, angular_velocities[:, 1], alpha=0.3, color='lightgreen')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=10)
        ax.set_title('Pitch (Y-axis)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # (3) Roll (X-axis)
        ax = axes[1, 0]
        ax.plot(time[:-1], angular_velocities[:, 2], linewidth=2, color='red')
        ax.fill_between(time[:-1], 0, angular_velocities[:, 2], alpha=0.3, color='lightcoral')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Angular Velocity (rad/s)', fontsize=10)
        ax.set_title('Roll (X-axis)', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # (4) Magnitude of angular velocity
        ax = axes[1, 1]
        # Convert to RPM for magnitude as well (using yaw component)
        ang_vel_rpm = np.abs(angular_velocities[:, 0]) * 60.0 / (2.0 * np.pi)
        ax.plot(time[:-1], ang_vel_rpm, linewidth=2, color='purple')
        ax.fill_between(time[:-1], 0, ang_vel_rpm, alpha=0.3, color='plum')
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Z-axis Spin (RPM)', fontsize=10)
        ax.set_title('Total Z-axis Spin Rate', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_release_parameters_6dof(
        analysis_results: Dict,
        throw_name: str = "6DOF Release Parameters"
    ) -> Figure:
        """
        Create 6-panel visualization of key release parameters.
        
        Args:
            analysis_results: Results from DiscAnalyzer.analyze_disc_trajectory()
            throw_name: Name of throw being analyzed
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{throw_name} - 6 Key Release Parameters", fontsize=14, fontweight='bold')
        
        params = [
            (axes[0, 0], 'disc_speed', 'Disc Speed (mm/s)', '#3498db', 0),
            (axes[0, 1], 'spin', 'Spin Rate (RPM)', '#e74c3c', 1),
            (axes[0, 2], 'hyzer_angle', 'Hyzer Angle +hyzer (deg)', '#2ecc71', 2),
            (axes[1, 0], 'launch_angle', 'Launch Angle (°)', '#f39c12', 3),
            (axes[1, 1], 'nose_angle', 'Nose Angle +up (deg)', '#9b59b6', 4),
            (axes[1, 2], 'wobble_amplitude', 'Wobble Amplitude (deg)', '#1abc9c', 5),
        ]
        
        for ax, key, label, color, idx in params:
            value = analysis_results.get(key)
            if value is not None:
                ax.barh([''], [value], color=color, edgecolor='black', linewidth=2, height=0.5)
                ax.set_xlim(0, max(abs(value) * 1.3, 1))
                ax.text(value + abs(value)*0.05, 0, f'{value:.1f}', va='center', fontweight='bold')
            ax.set_title(f"({idx+1}) {label.split('(')[0].strip()}", fontweight='bold')
            ax.set_yticks([])
            ax.grid(True, alpha=0.3, axis='x')
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_6dof_comprehensive_dashboard(
        body_data: Dict[str, np.ndarray],
        analysis_results: Dict,
        frame_rate: float = 240.0,
        title: str = "6DOF Comprehensive Analysis Dashboard"
    ) -> Figure:
        """
        Create comprehensive multi-panel dashboard combining motion and release parameters.
        
        Args:
            body_data: Dictionary with 'position' and 'rotation' data
            analysis_results: Results from DiscAnalyzer
            frame_rate: Capture frame rate in Hz
            title: Dashboard title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        positions = body_data['position']
        rotations = body_data['rotation']
        
        # Filter NaN
        valid_mask = ~np.isnan(positions).any(axis=1)
        positions = positions[valid_mask]
        rotations = rotations[valid_mask]
        
        dt = 1.0 / frame_rate
        time = np.arange(len(positions)) * dt
        
        # (1) 3D trajectory (large plot)
        ax = fig.add_subplot(gs[0:2, 0:2], projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 'b-', linewidth=2)
        ax.scatter(*positions[0], s=150, c='green', marker='o', edgecolors='darkgreen')
        ax.scatter(*positions[-1], s=150, c='red', marker='X', edgecolors='darkred')
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title('3D Disc Trajectory', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # (2) Velocity profile
        ax = fig.add_subplot(gs[0, 2])
        velocities = np.diff(positions, axis=0) / dt
        speeds = np.linalg.norm(velocities, axis=1)
        ax.plot(time[:-1], speeds, linewidth=2, color='darkblue')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Speed (mm/s)', fontsize=9)
        ax.set_title('Speed Profile', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # (3) Displacement
        ax = fig.add_subplot(gs[1, 2])
        distances = np.linalg.norm(positions - positions[0], axis=1)
        ax.plot(time, distances, linewidth=2, color='darkgreen')
        ax.fill_between(time, 0, distances, alpha=0.3, color='lightgreen')
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Distance (mm)', fontsize=9)
        ax.set_title('Displacement', fontweight='bold', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # (4) Euler angles
        ax = fig.add_subplot(gs[2, 0])
        euler_angles = []
        for R in rotations:
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            pitch = np.degrees(np.arctan2(-R[2, 0], sy))
            roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
            euler_angles.append([yaw, pitch, roll])
        euler_angles = np.array(euler_angles)
        ax.plot(time, euler_angles[:, 0], label='Yaw', linewidth=2)
        ax.plot(time, euler_angles[:, 1], label='Pitch', linewidth=2)
        ax.plot(time, euler_angles[:, 2], label='Roll', linewidth=2)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Angle (°)', fontsize=9)
        ax.set_title('Rotation Angles', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        # (5-7) Release parameters (key metrics)
        ax = fig.add_subplot(gs[2, 1])
        ax.axis('off')
        metrics_text = f"""RELEASE PARAMETERS
        
Speed: {analysis_results.get('disc_speed', 0):.0f} mm/s
Spin: {analysis_results.get('spin', 0):.0f} RPM
Hyzer (+hyzer): {analysis_results.get('hyzer_angle', 0):.1f}°
Launch: {analysis_results.get('launch_angle', 0):.1f}°
Nose (+up): {analysis_results.get('nose_angle', 0):.1f}°
Wobble: {analysis_results.get('wobble_amplitude', 0):.1f}°
"""
        ax.text(0.1, 0.5, metrics_text, fontsize=10, verticalalignment='center',
                family='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # (8) Angular velocity components
        ax = fig.add_subplot(gs[2, 2])
        angular_velocities = []
        for i in range(1, len(rotations)):
            R_dot = (rotations[i] - rotations[i-1]) / dt
            skew = R_dot @ rotations[i].T
            wx = skew[2, 1]
            wy = skew[0, 2]
            wz = skew[1, 0]
            angular_velocities.append([wz, wy, wx])
        angular_velocities = np.array(angular_velocities)
        
        # Convert yaw to RPM, keep pitch and roll in rad/s
        yaw_rpm = angular_velocities[:, 0] * 60.0 / (2.0 * np.pi)
        ax.plot(time[:-1], yaw_rpm, label='Yaw (RPM)', linewidth=1.5)
        ax.plot(time[:-1], angular_velocities[:, 1], label='Pitch (rad/s)', linewidth=1.5)
        ax.plot(time[:-1], angular_velocities[:, 2], label='Roll (rad/s)', linewidth=1.5)
        ax.set_xlabel('Time (s)', fontsize=9)
        ax.set_ylabel('Angular Velocity', fontsize=9)
        ax.set_title('Angular Velocity (Mixed Units)', fontweight='bold', fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        
        return fig
