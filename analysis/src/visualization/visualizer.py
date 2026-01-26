"""Visualization module for disc golf run-up and early flight analysis."""

from typing import Dict, Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation


class ThrowVisualizer:
    """Create visualizations for early-phase throw analysis results."""
    
    @staticmethod
    def plot_early_phase_3d(
        disc_positions: Dict[str, np.ndarray],
        max_duration: float = 1.0,
        frame_rate: float = 240.0,
        title: str = "Early Phase Disc Motion (3D)"
    ) -> Figure:
        """
        Create 3D visualization of disc motion during run-up and early flight.
        
        Args:
            disc_positions: Dictionary with disc position data
            max_duration: Maximum time to display (seconds, default 1.0)
            frame_rate: Capture frame rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        # Limit to early phase
        max_frames = int(max_duration * frame_rate)
        
        colors = {'Wide': 'red', 'Mid': 'orange', 'Center': 'green', 'Close': 'blue'}
        
        for position, positions in disc_positions.items():
            # Filter NaN values
            valid_mask = ~np.isnan(positions).any(axis=1)
            valid_positions = positions[valid_mask]
            
            # Limit to early phase
            valid_positions = valid_positions[:max_frames]
            
            if len(valid_positions) > 0:
                ax.plot(
                    valid_positions[:, 0],
                    valid_positions[:, 1],
                    valid_positions[:, 2],
                    label=position,
                    color=colors.get(position, 'black'),
                    linewidth=2
                )
                # Mark start and end
                ax.scatter(*valid_positions[0], s=100, marker='o', color=colors.get(position))
                if len(valid_positions) > 1:
                    ax.scatter(*valid_positions[-1], s=100, marker='s', color=colors.get(position))
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        
        return fig
    
    @staticmethod
    def plot_acceleration_phase(
        disc_positions: Dict[str, np.ndarray],
        frame_rate: float = 240.0,
        title: str = "Acceleration Phase Analysis"
    ) -> Figure:
        """
        Plot disc motion during acceleration phase.
        
        Args:
            disc_positions: Dictionary with disc position data
            frame_rate: Capture frame rate in Hz
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        dt = 1.0 / frame_rate
        colors = {'Wide': 'red', 'Mid': 'orange', 'Center': 'green', 'Close': 'blue'}
        
        # Speed profile
        ax = axes[0, 0]
        for position, positions in disc_positions.items():
            valid_mask = ~np.isnan(positions).any(axis=1)
            valid_positions = positions[valid_mask]
            
            if len(valid_positions) > 1:
                velocities = np.diff(valid_positions, axis=0) / dt
                speeds = np.linalg.norm(velocities, axis=1)
                
                time = np.arange(len(speeds)) / frame_rate
                ax.plot(time, speeds, label=position, color=colors.get(position), linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (mm/s)')
        ax.set_title('Velocity Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Acceleration profile
        ax = axes[0, 1]
        for position, positions in disc_positions.items():
            valid_mask = ~np.isnan(positions).any(axis=1)
            valid_positions = positions[valid_mask]
            
            if len(valid_positions) > 2:
                velocities = np.diff(valid_positions, axis=0) / dt
                accelerations = np.linalg.norm(np.diff(velocities, axis=0) / dt, axis=1)
                
                time = np.arange(len(accelerations)) / frame_rate
                ax.plot(time, accelerations, label=position, color=colors.get(position), linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (mm/s²)')
        ax.set_title('Acceleration Profile')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Displacement from start
        ax = axes[1, 0]
        if 'Center' in disc_positions:
            center = disc_positions['Center']
            valid_mask = ~np.isnan(center).any(axis=1)
            center = center[valid_mask]
            
            if len(center) > 0:
                displacement = np.linalg.norm(center - center[0], axis=1)
                time = np.arange(len(displacement)) / frame_rate
                
                ax.plot(time, displacement, linewidth=2, color='darkgreen')
                ax.fill_between(time, 0, displacement, alpha=0.3, color='lightgreen')
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Displacement (mm)')
                ax.set_title('Distance Traveled from Start')
                ax.grid(True, alpha=0.3)
        
        # Release point detection
        ax = axes[1, 1]
        if 'Center' in disc_positions:
            center = disc_positions['Center']
            valid_mask = ~np.isnan(center).any(axis=1)
            center = center[valid_mask]
            
            if len(center) > 1:
                velocities = np.diff(center, axis=0) / dt
                speeds = np.linalg.norm(velocities, axis=1)
                time = np.arange(len(speeds)) / frame_rate
                
                # Find peak velocity
                peak_idx = np.argmax(speeds)
                
                ax.plot(time, speeds, linewidth=2, color='navy')
                ax.axvline(x=time[peak_idx], color='red', linestyle='--', linewidth=2, 
                          label=f'Release: {time[peak_idx]:.4f}s')
                ax.scatter(time[peak_idx], speeds[peak_idx], s=200, color='red', zorder=5)
                
                ax.set_xlabel('Time (s)')
                ax.set_ylabel('Speed (mm/s)')
                ax.set_title('Release Point (Peak Velocity)')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_release_parameters(
        disc_analysis: Dict,
        throw_name: str = "Release Parameters"
    ) -> Figure:
        """
        Create comprehensive visualization of 6 key release parameters.
        
        Args:
            disc_analysis: Results from DiscAnalyzer.analyze_disc_trajectory()
            throw_name: Name of the throw being analyzed
            
        Returns:
            Matplotlib figure with subplots
        """
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"{throw_name} - 6 Key Release Parameters", fontsize=14, fontweight='bold')
        
        # (1) Disc Speed
        ax = axes[0, 0]
        speed = disc_analysis.get('disc_speed', 0)
        ax.barh(['Disc Speed'], [speed], color='#3498db', edgecolor='black', linewidth=2)
        ax.set_xlabel('Velocity (mm/s)')
        ax.set_title('(1) Disc Speed at Release', fontweight='bold')
        ax.text(speed + 100, 0, f'{speed:.0f} mm/s', va='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # (2) Spin Rate
        ax = axes[0, 1]
        spin = disc_analysis.get('spin', 0)
        ax.barh(['Spin Rate'], [spin], color='#e74c3c', edgecolor='black', linewidth=2)
        ax.set_xlabel('RPM')
        ax.set_title('(2) Spin Rate at Release', fontweight='bold')
        ax.text(spin + 100, 0, f'{spin:.0f} RPM', va='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # (3) Hyzer Angle
        ax = axes[0, 2]
        hyzer = disc_analysis.get('hyzer_angle', 0)
        ax.barh(['Hyzer'], [hyzer], color='#2ecc71', edgecolor='black', linewidth=2)
        ax.set_xlabel('Angle (degrees)')
        ax.set_title('(3) Hyzer Angle at Release', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.text(hyzer + 2, 0, f'{hyzer:.1f}°', va='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # (4) Launch Angle
        ax = axes[1, 0]
        launch = disc_analysis.get('launch_angle', 0)
        ax.barh(['Launch'], [launch], color='#f39c12', edgecolor='black', linewidth=2)
        ax.set_xlabel('Angle (degrees)')
        ax.set_title('(4) Launch Angle at Release', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.text(launch + 2, 0, f'{launch:.1f}°', va='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # (5) Nose Angle
        ax = axes[1, 1]
        nose = disc_analysis.get('nose_angle', 0)
        ax.barh(['Nose'], [nose], color='#9b59b6', edgecolor='black', linewidth=2)
        ax.set_xlabel('Angle (degrees)')
        ax.set_title('(5) Nose Angle at Release', fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax.text(nose + 2, 0, f'{nose:.1f}°', va='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        # (6) Wobble Amplitude
        ax = axes[1, 2]
        wobble = disc_analysis.get('wobble_amplitude', 0)
        ax.barh(['Wobble'], [wobble], color='#1abc9c', edgecolor='black', linewidth=2)
        ax.set_xlabel('Angle Range (degrees)')
        ax.set_title('(6) Wobble Amplitude in Early Phase', fontweight='bold')
        ax.text(wobble + 2, 0, f'{wobble:.1f}°', va='center', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def plot_early_phase_summary(
        disc_analysis: Dict,
        throw_name: str = "Early Phase Summary"
    ) -> Figure:
        """
        Create summary dashboard for early phase analysis.
        
        Args:
            disc_analysis: Results from DiscAnalyzer.analyze_disc_trajectory()
            throw_name: Name of the throw being analyzed
            
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)
        fig.suptitle(f"{throw_name} - Early Phase Analysis", fontsize=16, fontweight='bold')
        
        # Release parameters box
        ax = fig.add_subplot(gs[0, :])
        release_text = "RELEASE PARAMETERS\n" + "="*70 + "\n"
        release_text += f"  Disc Speed: {disc_analysis.get('disc_speed', 0):.0f} mm/s  |  "
        release_text += f"Spin: {disc_analysis.get('spin', 0):.0f} RPM  |  "
        release_text += f"Hyzer: {disc_analysis.get('hyzer_angle', 0):.1f}°  |  "
        release_text += f"Launch: {disc_analysis.get('launch_angle', 0):.1f}°  |  "
        release_text += f"Nose: {disc_analysis.get('nose_angle', 0):.1f}°\n"
        
        ax.text(0.05, 0.5, release_text, fontsize=11, family='monospace',
               verticalalignment='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightcyan', alpha=0.7, pad=1))
        ax.axis('off')
        
        # Acceleration metrics
        ax = fig.add_subplot(gs[1, 0])
        accel_text = "ACCELERATION PHASE\n" + "="*35 + "\n"
        accel_text += f"Duration: {disc_analysis.get('acceleration_phase_duration', 0):.3f} s\n"
        accel_text += f"Distance: {disc_analysis.get('acceleration_distance', 0):.0f} mm\n"
        accel_text += f"Max Accel: {disc_analysis.get('max_acceleration', 0):.0f} mm/s²\n"
        accel_text += f"Avg Accel: {disc_analysis.get('average_acceleration', 0):.0f} mm/s²\n"
        
        ax.text(0.1, 0.5, accel_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        ax.axis('off')
        
        # Motion characteristics
        ax = fig.add_subplot(gs[1, 1])
        motion_text = "EARLY PHASE TIMELINE\n" + "="*35 + "\n"
        motion_text += f"Total Duration: {disc_analysis.get('early_phase_duration', 0):.3f} s\n"
        motion_text += f"Frames Analyzed: {disc_analysis.get('total_frames_analyzed', 0)}\n"
        motion_text += f"Release Height: {disc_analysis.get('release_z_position', 0):.0f} mm\n"
        motion_text += f"Release Tilt: {disc_analysis.get('release_tilt_angle', 0):.1f}°\n"
        
        ax.text(0.1, 0.5, motion_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax.axis('off')
        
        # Angular motion
        ax = fig.add_subplot(gs[2, 0])
        angular_text = "ANGULAR MOTION\n" + "="*35 + "\n"
        angular_text += f"Yaw (Spin): {disc_analysis.get('angular_velocity_yaw', 0):.2f} rad/s\n"
        angular_text += f"Pitch: {disc_analysis.get('angular_velocity_pitch', 0):.2f} rad/s\n"
        angular_text += f"Roll: {disc_analysis.get('angular_velocity_roll', 0):.2f} rad/s\n"
        angular_text += f"Wobble Amp: {disc_analysis.get('wobble_amplitude', 0):.1f}°\n"
        
        ax.text(0.1, 0.5, angular_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax.axis('off')
        
        # Spin evolution
        ax = fig.add_subplot(gs[2, 1])
        spin_text = "DISC DYNAMICS\n" + "="*35 + "\n"
        spin_evo = disc_analysis.get('spin_rate_evolution', 0)
        spin_text += f"Spin Evolution: {spin_evo:+.0f} RPM\n"
        spin_text += f"Wobble Amplitude: {disc_analysis.get('wobble_amplitude', 0):.1f}°\n"
        spin_text += f"Focus: Run-up & Early Flight\n"
        spin_text += f"(First 1 second of motion)\n"
        
        ax.text(0.1, 0.5, spin_text, fontsize=10, family='monospace',
               verticalalignment='center', transform=ax.transAxes,
               bbox=dict(boxstyle='round', facecolor='plum', alpha=0.7))
        ax.axis('off')
        
        return fig


class InteractiveVisualizer:
    """Interactive 3D visualization of disc flight with playback controls."""
    
    def __init__(self, disc_positions: Dict[str, np.ndarray],
                 frame_rate: float = 240.0):
        """
        Initialize interactive visualizer.
        
        Args:
            disc_positions: Dictionary with disc position data
            frame_rate: Capture frame rate in Hz
        """
        self.disc_positions = disc_positions
        self.frame_rate = frame_rate
        self.current_frame = 0
    
    def create_animation(self, max_frames: Optional[int] = None) -> Figure:
        """
        Create animated 3D visualization of throw.
        
        Args:
            max_frames: Limit animation to specific number of frames (optional)
            
        Returns:
            Matplotlib figure with animation
        """
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # Determine frame range
        all_positions = []
        for positions in self.disc_positions.values():
            valid_mask = ~np.isnan(positions).any(axis=1)
            all_positions.append(np.sum(valid_mask))
        
        max_frame = min(all_positions) if all_positions else 100
        if max_frames:
            max_frame = min(max_frame, max_frames)
        
        # Initialize scatter plots
        scatter_dict = {}
        colors = {'Wide': 'red', 'Mid': 'orange', 'Center': 'green', 'Close': 'blue'}
        
        for position, color in colors.items():
            scatter = ax.scatter([], [], [], color=color, label=position, s=50)
            scatter_dict[position] = scatter
        
        # Set axis limits
        all_data = np.vstack(list(self.disc_positions.values()))
        valid_mask = ~np.isnan(all_data).any(axis=1)
        if np.sum(valid_mask) > 0:
            valid_data = all_data[valid_mask]
            ax.set_xlim(valid_data[:, 0].min(), valid_data[:, 0].max())
            ax.set_ylim(valid_data[:, 1].min(), valid_data[:, 1].max())
            ax.set_zlim(valid_data[:, 2].min(), valid_data[:, 2].max())
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.legend()
        
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes)
        
        def animate(frame):
            for position, scatter in scatter_dict.items():
                if position in self.disc_positions:
                    pos = self.disc_positions[position][frame]
                    if not np.isnan(pos).any():
                        scatter._offsets3d = ([pos[0]], [pos[1]], [pos[2]])
            
            time_text.set_text(f"Time: {frame / self.frame_rate:.3f}s")
            return list(scatter_dict.values()) + [time_text]
        
        ani = FuncAnimation(fig, animate, frames=max_frame, blit=True, interval=1000/self.frame_rate)
        
        return fig
