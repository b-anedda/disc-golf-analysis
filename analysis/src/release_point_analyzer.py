"""Detailed release point detection analysis and visualization."""

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class ReleasePointAnalyzer:
    """Analyze and visualize release point detection in detail."""
    
    def __init__(self, frame_rate: float = 240.0):
        """
        Initialize release point analyzer.
        
        Args:
            frame_rate: Capture frame rate in Hz
        """
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
    
    def detect_release_frame_detailed(self, positions: np.ndarray) -> dict:
        """
        Detect release frame using multiple criteria and return detailed analysis.
        
        Args:
            positions: Position array (num_frames, 3) in mm
            
        Returns:
            Dictionary with detection results and metrics
        """
        results = {
            'release_frame': 0,
            'release_time': 0.0,
            'release_speed': 0.0,
            'release_position': positions[0],
            'acceleration_magnitude': 0.0,
            'jerk_magnitude': 0.0,
            'peak_speed': 0.0,
            'peak_accel': 0.0,
            'detection_confidence': 0.0,
            'detection_method': 'unknown',
            'all_candidates': [],
        }
        
        # Calculate velocities and accelerations
        velocities = np.diff(positions, axis=0) / self.dt
        speeds = np.linalg.norm(velocities, axis=1)
        
        accelerations = np.diff(velocities, axis=0) / self.dt
        accels = np.linalg.norm(accelerations, axis=1)
        
        if len(accelerations) > 0:
            jerks = np.diff(accelerations, axis=0) / self.dt
            jerks_mag = np.linalg.norm(jerks, axis=1)
        else:
            jerks_mag = np.array([])
        
        results['peak_speed'] = np.max(speeds) if len(speeds) > 0 else 0
        results['peak_accel'] = np.max(accels) if len(accels) > 0 else 0
        
        # Method 1: Peak speed detection
        if len(speeds) >= 3:
            peaks_speed, props_speed = find_peaks(speeds, distance=int(self.frame_rate * 0.05))
            if len(peaks_speed) > 0:
                first_peak = peaks_speed[0]
                results['all_candidates'].append({
                    'frame': first_peak,
                    'method': 'peak_speed',
                    'speed': speeds[first_peak],
                    'confidence': 0.8
                })
        
        # Method 2: Peak acceleration detection
        if len(accels) >= 3:
            peaks_accel, _ = find_peaks(accels, distance=int(self.frame_rate * 0.03))
            if len(peaks_accel) > 0:
                first_peak_accel = peaks_accel[0]
                results['all_candidates'].append({
                    'frame': first_peak_accel,
                    'method': 'peak_accel',
                    'accel': accels[first_peak_accel],
                    'confidence': 0.7
                })
        
        # Method 3: Zero-crossing of jerk (inflection point)
        if len(jerks_mag) > 0:
            # Look for where acceleration reaches maximum
            max_accel_idx = np.argmax(accels)
            results['all_candidates'].append({
                'frame': max_accel_idx,
                'method': 'max_acceleration',
                'accel': accels[max_accel_idx],
                'confidence': 0.75
            })
        
        # Method 4: Kinetic energy changes
        kinetic_energy = 0.5 * speeds**2  # Proportional (mass is constant)
        if len(kinetic_energy) >= 3:
            peaks_ke, _ = find_peaks(kinetic_energy, distance=int(self.frame_rate * 0.05))
            if len(peaks_ke) > 0:
                first_peak_ke = peaks_ke[0]
                results['all_candidates'].append({
                    'frame': first_peak_ke,
                    'method': 'kinetic_energy',
                    'confidence': 0.6
                })
        
        # Select best candidate
        if results['all_candidates']:
            # Prefer speed-based detection
            for candidate in results['all_candidates']:
                if candidate['method'] == 'peak_speed':
                    best_candidate = candidate
                    break
            else:
                # Otherwise use highest confidence
                best_candidate = max(results['all_candidates'], key=lambda x: x.get('confidence', 0))
            
            release_frame = best_candidate['frame']
            results['release_frame'] = release_frame
            results['detection_method'] = best_candidate['method']
            results['detection_confidence'] = best_candidate.get('confidence', 0.5)
            
            # Get release parameters
            if release_frame < len(speeds):
                results['release_speed'] = speeds[release_frame]
            if release_frame < len(accels):
                results['acceleration_magnitude'] = accels[release_frame]
            if release_frame < len(positions):
                results['release_position'] = positions[release_frame]
            
            results['release_time'] = release_frame * self.dt
        
        return results
    
    def plot_release_detection_analysis(
        self,
        positions: np.ndarray,
        title: str = "Release Point Detection Analysis"
    ) -> Figure:
        """
        Create comprehensive visualization of release point detection methods.
        
        Args:
            positions: Position array (num_frames, 3)
            title: Plot title
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(3, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=14, fontweight='bold')
        
        # Calculate derivatives
        velocities = np.diff(positions, axis=0) / self.dt
        speeds = np.linalg.norm(velocities, axis=1)
        
        accelerations = np.diff(velocities, axis=0) / self.dt
        accels = np.linalg.norm(accelerations, axis=1)
        
        jerks = np.diff(accelerations, axis=0) / self.dt
        jerks_mag = np.linalg.norm(jerks, axis=1)
        
        time = np.arange(len(positions)) * self.dt
        time_vel = time[:-1]
        time_accel = time[:-2]
        time_jerk = time[:-3]
        
        # Get detection results
        detection = self.detect_release_frame_detailed(positions)
        release_frame = detection['release_frame']
        
        # Panel 1: Position components
        ax = axes[0, 0]
        ax.plot(time, positions[:, 0], label='X', linewidth=2, alpha=0.7)
        ax.plot(time, positions[:, 1], label='Y', linewidth=2, alpha=0.7)
        ax.plot(time, positions[:, 2], label='Z', linewidth=2, alpha=0.7)
        ax.axvline(x=release_frame * self.dt, color='red', linestyle='--', linewidth=2, label='Release')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Position (mm)')
        ax.set_title('Position Components', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Speed profile with peaks
        ax = axes[0, 1]
        ax.plot(time_vel, speeds, linewidth=2, color='darkblue', label='Speed')
        peaks_speed, _ = find_peaks(speeds, distance=int(self.frame_rate * 0.05))
        if len(peaks_speed) > 0:
            ax.scatter(peaks_speed * self.dt, speeds[peaks_speed], s=100, color='red', 
                      marker='o', label='Speed Peaks', zorder=5)
        ax.axvline(x=release_frame * self.dt, color='red', linestyle='--', linewidth=2)
        ax.scatter(release_frame * self.dt, speeds[release_frame], s=200, color='red', 
                  marker='X', zorder=5, edgecolors='darkred', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Speed (mm/s)')
        ax.set_title('Velocity Magnitude & Peak Detection', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 3: Acceleration profile with peaks
        ax = axes[1, 0]
        ax.plot(time_accel, accels, linewidth=2, color='darkgreen', label='Acceleration')
        peaks_accel, _ = find_peaks(accels, distance=int(self.frame_rate * 0.03))
        if len(peaks_accel) > 0:
            ax.scatter(peaks_accel * self.dt, accels[peaks_accel], s=100, color='orange', 
                      marker='s', label='Acceleration Peaks', zorder=5)
        max_accel_idx = np.argmax(accels)
        ax.scatter(max_accel_idx * self.dt, accels[max_accel_idx], s=100, color='purple', 
                  marker='^', label='Max Acceleration', zorder=5)
        ax.axvline(x=release_frame * self.dt, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Acceleration (mm/s²)')
        ax.set_title('Acceleration Profile & Peaks', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Kinetic energy
        ax = axes[1, 1]
        kinetic_energy = 0.5 * speeds**2
        ax.plot(time_vel, kinetic_energy, linewidth=2, color='purple', label='Kinetic Energy')
        peaks_ke, _ = find_peaks(kinetic_energy, distance=int(self.frame_rate * 0.05))
        if len(peaks_ke) > 0:
            ax.scatter(peaks_ke * self.dt, kinetic_energy[peaks_ke], s=100, color='green', 
                      marker='D', label='KE Peaks', zorder=5)
        ax.axvline(x=release_frame * self.dt, color='red', linestyle='--', linewidth=2, label='Release')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Kinetic Energy (a.u.)')
        ax.set_title('Kinetic Energy Profile', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 5: Jerk (derivative of acceleration)
        ax = axes[2, 0]
        ax.plot(time_jerk, jerks_mag, linewidth=2, color='brown', label='Jerk Magnitude')
        ax.axvline(x=release_frame * self.dt, color='red', linestyle='--', linewidth=2, label='Release')
        ax.fill_between(time_jerk, 0, jerks_mag, alpha=0.2, color='brown')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Jerk (mm/s³)')
        ax.set_title('Jerk Profile', fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        
        # Panel 6: Summary metrics
        ax = axes[2, 1]
        ax.axis('off')
        
        summary_text = f"""RELEASE POINT ANALYSIS SUMMARY

Detection Method: {detection['detection_method']}
Confidence: {detection['detection_confidence']:.1%}

Release Frame: {detection['release_frame']}
Release Time: {detection['release_time']:.4f} s

Release Speed: {detection['release_speed']:.0f} mm/s
Peak Speed: {detection['peak_speed']:.0f} mm/s

Acceleration @ Release: {detection['acceleration_magnitude']:.0f} mm/s²
Peak Acceleration: {detection['peak_accel']:.0f} mm/s²

Release Position:
  X: {detection['release_position'][0]:.1f} mm
  Y: {detection['release_position'][1]:.1f} mm
  Z: {detection['release_position'][2]:.1f} mm

Alternative Candidates ({len(detection['all_candidates'])}):"""
        
        for i, candidate in enumerate(detection['all_candidates'][:3]):
            summary_text += f"\n  {i+1}. {candidate['method']}: frame {candidate['frame']} (conf: {candidate.get('confidence', 0):.2f})"
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        fig.tight_layout()
        return fig
