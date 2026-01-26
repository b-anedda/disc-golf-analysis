"""Analyze disc dynamics during run-up and early flight phase using 6DOF rigid body data.

This module focuses on the initial acceleration phase (run-up) and early flight (<1 second),
extracting key disc parameters at release. No thrower biomechanics analysis.
"""

from typing import Dict, Optional, Tuple
import numpy as np
from scipy.signal import find_peaks, butter, sosfiltfilt


class DiscAnalyzer:
    """Analyze disc dynamics during run-up and early flight phase from 6DOF tracking."""
    
    # Configuration
    EARLY_PHASE_DURATION = 1.0  # Seconds - capture first 1 second of motion
    DEFAULT_FRAME_RATE = 240.0  # Hz - standard high-speed capture
    
    def __init__(self, frame_rate: float = 240.0):
        """
        Initialize disc analyzer for run-up and early flight analysis.
        
        Args:
            frame_rate: Capture frame rate in Hz (default: 240 Hz for high-speed)
        """
        self.frame_rate = frame_rate
        self.dt = 1.0 / frame_rate
    
    def analyze_disc_trajectory(self, body_data: Dict[str, np.ndarray]) -> Dict:
        """
        Analyze disc dynamics during run-up and early flight phase.
        
        Focuses on the initial acceleration phase and first second of flight,
        analyzing disc-only parameters at release.
        
        Args:
            body_data: Dictionary from QTMLoader with keys:
                - 'position': shape (num_frames, 3) - XYZ in mm
                - 'rotation': shape (num_frames, 3, 3) - rotation matrices
                - 'residual': shape (num_frames,) - model fit quality
            
        Returns:
            Dictionary with release parameters and early-phase disc metrics
        """
        results = {
            # Early phase duration and timeline
            'early_phase_duration': None,
            'total_frames_analyzed': None,
            
            # 6 KEY DISC PARAMETERS AT RELEASE
            'disc_speed': None,           # (1) Disc velocity at release (km/h)
            'spin': None,                 # (2) Spin rate at release (RPM)
            'hyzer_angle': None,          # (3) Hyzer angle at release (degrees)
            'launch_angle': None,         # (4) Launch angle at release (degrees)
            'nose_angle': None,           # (5) Nose angle at release (degrees)
            'wobble_amplitude': None,     # (6) Roll angle range during early phase (degrees)
            
            # Early phase disc dynamics
            'acceleration_phase_duration': None,  # Time from start to max velocity
            'acceleration_distance': None,        # Distance traveled during acceleration
            'max_acceleration': None,             # Peak acceleration (mm/s²)
            'average_acceleration': None,         # Average acceleration (mm/s²)
            'spin_rate_evolution': None,          # Spin rate change during early phase (RPM)
            'release_z_position': None,           # Release height (mm)
            'release_tilt_angle': None,           # Release disc tilt (degrees)
            
            # Angular velocity components at release
            'angular_velocity_yaw': None,         # Z-axis rotation (rad/s)
            'angular_velocity_pitch': None,       # Y-axis rotation (rad/s)
            'angular_velocity_roll': None,        # X-axis rotation (rad/s)
        }
        
        # Extract position and rotation data
        positions = body_data['position']
        rotations = body_data['rotation']
        
        # Remove NaN values
        valid_mask = ~np.isnan(positions).any(axis=1)
        if np.sum(valid_mask) < 2:
            return results
        
        positions = positions[valid_mask]
        rotations = rotations[valid_mask]
        
        # EXTRACT EARLY PHASE (first ~1 second or all data if shorter)
        early_phase_frames = min(
            len(positions),
            int(self.EARLY_PHASE_DURATION * self.frame_rate)
        )
        
        positions_early = positions[:early_phase_frames]
        rotations_early = rotations[:early_phase_frames]
        
        results['total_frames_analyzed'] = len(positions_early)
        results['early_phase_duration'] = len(positions_early) / self.frame_rate
        
        # Calculate velocities
        velocities_early = np.diff(positions_early, axis=0) / self.dt
        speeds_early = np.linalg.norm(velocities_early, axis=1)
        
        # Detect release (when disc achieves peak velocity during acceleration)
        release_frame = self._detect_release_frame(speeds_early)
        
        # Calculate 6 key disc parameters at release
        release_params = self._calculate_release_parameters(
            positions_early, rotations_early, velocities_early, release_frame
        )
        results.update(release_params)
        
        # Angular velocities from rotation matrix derivatives
        angular_velocities = self._calculate_angular_velocities_from_rotation(rotations_early)
        results['angular_velocity_yaw'] = angular_velocities['yaw']
        results['angular_velocity_pitch'] = angular_velocities['pitch']
        results['angular_velocity_roll'] = angular_velocities['roll']
        
        # Early phase acceleration analysis
        accel_params = self._analyze_acceleration_phase(positions_early, speeds_early)
        results.update(accel_params)
        
        # Wobble analysis during early phase (RMS and range)
        wobble_res = self._calculate_roll_angle_range(rotations_early)
        if isinstance(wobble_res, tuple) and len(wobble_res) == 2:
            results['wobble_amplitude'] = wobble_res[0]  # RMS
            results['wobble_range'] = wobble_res[1]
        else:
            results['wobble_amplitude'] = wobble_res
        
        # Release position
        results['release_z_position'] = positions_early[0, 2]
        results['release_tilt_angle'] = self._get_tilt_from_rotation(rotations_early[0])
        
        return results
    
    def _detect_release_frame(self, speeds: np.ndarray) -> int:
        """
        Detect release frame using advanced deceleration + zero-crossing method.
        
        Physics: At release:
        1. Hand stops pushing → acceleration drops sharply (high deceleration)
        2. Force ends → acceleration reaches zero (transition point)
        3. Velocity is still high but below peak (aerodynamics take over)
        
        Strategy:
        1. Calculate 5-frame moving average acceleration (smooth signal)
        2. Find highest deceleration (most negative acceleration change)
        3. Find where filtered acceleration crosses zero after that point
        4. Validate: velocity should be ~10% of global maximum at this point
        5. If all conditions met, this is the release point
        
        Args:
            speeds: Speed array (mm/s) of shape (num_frames,)
            
        Returns:
            Frame index of release (default to frame 0 if detection fails)
        """
        try:
            if len(speeds) < 10:
                return 0
            
            # STEP 1: Calculate acceleration with 5-frame moving average
            accelerations = np.diff(speeds)
            
            # 5-frame moving average of acceleration for smooth detection
            window_size = 5
            if len(accelerations) > window_size:
                # Pad the beginning to maintain length alignment
                accel_padded = np.concatenate([np.full(window_size - 1, accelerations[0]), accelerations])
                accel_filtered = np.convolve(accel_padded, np.ones(window_size) / window_size, mode='valid')
                # Trim to match original length
                accel_filtered = accel_filtered[:len(accelerations)]
            else:
                accel_filtered = accelerations
            
            # STEP 2: Find the highest deceleration (most negative acceleration change)
            # Calculate the rate of change of acceleration (jerk)
            if len(accel_filtered) > 5:
                jerk = np.diff(accel_filtered)
                min_jerk_idx = np.argmin(jerk)  # Most negative jerk = highest deceleration
                jerk_window_start = max(0, min_jerk_idx - 2)
                jerk_window_end = min(len(accel_filtered), min_jerk_idx + 3)
            else:
                jerk_window_start = 0
                jerk_window_end = len(accel_filtered)
            
            # STEP 3: In or after the deceleration window, find where acceleration reaches zero
            search_start = jerk_window_end
            search_end = min(len(accel_filtered), search_start + int(self.frame_rate * 0.2))  # Search forward 200ms
            
            zero_crossing_frame = None
            for frame_idx in range(search_start, search_end):
                if frame_idx < len(accel_filtered) - 1:
                    # Find zero crossing: where acceleration changes from positive to negative or vice versa
                    if accel_filtered[frame_idx] * accel_filtered[frame_idx + 1] <= 0:
                        # Zero crossing detected
                        zero_crossing_frame = frame_idx
                        break
                    # Also check if acceleration is very close to zero (within 5% of peak accel)
                    peak_accel = np.max(np.abs(accel_filtered))
                    if peak_accel > 0 and np.abs(accel_filtered[frame_idx]) < peak_accel * 0.05:
                        zero_crossing_frame = frame_idx
                        break
            
            # STEP 4: Validate the candidate release frame
            if zero_crossing_frame is not None and zero_crossing_frame < len(speeds):
                velocity_at_candidate = speeds[zero_crossing_frame]
                max_velocity = np.max(speeds)
                
                # Check if velocity is reasonable (should be high, at or near peak)
                # At release, velocity should be close to peak (within 20% of max)
                if velocity_at_candidate > max_velocity * 0.8:
                    return zero_crossing_frame
            
            # FALLBACK: Find maximum speed if validation fails
            return np.argmax(speeds)
            
        except Exception:
            # If anything fails, use maximum speed
            try:
                return np.argmax(speeds)
            except:
                return 0
    
    def _calculate_release_parameters(
        self, positions: np.ndarray, rotations: np.ndarray, 
        velocities: np.ndarray, release_frame: int = 0
    ) -> Dict[str, Optional[float]]:
        """
        Calculate 6 key disc parameters at the moment of release.
        
        Args:
            positions: Position array (num_frames, 3)
            rotations: Rotation matrix array (num_frames, 3, 3)
            velocities: Velocity array (num_frames-1, 3)
            release_frame: Frame index of release (default 0)
            
        Returns:
            Dictionary with 6 key parameters:
            - disc_speed: Disc velocity at release (km/h)
            - spin: Spin rate around z-axis at release (RPM)
            - hyzer_angle: Hyzer angle at release (degrees)
            - launch_angle: Launch angle at release (degrees)
            - nose_angle: Nose angle at release (degrees)
            - wobble_amplitude: Range of roll angle during early phase (degrees)
        """
        results = {
            'disc_speed': None,
            'spin': None,
            'hyzer_angle': None,
            'launch_angle': None,
            'nose_angle': None,
            'spin_rate_evolution': None,
            'release_frame': None,
        }
        
        try:
            # Use an averaging window of the first N frames after release
            window_size = 10

            # Clamp release frame to valid range (index into velocities)
            release_idx = min(release_frame, max(0, len(velocities) - 1))

            # Define window over velocities: start (inclusive) -> end (exclusive)
            start_idx = release_idx
            end_idx = min(len(velocities), release_idx + window_size)

            if start_idx >= end_idx:
                # Not enough data to form a window; fallback to single-frame at release_idx
                if len(velocities) == 0:
                    return results
                start_idx = release_idx
                end_idx = release_idx + 1

            # Mean velocity vector across the window
            vel_window = velocities[start_idx:end_idx]
            mean_vel = np.mean(vel_window, axis=0)
            results['disc_speed'] = float(np.linalg.norm(mean_vel) * 0.0036)

            # Rotation matrices corresponding to the velocity indices: use rotations[start_idx:end_idx]
            rot_window = rotations[start_idx:end_idx]

            # (2) SPIN: average spin rate (z-axis) across window
            omega_z_list = []
            for i in range(start_idx, end_idx):
                if i > 0 and i < len(rotations):
                    R_dot = (rotations[i] - rotations[i-1]) / self.dt
                    skew = R_dot @ rotations[i].T
                    omega_z = skew[1, 0]
                    omega_z_list.append(abs(omega_z))

            if omega_z_list:
                mean_omega = np.mean(omega_z_list)
                results['spin'] = float(mean_omega * 60.0 / (2.0 * np.pi))

            # (3) HYZER ANGLE: average roll/bank across window
            roll_angles = []
            for R in rot_window:
                roll_rad = np.arctan2(R[2, 1], R[2, 2])
                roll_angles.append(np.degrees(roll_rad))
            if roll_angles:
                results['hyzer_angle'] = float(np.mean(roll_angles))

            # (4) LAUNCH ANGLE: compute from position delta in global X-Z plane
            # Use disc position at release and 10 frames later to form a displacement
            # and compute the angle in the global x-z plane (arctan2(dz, dx)).
            try:
                pos_count = len(positions)
                # Map release index in velocities to position index (velocities[i] is positions[i]->positions[i+1])
                pos_release_idx = min(release_idx, max(0, pos_count - 1))
                # Record the release frame index relative to the positions array
                results['release_frame'] = int(pos_release_idx)
                pos_later_idx = min(pos_release_idx + window_size, pos_count - 1)

                pos_release = positions[pos_release_idx]
                pos_later = positions[pos_later_idx]
                delta_pos = pos_later - pos_release

                launch_rad = np.arctan2(delta_pos[2], delta_pos[0])
                results['launch_angle'] = float(np.degrees(launch_rad))
            except Exception:
                # Fallback to mean velocity method if positions unavailable
                launch_rad = np.arctan2(mean_vel[2], mean_vel[0])
                results['launch_angle'] = float(np.degrees(launch_rad))

            # (5) NOSE ANGLE: angle of attack = angle between velocity and disc plane
            # Disc plane normal is the disc's local Z-axis (third column of R).
            if len(rot_window) > 0:
                disc_normals = np.array([R[:, 2] for R in rot_window])
                mean_normal = np.mean(disc_normals, axis=0)

                # Normalize vectors
                mean_normal_norm = mean_normal / (np.linalg.norm(mean_normal) + 1e-12)
                mean_vel_norm = mean_vel / (np.linalg.norm(mean_vel) + 1e-12)

                # Angle between velocity and disc plane = arcsin( dot(mean_normal, mean_vel) )
                dot_n = np.dot(mean_normal_norm, mean_vel_norm)
                dot_n = np.clip(dot_n, -1.0, 1.0)
                nose_rad = np.arcsin(dot_n)
                results['nose_angle'] = float(np.degrees(nose_rad))

            # Spin evolution: change in spin over the early phase (look-ahead up to window_size)
            if len(rotations) > 1:
                spin_rates = []
                look_ahead_end = min(len(rotations), release_idx + window_size)
                for i in range(1, look_ahead_end):
                    R_dot = (rotations[i] - rotations[i-1]) / self.dt
                    skew = R_dot @ rotations[i].T
                    omega_z = skew[1, 0]
                    spin_rpm = abs(omega_z) * 60.0 / (2.0 * np.pi)
                    spin_rates.append(spin_rpm)

                if len(spin_rates) > 1:
                    results['spin_rate_evolution'] = spin_rates[-1] - spin_rates[0]

            return results

        except Exception:
            # Silent failure: return partial results
            return results
    
    def _analyze_acceleration_phase(
        self, positions: np.ndarray, speeds: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """
        Analyze disc acceleration during run-up phase.
        
        Args:
            positions: Position array (num_frames, 3) in mm
            speeds: Speed array (num_frames-1,) in mm/s
            
        Returns:
            Dictionary with acceleration metrics
        """
        results = {
            'acceleration_phase_duration': None,
            'acceleration_distance': None,
            'max_acceleration': None,
            'average_acceleration': None,
        }
        
        try:
            if len(speeds) < 2:
                return results
            
            # Calculate acceleration
            accelerations = np.diff(speeds) / self.dt
            
            results['max_acceleration'] = float(np.max(np.abs(accelerations)))
            results['average_acceleration'] = float(np.mean(np.abs(accelerations)))
            
            # Find end of acceleration (peak velocity)
            peak_idx = np.argmax(speeds)
            
            # Acceleration phase duration and distance
            if peak_idx > 0:
                results['acceleration_phase_duration'] = peak_idx / self.frame_rate
                
                # Distance during acceleration
                accel_distance = np.linalg.norm(positions[peak_idx] - positions[0])
                results['acceleration_distance'] = float(accel_distance)
            
            return results
        except Exception:
            return results
    
    def _calculate_roll_angle_range(self, rotations: np.ndarray) -> Optional[float]:
        """
        Compute wobble metrics from the sequence of rotation matrices.

        Returns the RMS deviation and the range (max-min) of the instantaneous
        spin-axis angles (degrees) relative to the mean spin axis. This provides
        a robust measure of spin-axis variation (wobble) during the analyzed
        window.

        Args:
            rotations: Rotation matrix array (num_frames, 3, 3)

        Returns:
            Tuple (rms_deviation_deg, range_deg) or None on failure.
        """
        try:
            if len(rotations) < 2:
                return None

            # Compute the spin axis (local Z) for each rotation matrix in global coords
            spin_axes = []
            for R in rotations:
                if np.isnan(R).any():
                    continue
                axis = R[:, 2]
                norm = np.linalg.norm(axis)
                if norm <= 1e-12:
                    continue
                spin_axes.append(axis / norm)

            if not spin_axes:
                return None

            spin_axes = np.array(spin_axes)

            # Mean spin axis
            mean_axis = np.mean(spin_axes, axis=0)
            mean_axis = mean_axis / (np.linalg.norm(mean_axis) + 1e-12)

            # Angles between each instantaneous spin axis and the mean axis
            angles_deg = []
            for a in spin_axes:
                dot = np.clip(np.dot(a, mean_axis), -1.0, 1.0)
                angle_rad = np.arccos(dot)
                angles_deg.append(np.degrees(angle_rad))

            angles_deg = np.array(angles_deg)

            # RMS deviation (degrees)
            rms = float(np.sqrt(np.mean(angles_deg**2)))
            # Range (max - min) as complementary metric
            rng = float(np.max(angles_deg) - np.min(angles_deg))

            # Return RMS (primary) and range as tuple
            return rms, rng

        except Exception:
            return None
    
    def _get_tilt_from_rotation(self, rotation_matrix: np.ndarray) -> Optional[float]:
        """
        Extract disc tilt angle from rotation matrix (pitch angle).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Tilt angle in degrees (pitch)
        """
        try:
            # Pitch is rotation around Y-axis
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            if sy < 1e-6:
                # Singular case
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            else:
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
            
            return np.degrees(pitch)
        except Exception:
            return None
    
    def _calculate_angular_velocities_from_rotation(
        self, rotations: np.ndarray
    ) -> Dict[str, Optional[float]]:
        """
        Calculate angular velocities (yaw, pitch, roll) from rotation matrix derivatives.
        
        Args:
            rotations: Array of shape (num_frames, 3, 3) with rotation matrices
            
        Returns:
            Dictionary with 'yaw', 'pitch', 'roll' in rad/s
        """
        try:
            result = {'yaw': None, 'pitch': None, 'roll': None}
            
            if len(rotations) < 2:
                return result
            
            # Calculate rotation matrix derivatives
            R_derivatives = np.diff(rotations, axis=0) / self.dt
            
            # Angular velocity from skew-symmetric part: [ω]× = R_dot * R^T
            angular_velocities = []
            for i in range(len(R_derivatives)):
                R_curr = rotations[i]
                R_dot = R_derivatives[i]
                
                # Skew-symmetric matrix
                skew = R_dot @ R_curr.T
                
                # Extract angular velocity components from skew matrix
                wx = skew[2, 1]
                wy = skew[0, 2]
                wz = skew[1, 0]
                
                angular_velocities.append([wz, wy, wx])  # yaw, pitch, roll
            
            angular_velocities = np.array(angular_velocities)
            
            result['yaw'] = np.mean(angular_velocities[:, 0]) * 60.0 / (2.0 * np.pi)
            result['pitch'] = np.mean(angular_velocities[:, 1])
            result['roll'] = np.mean(angular_velocities[:, 2])
            
            return result
        except Exception:
            return {'yaw': None, 'pitch': None, 'roll': None}
    
    def compare_throws(self, throws_analysis: Dict[str, Dict]) -> Dict:
        """
        Compare disc parameters across multiple throws during early phase.
        
        Args:
            throws_analysis: Dictionary mapping throw names to analysis results
            
        Returns:
            Comparison statistics for early phase metrics
        """
        if not throws_analysis:
            return {}
        
        comparison = {
            'throw_count': len(throws_analysis),
            'best_disc_speed': None,
            'average_disc_speed': None,
            'best_spin': None,
            'average_spin': None,
            'average_hyzer_angle': None,
            'average_launch_angle': None,
            'average_nose_angle': None,
            'best_acceleration': None,
            'average_acceleration': None,
        }
        
        disc_speeds = [a.get('disc_speed', 0) for a in throws_analysis.values() if a.get('disc_speed')]
        spins = [a.get('spin', 0) for a in throws_analysis.values() if a.get('spin')]
        hyzer_angles = [a.get('hyzer_angle', 0) for a in throws_analysis.values() if a.get('hyzer_angle') is not None]
        launch_angles = [a.get('launch_angle', 0) for a in throws_analysis.values() if a.get('launch_angle') is not None]
        nose_angles = [a.get('nose_angle', 0) for a in throws_analysis.values() if a.get('nose_angle') is not None]
        accelerations = [a.get('max_acceleration', 0) for a in throws_analysis.values() if a.get('max_acceleration')]
        
        if disc_speeds:
            comparison['best_disc_speed'] = max(disc_speeds)  # km/h
            comparison['average_disc_speed'] = np.mean(disc_speeds)  # km/h
        
        if spins:
            comparison['best_spin'] = max(spins)
            comparison['average_spin'] = np.mean(spins)
        
        if hyzer_angles:
            comparison['average_hyzer_angle'] = np.mean(hyzer_angles)
        
        if launch_angles:
            comparison['average_launch_angle'] = np.mean(launch_angles)
        
        if nose_angles:
            comparison['average_nose_angle'] = np.mean(nose_angles)
        
        if accelerations:
            comparison['best_acceleration'] = max(accelerations)
            comparison['average_acceleration'] = np.mean(accelerations)
        
        return comparison
