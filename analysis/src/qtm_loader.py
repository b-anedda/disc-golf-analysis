"""Load and parse exported QTM JSON motion-capture data (JSON-only).

This simplified loader intentionally drops live QTM REST/API support and
focuses on reading exported QTM JSON files. The older QTM API methods have
been disabled to avoid depending on QTM internals or network plugins.
"""

from typing import Dict, List, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


class QTMLoader:
    """JSON-only loader for QTM exported 6DOF rigid-body data.

    Use `load_from_json()` to populate `self.body_data` and then
    `extract_disc_data()` to retrieve the position/rotation/residual arrays.
    """

    def __init__(self):
        self.body_data: Optional[Dict[str, np.ndarray]] = None
        self.frame_rate: Optional[float] = None
        self.sample_count: Optional[int] = None
        self.body_name: str = 'Discgolf_Midrange_Hex'

    # --- Deprecated / disabled QTM API methods ---
    def connect_to_qtm(self) -> bool:
        logger.warning('QTM REST/API support disabled in JSON-only loader')
        return False

    def load_measurement(self, measurement_path: str) -> bool:
        logger.warning('Direct .qtm loading is disabled; use load_from_json()')
        return False

    def load_project(self, project_path: str) -> bool:
        logger.warning('Project loading disabled in JSON-only loader')
        return False

    def get_available_bodies(self) -> List[str]:
        logger.warning('get_available_bodies: QTM API disabled')
        return []

    # Keep extract_rigid_body_6dof but restrict to already-loaded JSON data
    def extract_rigid_body_6dof(self, body_name: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        if body_name is None:
            body_name = self.body_name

        if self.body_data is not None:
            return self.body_data

        logger.error('No JSON data loaded; call load_from_json() first')
        return None
    
    def _extract_6dof_rest_api(self, body_name: str) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract 6DOF data using QTM REST API via /api/scripting/qtm/.
        
        Args:
            body_name: Name of the rigid body (e.g., 'Discgolf_Midrange_Hex')
            
        Returns:
            Dictionary with position, rotation, residual arrays
        """
        try:
            # First, get list of available 6DOF bodies
            response = self.requests.get(f"{self.rest_api_url}/data/series/_6d", timeout=5)
            if response.status_code != 200:
                print(f"Error: Could not get 6DOF bodies list (status {response.status_code})")
                return None
            
            bodies = response.json()
            if not bodies:
                print("Error: No 6DOF bodies available in current measurement")
                print("Make sure:")
                print("  1. A measurement is loaded in QTM")
                print("  2. The measurement has the disc 6DOF rigid body")
                print(f"  3. Body name matches: '{body_name}'")
                return None
            
            # Check if our body exists
            if body_name not in bodies:
                print(f"Error: Body '{body_name}' not found in measurement")
                print(f"Available bodies: {bodies}")
                return None
            
            # Get 6DOF data for our body
            print(f"   Fetching 6DOF body '{body_name}'...")
            response = self.requests.get(
                f"{self.rest_api_url}/data/series/_6d/{body_name}",
                timeout=10
            )
            
            if response.status_code != 200:
                print(f"Error: Could not fetch body data (status {response.status_code})")
                return None
            
            body_data = response.json()
            
            # Parse the response data
            # QTM REST API typically returns data in format:
            # { "data": [ [x, y, z, ...], [...], ... ] }
            # or { "frames": [ {position, rotation, residual}, ...] }
            
            positions = []
            rotations = []
            residuals = []
            
            if isinstance(body_data, dict):
                # Check for common response formats
                frames_data = None
                
                if 'frames' in body_data:
                    frames_data = body_data['frames']
                elif 'Data' in body_data:
                    frames_data = body_data['Data']
                elif 'data' in body_data:
                    frames_data = body_data['data']
                else:
                    # Maybe the whole thing is array-like
                    frames_data = body_data
                
                # If frames_data is a dict of lists (position, rotation arrays)
                if isinstance(frames_data, dict):
                    position_data = frames_data.get('Position') or frames_data.get('position') or []
                    rotation_data = frames_data.get('Rotation') or frames_data.get('rotation') or []
                    residual_data = frames_data.get('Residual') or frames_data.get('residual') or []
                    
                    # Convert position arrays
                    for pos in position_data:
                        positions.append(pos if isinstance(pos, (list, tuple)) else [np.nan, np.nan, np.nan])
                    
                    # Convert rotation data
                    for rot in rotation_data:
                        if isinstance(rot, list):
                            if len(rot) == 9:
                                rotations.append(np.array(rot).reshape((3, 3)))
                            elif len(rot) == 3 and isinstance(rot[0], list):
                                rotations.append(np.array(rot))
                            else:
                                rotations.append(np.eye(3))
                        else:
                            rotations.append(np.eye(3))
                    
                    # Convert residuals
                    residuals = residual_data if residual_data else [np.nan] * len(positions)
                    
                # If frames_data is a list of frame objects
                elif isinstance(frames_data, list):
                    for frame in frames_data:
                        if isinstance(frame, dict):
                            # Each frame has position, rotation, residual
                            pos = frame.get('Position') or frame.get('position') or [np.nan, np.nan, np.nan]
                            positions.append(pos)
                            
                            rot_data = frame.get('Rotation') or frame.get('rotation') or np.eye(3)
                            if isinstance(rot_data, list):
                                if len(rot_data) == 9:
                                    rot = np.array(rot_data).reshape((3, 3))
                                elif len(rot_data) == 3 and isinstance(rot_data[0], list):
                                    rot = np.array(rot_data)
                                else:
                                    rot = np.eye(3)
                            else:
                                rot = np.eye(3)
                            rotations.append(rot)
                            
                            res = frame.get('Residual') or frame.get('residual') or np.nan
                            residuals.append(res)
                        else:
                            positions.append([np.nan, np.nan, np.nan])
                            rotations.append(np.eye(3))
                            residuals.append(np.nan)
            
            if not positions:
                print("Error: Could not parse frame data from response")
                return None
            
            self.body_data = {
                'position': np.array(positions),
                'rotation': np.array(rotations),
                'residual': np.array(residuals)
            }
            
            print(f"   [OK] Extracted {len(positions)} frames")
            return self.body_data
            
        except Exception as e:
            print(f"Error extracting 6DOF via REST API: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def extract_disc_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract disc rigid body 6DOF data.
        
        Returns:
            Dictionary with position, rotation, and residual arrays
        """
        return self.extract_rigid_body_6dof()
    
    def rotation_matrix_to_euler(self, rotation_matrix: np.ndarray) -> np.ndarray:
        """
        Convert 3x3 rotation matrix to Euler angles (yaw, pitch, roll in degrees).
        
        Args:
            rotation_matrix: 3x3 rotation matrix
            
        Returns:
            Array of [yaw, pitch, roll] in degrees
        """
        try:
            # Extract angles from rotation matrix
            # Assuming Z-Y-X (yaw-pitch-roll) convention
            sy = np.sqrt(rotation_matrix[0, 0]**2 + rotation_matrix[1, 0]**2)
            
            singular = sy < 1e-6
            
            if not singular:
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
            else:
                yaw = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
                pitch = np.arctan2(-rotation_matrix[2, 0], sy)
                roll = 0
            
            return np.array([np.degrees(yaw), np.degrees(pitch), np.degrees(roll)])
        except Exception as e:
            logger.exception("Error converting rotation matrix: %s", e)
            return np.array([np.nan, np.nan, np.nan])
    
    def get_frame_rate(self) -> Optional[float]:
        """Get frame rate of the measurement in Hz."""
        try:
            # QTM API not supported in JSON-only loader
            logger.warning('get_frame_rate: QTM API not available in JSON-only loader')
            return None
        except Exception:
            return None
    
    def calculate_velocity(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate velocity from position trajectory.
        
        Args:
            positions: Array of shape (num_frames, 3) with XYZ positions
            
        Returns:
            Array of shape (num_frames-1, 3) with velocity vectors
        """
        if self.frame_rate is None:
            # If frame_rate wasn't set by JSON, default to 240 Hz common for high-speed
            logger.info('Frame rate unknown, defaulting to 240 Hz')
            self.frame_rate = 240.0
        
        # Filter out NaN values for derivative calculation
        dt = 1.0 / self.frame_rate
        velocity = np.diff(positions, axis=0) / dt
        return velocity
    
    def calculate_acceleration(self, positions: np.ndarray) -> np.ndarray:
        """
        Calculate acceleration from position trajectory.
        
        Args:
            positions: Array of shape (num_frames, 3) with XYZ positions
            
        Returns:
            Array of shape (num_frames-2, 3) with acceleration vectors
        """
        velocity = self.calculate_velocity(positions)
        acceleration = self.calculate_velocity(velocity)
        return acceleration
    
    def load_from_json(self, json_file: str, body_name: Optional[str] = None) -> bool:
        """
        Load 6DOF data from QTM JSON export file.
        
        Args:
            json_file: Path to exported .json file from QTM
            body_name: Name of rigid body to load (default: Discgolf_Midrange_Hex)
            
        Returns:
            bool: True if successful
        """
        import json
        from pathlib import Path
        
        if body_name is None:
            body_name = self.body_name

        json_path = Path(json_file)
        if not json_path.exists():
            logger.error('JSON file not found: %s', json_path)
            return False

        try:
            logger.info('Loading from JSON: %s', json_path.name)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Extract frame rate
            if 'Timebase' in data and 'Frequency' in data['Timebase']:
                self.frame_rate = float(data['Timebase']['Frequency'])
                logger.info('Frame rate: %s Hz', self.frame_rate)
            
            # Find rigid body
            rigid_bodies = data.get('RigidBodies', [])
            body = None
            for rb in rigid_bodies:
                if rb.get('Name') == body_name:
                    body = rb
                    break
            
            if body is None:
                logger.error("Body '%s' not found in JSON", body_name)
                available = [rb.get('Name', 'Unknown') for rb in rigid_bodies]
                logger.info('Available bodies: %s', available)
                return False
            
            # Extract position, rotation, residual from all parts
            positions = []
            rotations = []
            residuals = []
            
            parts = body.get('Parts', [])
            for part in parts:
                values = part.get('Values', [])
                for frame_data in values:
                    # Frame data format: [[x, y, z, residual], [rotation_matrix_9_elements]]
                    pos_data = frame_data[0] if len(frame_data) > 0 else [np.nan, np.nan, np.nan, np.nan]
                    
                    # Extract just XYZ (first 3 elements), ignore residual in position array
                    pos = pos_data[:3] if isinstance(pos_data, (list, tuple)) else [np.nan, np.nan, np.nan]
                    
                    # Rotation is second element (9-element list that needs reshaping)
                    if len(frame_data) > 1:
                        rot_data = frame_data[1]
                        if isinstance(rot_data, list) and len(rot_data) == 9:
                            rot = np.array(rot_data).reshape((3, 3))
                        else:
                            rot = np.eye(3)
                    else:
                        rot = np.eye(3)
                    
                    # Residual from position data (4th element if available)
                    if isinstance(pos_data, (list, tuple)) and len(pos_data) > 3:
                        residual = pos_data[3]
                    else:
                        residual = np.nan
                    
                    positions.append(pos)
                    rotations.append(rot)
                    residuals.append(residual)
            
            if not positions:
                logger.error("No frame data found for body '%s'", body_name)
                return False
            
            self.body_data = {
                'position': np.array(positions),
                'rotation': np.array(rotations),
                'residual': np.array(residuals)
            }
            self.sample_count = len(positions)

            # Extract marker trajectories (for more robust velocity calculations)
            markers = self._extract_markers_from_json(data)
            if markers:
                self.body_data['markers'] = markers

            logger.info('Loaded %d frames', self.sample_count)
            logger.debug('Position shape: %s', self.body_data['position'].shape)
            logger.debug('Rotation shape: %s', self.body_data['rotation'].shape)
            return True
            
        except Exception as e:
            logger.exception('Error loading JSON: %s', e)
            return False
    
    def _extract_markers_from_json(self, data: Dict) -> Optional[Dict[str, np.ndarray]]:
        """Extract individual marker trajectories from JSON data.
        
        Args:
            data: Parsed JSON dictionary
            
        Returns:
            Dictionary with marker names as keys and (N,3) position arrays as values
        """
        try:
            markers_data = data.get('Markers', [])
            if not markers_data:
                return None
            
            markers = {}
            for marker in markers_data:
                marker_name = marker.get('Name')
                if not marker_name:
                    continue
                
                # Extract position values from all parts
                positions = []
                parts = marker.get('Parts', [])
                for part in parts:
                    values = part.get('Values', [])
                    for frame_data in values:
                        # Frame data format: [x, y, z, residual]
                        if isinstance(frame_data, (list, tuple)) and len(frame_data) >= 3:
                            pos = frame_data[:3]  # Just XYZ
                            positions.append(pos)
                        else:
                            positions.append([np.nan, np.nan, np.nan])
                
                if positions:
                    markers[marker_name] = np.array(positions)
                    logger.debug('Extracted marker "%s": %d frames', marker_name, len(positions))
            
            return markers if markers else None
            
        except Exception as e:
            logger.warning('Failed to extract markers: %s', e)
            return None
