"""Load and parse QTM motion capture data for disc golf analysis."""

from typing import Dict, List, Tuple, Optional
import numpy as np


class QTMLoader:
    """Interface to load and extract 6DOF rigid body data from QTM scripting API."""
    
    def __init__(self):
        """Initialize QTM loader."""
        self.body_data: Optional[Dict[str, np.ndarray]] = None
        self.frame_rate: Optional[float] = None
        self.sample_count: Optional[int] = None
        self.body_name: str = 'Discgolf_Midrange_Hex'  # 6DOF rigid body name
        
    def connect_to_qtm(self) -> bool:
        """
        Connect to QTM application via REST API.
        
        This uses QTM's scripting REST API to fetch data without needing the Python console.
        Make sure QTM is running and has a measurement loaded.
        
        Returns:
            bool: True if connection successful, False otherwise.
        """
        try:
            import requests
        except ImportError:
            print("Error: requests library not installed")
            print("Install with: pip install requests")
            return False
        
        # QTM REST API is on port 7979 via /api/scripting/qtm/...
        base_url = "http://localhost:7979/api/scripting/qtm"
        
        try:
            # Test connection to data/series/_6d endpoint (will be empty if no measurement loaded)
            response = requests.get(f"{base_url}/data/series/_6d", timeout=2)
            if response.status_code == 200:
                self.rest_api_url = base_url
                self.requests = requests
                print(f"   [OK] Connected to QTM REST API at {base_url}")
                return True
        except Exception as e:
            pass
        
        print("Error: Could not connect to QTM REST API")
        print("Make sure:")
        print("  1. QTM application is running")
        print("  2. A measurement is loaded in QTM (File > Load > select .qtm file)")
        print("  3. REST API is enabled (Tools > Options > REST API)")
        print("  4. The disc 6DOF body exists in your AIM model")
        return False
    
    def load_measurement(self, measurement_path: str) -> bool:
        """
        Load a QTM measurement file (.qtm).
        
        Note: Direct loading of .qtm files requires access to QTM's scripting API,
        which is only available when running through QTM's Python interface or 
        REST API.
        
        Args:
            measurement_path: Path to .qtm measurement file
            
        Returns:
            bool: True if successful
        """
        print(f"\nAttempting to load: {measurement_path}")
        print("\nNOTE: To load .qtm files, use one of these methods:")
        print()
        print("METHOD 1: Use QTM's Python Scripting Console (Recommended)")
        print("  1. Open QTM application")
        print("  2. Tools → Python Console")
        print("  3. Paste and run this code:")
        print(f"""
from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer

loader = QTMLoader()
loader.load_measurement("{measurement_path}")
body_data = loader.extract_disc_data()

analyzer = DiscAnalyzer(frame_rate=240.0)
results = analyzer.analyze_disc_trajectory(body_data)
print("Disc Speed:", results['disc_speed'])
print("Spin Rate:", results['spin'])
# ... etc
""")
        print()
        print("METHOD 2: Export from QTM and Load Exported Data")
        print("  1. In QTM: Open your measurement")
        print("  2. File → Export → Rigid Body 6DOF (TSV format)")
        print("  3. Run: loader.load_from_exported_file('exported_data.tsv')")
        print()
        
        return False
    
    def load_project(self, project_path: str) -> bool:
        """
        Load a QTM project file.
        
        Args:
            project_path: Path to .qtmproj file
            
        Returns:
            bool: True if successful
        """
        try:
            import qtm.file as qtm_file
            qtm_file.open(project_path)
            return True
        except Exception as e:
            print(f"Error loading project: {e}")
            return False
    
    def get_available_bodies(self) -> List[str]:
        """
        Get list of all available rigid bodies in current measurement.
        
        Returns:
            List of rigid body names
        """
        try:
            from qtm.data.object import body
            bodies = body.get_all_bodies()
            return [b['name'] for b in bodies] if bodies else []
        except Exception as e:
            print(f"Error getting bodies: {e}")
            return []
    
    def extract_rigid_body_6dof(self, body_name: Optional[str] = None) -> Optional[Dict[str, np.ndarray]]:
        """
        Extract 6DOF rigid body data (position + rotation matrix).
        
        Works with:
        1. Pre-loaded JSON data (self.body_data)
        2. QTM REST API
        3. Direct QTM API (if available)
        
        Args:
            body_name: Name of the rigid body. Defaults to 'Discgolf_Midrange_Hex'
            
        Returns:
            Dictionary with keys:
            - 'position': array of shape (num_frames, 3) - XYZ positions in mm
            - 'rotation': array of shape (num_frames, 3, 3) - rotation matrices
            - 'residual': array of shape (num_frames,) - model fit residuals
        """
        if body_name is None:
            body_name = self.body_name
        
        # If data already loaded from JSON, return it
        if self.body_data is not None:
            return self.body_data
        
        # Try REST API
        if hasattr(self, 'rest_api_url') and hasattr(self, 'requests'):
            return self._extract_6dof_rest_api(body_name)
        
        # Fall back to direct API
        try:
            from qtm.data.object import body as qtm_body
            from qtm.data.series import rigid_body
            
            # Find body by name
            body_id = qtm_body.find_body(body_name)
            if body_id is None:
                print(f"Body '{body_name}' not found.")
                return None
            
            # Get sample count
            num_samples = rigid_body.get_sample_count(body_id)
            self.sample_count = num_samples
            
            # Extract all 6DOF samples
            positions = []
            rotations = []
            residuals = []
            
            for frame in range(num_samples):
                sample = rigid_body.get_sample(body_id, frame)
                if sample:
                    pos = sample.get('position', [np.nan, np.nan, np.nan])
                    positions.append(pos)
                    
                    rot = sample.get('rotation', np.eye(3))
                    rotations.append(rot)
                    
                    res = sample.get('residual', np.nan)
                    residuals.append(res)
                else:
                    positions.append([np.nan, np.nan, np.nan])
                    rotations.append(np.full((3, 3), np.nan))
                    residuals.append(np.nan)
            
            self.body_data = {
                'position': np.array(positions),
                'rotation': np.array(rotations),
                'residual': np.array(residuals)
            }
            return self.body_data
            
        except Exception as e:
            print(f"Error extracting rigid body '{body_name}': {e}")
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
            print(f"Error converting rotation matrix: {e}")
            return np.array([np.nan, np.nan, np.nan])
    
    def get_frame_rate(self) -> Optional[float]:
        """Get frame rate of the measurement in Hz."""
        try:
            import qtm
            fr = qtm.settings.processing.get_frequency()
            self.frame_rate = float(fr)
            return self.frame_rate
        except Exception as e:
            print(f"Error getting frame rate: {e}")
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
            self.get_frame_rate()
        
        if self.frame_rate is None:
            print("Warning: Frame rate unknown, using 1.0 Hz")
            self.frame_rate = 1.0
        
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
            print(f"✗ JSON file not found: {json_path}")
            return False
        
        try:
            print(f"Loading from JSON: {json_path.name}")
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            # Extract frame rate
            if 'Timebase' in data and 'Frequency' in data['Timebase']:
                self.frame_rate = float(data['Timebase']['Frequency'])
                print(f"  Frame rate: {self.frame_rate} Hz")
            
            # Find rigid body
            rigid_bodies = data.get('RigidBodies', [])
            body = None
            for rb in rigid_bodies:
                if rb.get('Name') == body_name:
                    body = rb
                    break
            
            if body is None:
                print(f"✗ Body '{body_name}' not found in JSON")
                available = [rb.get('Name', 'Unknown') for rb in rigid_bodies]
                print(f"  Available bodies: {available}")
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
                print(f"✗ No frame data found for body '{body_name}'")
                return False
            
            self.body_data = {
                'position': np.array(positions),
                'rotation': np.array(rotations),
                'residual': np.array(residuals)
            }
            self.sample_count = len(positions)
            
            print(f"  ✓ Loaded {self.sample_count} frames")
            print(f"  Position shape: {self.body_data['position'].shape}")
            print(f"  Rotation shape: {self.body_data['rotation'].shape}")
            return True
            
        except Exception as e:
            print(f"✗ Error loading JSON: {e}")
            import traceback
            traceback.print_exc()
            return False
