# Disc Golf Flight Analysis

Python-based analysis tool for disc flight dynamics using motion capture data from Qualisys Track Manager (QTM).

## Features

- **Disc Dynamics Analysis**: Spin rate, stability, wobble characteristics
- **Flight Parameters**: Velocity, trajectory, glide ratio, flight angle
- **Comprehensive Visualization**: Stability metrics, individual marker tracking, 3D flight paths
- **Real-time Analysis**: Fast computation of flight biomechanics
- **Flexible Data Source**: Works with QTM captures or synthetic demo data

## Project Structure

```
analysis/
├── src/
│   ├── app.py                 # PyQt6 GUI application
│   ├── qtm_loader.py          # QTM API data interface
│   ├── throw_analysis.py      # Disc flight analysis engine
│   └── visualization/
│       └── visualizer.py      # Matplotlib visualizations
├── examples/
│   └── basic_analysis.py      # Runnable example workflow
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Ensure **Qualisys Track Manager** is installed with scripting support

## Usage

### GUI Application

```bash
cd analysis
python -m src.app
```

**Workflow:**
1. Click "Load QTM Project" to open your project
2. Select a disc flight from the dropdown
3. Click "Load Throw Data"
4. Click "Analyze" to compute dynamics and create visualizations

**Tabs available:**
- **3D Flight Path**: Full trajectory of all four disc markers
- **Stability Metrics**: Four-panel analysis of wobble, separation, height, tilt
- **Individual Markers**: Top-down view of each marker's separate path
- **Analysis Summary**: Comprehensive 6-panel dashboard

### Python Script

```python
from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer
from src.visualization.visualizer import ThrowVisualizer

# Load data
loader = QTMLoader()
loader.connect_to_qtm()
loader.load_project("path/to/Settings.qtmproj")
disc_data = loader.extract_disc_data()

# Analyze disc flight dynamics
analyzer = DiscAnalyzer(frame_rate=240.0)
results = analyzer.analyze_disc_trajectory(disc_data)

# Access comprehensive metrics
print(f"Spin Rate: {results['spin_rate']:.0f} RPM")
print(f"Stability Index: {results['stability_index']:.2f}")
print(f"Wobble Frequency: {results['wobble_frequency']:.1f} Hz")
print(f"Flight Distance: {results['flight_distance']:.0f} mm")

# Visualize
visualizer = ThrowVisualizer()
fig = visualizer.plot_disc_stability_metrics(disc_data)
```

### Command Line Example

```bash
python -m examples.basic_analysis
```

Generates comprehensive flight analysis with synthetic demo data.

## Disc Markers

The project tracks disc position at four points for comprehensive dynamics analysis:

- **Wide**: Outer edge of disc (rim side)
  - Used for spin rate calculation
  - Key for wobble detection
  - Indicator of disc rotation around vertical axis

- **Mid**: Mid-band of disc
  - Stability and marker separation tracking
  - Validates center-based analysis

- **Center**: Center point of disc
  - Primary trajectory for flight analysis
  - Used for overall flight parameters

- **Close**: Inner edge of disc
  - Secondary reference for tilt estimation
  - Completes the four-point tracking system

## Analysis Outputs

### Flight Metrics
- **Release velocity** (mm/s) - Speed at moment of release
- **Peak velocity** (mm/s) - Maximum speed during flight
- **Average velocity** (mm/s) - Mean velocity over entire flight
- **Flight distance** (mm) - Total horizontal distance traveled
- **Max height** (mm) - Maximum vertical elevation
- **Flight angle** (degrees) - Launch angle relative to horizontal
- **Flight duration** (seconds) - Total time in air
- **Glide ratio** - Distance-to-height ratio (aerodynamic efficiency measure)
- **Release height** (mm) - Z-position at release
- **Landing position** (X, Y mm) - Final impact location

### Disc Dynamics & Stability
- **Spin rate** (RPM) - Disc rotation speed around vertical axis
  - Calculated from Wide marker separation changes
  - Indicates throwing power and technique
  
- **Angular velocities** (rad/s or deg/s)
  - Yaw: Rotation around Z axis (spin)
  - Pitch: Rotation around Y axis (nose angle)
  - Roll: Rotation around X axis (banking)
  
- **Stability index** (unitless)
  - Ratio of disc radius to wobble amplitude
  - Higher = more stable, lower = more oscillating
  - Good flights typically > 1.0
  
- **Wobble frequency** (Hz)
  - How often the disc oscillates during flight
  - Typical range: 0.5-3 Hz for disc golf
  
- **Wobble amplitude** (mm)
  - Magnitude of radius variation during flight

### 6 Key Release Parameters

These 6 physics-based parameters are calculated at the **exact moment of disc release** (frame 0) and characterize the throw:

1. **Disc Speed** (mm/s): Velocity magnitude at release
   - Typical range: 6,000-10,000 mm/s
   - Higher = more forceful throw
   
2. **Spin Rate** (RPM): Rotational speed around disc z-axis
   - Typical range: 1,500-3,000 RPM  
   - Higher spin = more stable flight
   
3. **Hyzer Angle** (degrees): Tilt of disc in z-y plane
  - Computed from disc normal projected into the global Y-Z plane
  - 0° = flat/level, positive = hyzer, negative = anhyzer
   
4. **Launch Angle** (degrees): Pitch angle of velocity
   - Range: 0° to 45° for typical throws
   - Controls flight duration and max height
   
5. **Nose Angle** (degrees): Disc nose-up/down relative to flight direction
  - Computed from disc normal and full 3D velocity, projected into the flight vertical plane
  - Positive = nose up, negative = nose down
   
6. **Wobble Amplitude** (degrees): Roll angle oscillation amplitude at release
   - Calculated as (max - min) roll angle over 10-frame release window ÷ 2
   - Typical range: 2-10° for stable throws
   - Indicator of release quality and disc stability

**See [SIX_PARAMETERS.md](../../SIX_PARAMETERS.md) for detailed physics, formulas, and interpretation.**
  - Smaller amplitude = more stable flight
  
- **Release tilt angle** (degrees)
  - Disc angle at moment of release
  - Affects initial flight trajectory

## Visualization Tools

### 3D Flight Path
- Full 3D trajectory for all four disc markers
- Color-coded by marker position (Wide=red, Mid=orange, Center=green, Close=blue)
- Start/end points marked for easy orientation
- Interactive rotation and zoom

### Stability Metrics (4-panel analysis)
- **Disc Radius Over Flight**: Wobble visualization showing radius changes
- **Marker Separation from Center**: Individual distance tracking for Wide, Mid, Close
- **Height and Distance Progression**: Dual-axis plot of vertical and horizontal motion
- **Estimated Tilt Angle Profile**: Disc orientation throughout flight

### Individual Marker Trajectories
- Separate top-down (XY) views for each of four markers
- Start (circle) and end (X) point markers
- Detailed position analysis enabling comparison

### Comprehensive Summary Dashboard (6 panels)
1. **Flight Parameters**: Bar chart of distance, height, duration
2. **Velocity Profile**: Release/peak/average velocity comparison
3. **Disc Dynamics**: Spin rate, stability index, wobble metrics
4. **Flight Characteristics**: Angle, glide ratio, release height
5. **Angular Motion**: Yaw, pitch, roll, tilt data
6. **Efficiency Metrics**: Distance/velocity ratios, stability assessment

## Frame Rate

Default frame rate: **240 Hz** (typical high-speed capture)

Configure in code:
```python
analyzer = DiscAnalyzer(frame_rate=240.0)  # Change if different
```

All velocity and acceleration calculations automatically adjust to frame rate.

## Key Concepts

### Stability Index
Higher is better. Indicates how much the disc wobbles relative to its size:
- `> 2.0`: Excellent stability
- `1.0-2.0`: Good stability
- `< 1.0`: Variable or unstable flight

### Glide Ratio
Distance ÷ Height. Measures aerodynamic efficiency:
- Higher glide ratio = more distance per unit height
- Good disc golf throws: 4-8 ratio
- Exceptional throws: > 8 ratio

### Angular Velocities
From four marker positions, estimates disc orientation changes:
- **Yaw** (spin): Most important for disc golf stability
- **Pitch** (nose): Controls lift and drop
- **Roll** (banking): Affects lateral movement

## Development

### Adding New Analysis Metrics

Extend `DiscAnalyzer` in `src/throw_analysis.py`:

```python
def analyze_new_metric(self, disc_positions: Dict[str, np.ndarray]) -> float:
    """Calculate a new metric."""
    # Your analysis code
    return metric_value
```

### Custom Visualizations

Add methods to `ThrowVisualizer` in `src/visualization/visualizer.py`:

```python
@staticmethod
def plot_custom_metric(disc_positions: Dict[str, np.ndarray]) -> Figure:
    """Create custom visualization."""
    fig, ax = plt.subplots()
    # Your plotting code
    return fig
```

## Testing

Run the example script to validate setup without QTM:

```bash
python -m examples.basic_analysis
```

This generates synthetic flight data and creates all visualizations.

## Troubleshooting

### "QTM API not available"
- Ensure Qualisys Track Manager is installed
- Python environment needs QTM scripting support
- App will fall back to demo mode with synthetic data

### No trajectories found
- Ensure .qtmproj contains processed 3D data
- Verify marker names match `DiscGolf_Labels.xml`
- Check calibration files exist for capture date

### Import errors
Run: `pip install -r requirements.txt`

## Dependencies

- **numpy** (1.24+) - Numerical computation
- **scipy** (1.10+) - Signal processing
- **matplotlib** (3.7+) - Visualization
- **PyQt6** (6.5+) - GUI framework
- **Qualisys QTM** (optional) - Motion capture system

## Notes

- All coordinates in **millimeters (mm)** and **seconds (s)**
- Frame rate-aware calculations throughout
- NaN values handled for tracking gaps
- QTM connection optional—works standalone with demo data
- **Velocity and trajectory calculations**: Uses **Mid marker** position from marker trajectories for robust velocity and position analysis
  - More reliable than 6DOF rigid body center of mass (which is calculated from all markers)
  - Direct measurement from single marker reduces noise and filtering artifacts
  - Falls back to 6DOF body position if marker data unavailable
- **Orientation calculations**: Uses 6DOF rotation matrices for all angle measurements (hyzer, nose, wobble)
  - Provides accurate disc orientation throughout flight
  - Essential for spin-invariant angle calculations
- **Release point detection**: Automatically identifies disc release using velocity analysis
  - Finds first velocity peak within valid range (3-40 m/s)
  - Verifies acceleration phase precedes release
  - Ensures accurate parameter measurement at true release moment
- **Net impact detection**: Automatically detects and excludes post-impact data when disc hits net
  - Identifies first frame after release where velocity drops below 50% of release velocity
  - Only analyzes flight data up to impact point
  - Ensures clean release parameters without net collision artifacts

## Performance

- Single flight analysis: < 1 second
- Visualization generation: 1-3 seconds depending on quality
- Handles 240 Hz 10-second captures efficiently
- GUI remains responsive during computation

## License

Internal project - DiscGolf Flight Analysis
