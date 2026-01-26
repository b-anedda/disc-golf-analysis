"""PyQt6-based GUI application for disc golf throw analysis."""

import sys
from typing import Optional, Dict
import numpy as np
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QComboBox, QLabel, QProgressBar, QTabWidget,
    QStatusBar, QFileDialog, QMessageBox
)
from PyQt6.QtCore import Qt, QTimer

import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer
from src.visualization.visualizer import ThrowVisualizer


class DiscGolfAnalysisApp(QMainWindow):
    """Main application window for disc flight analysis."""
    
    def __init__(self):
        """Initialize application."""
        super().__init__()
        
        self.qtm_loader = QTMLoader()
        self.disc_analyzer: Optional[DiscAnalyzer] = None
        self.body_data = None  # 6DOF rigid body data
        self.current_throw_name = "Unknown"
        
        self.init_ui()
        
    def init_ui(self):
        """Initialize user interface."""
        self.setWindowTitle("Disc Golf Throw Analysis")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central = QWidget()
        self.setCentralWidget(central)
        
        # Main layout
        layout = QVBoxLayout()
        
        # Control panel
        control_layout = QHBoxLayout()
        
        load_btn = QPushButton("Load QTM Project")
        load_btn.clicked.connect(self.load_project)
        control_layout.addWidget(load_btn)
        
        load_data_btn = QPushButton("Load Throw Data")
        load_data_btn.clicked.connect(self.load_throw_data)
        control_layout.addWidget(load_data_btn)
        
        control_layout.addWidget(QLabel("Select Throw:"))
        self.throw_combo = QComboBox()
        self.throw_combo.addItems(["Throw1", "throw2", "Static_Disc"])
        control_layout.addWidget(self.throw_combo)
        
        analyze_btn = QPushButton("Analyze")
        analyze_btn.clicked.connect(self.analyze_throw)
        control_layout.addWidget(analyze_btn)
        
        layout.addLayout(control_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Tab widget for visualizations
        self.tabs = QTabWidget()
        
        self.trajectory_3d = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        self.tabs.addTab(self.trajectory_3d, "3D Flight Path")
        
        self.stability_metrics = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        self.tabs.addTab(self.stability_metrics, "Stability Metrics")
        
        self.marker_trajectories = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        self.tabs.addTab(self.marker_trajectories, "Individual Markers")
        
        self.summary_plot = FigureCanvas(Figure(figsize=(5, 4), dpi=100))
        self.tabs.addTab(self.summary_plot, "Analysis Summary")
        
        layout.addWidget(self.tabs)
        
        central.setLayout(layout)
        
        # Status bar
        self.statusBar().showMessage("Ready")
    
    def load_project(self):
        """Load QTM project file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open QTM Project", 
            "c:\\Users\\Contemplas\\Documents\\DiscGolf",
            "QTM Projects (*.qtmproj)"
        )
        
        if not file_path:
            return
        
        if self.qtm_loader.connect_to_qtm():
            if self.qtm_loader.load_project(file_path):
                self.statusBar().showMessage(f"Loaded: {file_path}")
                QMessageBox.information(self, "Success", "QTM project loaded successfully!")
            else:
                QMessageBox.warning(self, "Error", "Failed to load project")
        else:
            QMessageBox.warning(self, "Error", 
                              "QTM scripting API not available.\n" +
                              "Ensure QTM is installed and running.")
    
    def load_throw_data(self):
        """Load 6DOF rigid body data for selected throw."""
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        throw_name = self.throw_combo.currentText()
        self.current_throw_name = throw_name
        
        try:
            # Try to load from QTM if available
            if hasattr(self.qtm_loader, 'connect_to_qtm') and self.qtm_loader.connect_to_qtm():
                self.body_data = self.qtm_loader.extract_disc_data()
            else:
                # Generate synthetic 6DOF data for demonstration
                self.generate_synthetic_body_data(throw_name)
            
            if self.body_data is not None and 'position' in self.body_data:
                self.progress.setValue(100)
                self.statusBar().showMessage(f"Loaded throw data: {throw_name}")
                QMessageBox.information(self, "Success", 
                                      f"Data loaded for {throw_name}")
            else:
                raise Exception("No 6DOF body data found")
                
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to load data: {e}")
        finally:
            self.progress.setVisible(False)
    
    def generate_synthetic_body_data(self, throw_name: str):
        """Generate synthetic 6DOF rigid body data for demonstration."""
        num_frames = 150  # ~0.625 seconds at 240 Hz
        t = np.linspace(0, 0.625, num_frames)
        
        # Parabolic flight path
        x = 6000 * t  # Forward distance  
        y = 400 * np.sin(2 * np.pi * t / 0.625)  # Wobble
        z = 1500 + 4500 * t - 3600 * t**2  # Vertical arc
        
        positions = np.column_stack([x, y, z])
        
        # Generate rotation matrices (primarily spinning with slight pitch/roll)
        rotations = []
        for i, frame_t in enumerate(t):
            # Yaw (spin) - 200 RPM
            yaw = 2 * np.pi * 200 * frame_t
            # Pitch (wobble)
            pitch = 0.1 * np.sin(2 * np.pi * frame_t / 0.625)
            # Roll (banking)
            roll = 0.05 * np.cos(2 * np.pi * frame_t / 0.625)
            
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
        residuals = np.random.normal(0.5, 0.1, num_frames)
        residuals = np.clip(residuals, 0, 2)
        
        self.body_data = {
            'position': positions,
            'rotation': rotations,
            'residual': residuals
        }
    
    def analyze_throw(self):
        """Analyze loaded 6DOF rigid body data."""
        if self.body_data is None:
            QMessageBox.warning(self, "Error", "No 6DOF body data loaded")
            return
        
        self.progress.setVisible(True)
        self.progress.setValue(0)
        
        try:
            # Initialize analyzer
            self.disc_analyzer = DiscAnalyzer(frame_rate=240.0)
            
            # Analyze disc trajectory from 6DOF data
            disc_analysis = self.disc_analyzer.analyze_disc_trajectory(self.body_data)
            self.progress.setValue(50)
            
            # Create visualizations
            self.update_visualizations(disc_analysis)
            self.progress.setValue(100)
            
            # Display results
            summary_text = self._format_analysis_summary(disc_analysis)
            QMessageBox.information(self, "Analysis Complete", summary_text)
            
            self.statusBar().showMessage(f"Analysis complete: {self.current_throw_name}")
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Analysis failed: {e}")
        finally:
            self.progress.setVisible(False)
    
    def update_visualizations(self, disc_analysis):
        """Update all visualization tabs with 6DOF body data."""
        visualizer = ThrowVisualizer()
        
        # Extract position for visualization
        positions = self.body_data['position']
        rotations = self.body_data['rotation']
        
        # 3D Trajectory - just body center of mass path
        fig = self._plot_body_trajectory_3d(positions, self.current_throw_name)
        self.trajectory_3d.figure = fig
        self.trajectory_3d.draw()
        
        # Stability Metrics
        fig = self._plot_stability_metrics(positions)
        self.stability_metrics.figure = fig
        self.stability_metrics.draw()
        
        # Angular velocities and orientation
        fig = self._plot_rotation_and_orientation(rotations, self.current_throw_name)
        self.marker_trajectories.figure = fig  # Reuse this canvas
        self.marker_trajectories.draw()
        
        # Summary
        fig = self._plot_analysis_summary(disc_analysis, self.current_throw_name)
        self.summary_plot.figure = fig
        self.summary_plot.draw()
    
    @staticmethod
    def _plot_body_trajectory_3d(positions: np.ndarray, title: str) -> Figure:
        """Plot 3D body trajectory."""
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection='3d')
        
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_pos = positions[valid_mask]
        
        ax.plot(valid_pos[:, 0], valid_pos[:, 1], valid_pos[:, 2], 
               'b-', linewidth=2, label='Flight Path')
        ax.scatter(*valid_pos[0], s=100, c='green', marker='o', label='Release')
        ax.scatter(*valid_pos[-1], s=100, c='red', marker='x', label='Landing')
        
        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_zlabel('Z (mm)')
        ax.set_title(f"{title} - 3D Flight Path")
        ax.legend()
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _plot_stability_metrics(positions: np.ndarray) -> Figure:
        """Plot stability metrics over time."""
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_pos = positions[valid_mask]
        
        t = np.arange(len(valid_pos)) / 240.0
        
        # XY deviation from average
        xy_avg = np.mean(valid_pos[:, :2], axis=0)
        xy_dist = np.linalg.norm(valid_pos[:, :2] - xy_avg, axis=1)
        
        ax.plot(t, xy_dist, linewidth=2)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('XY Deviation (mm)')
        ax.set_title('Flight Stability (Path Deviation)')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _plot_rotation_and_orientation(rotations: np.ndarray, title: str) -> Figure:
        """Plot rotation/orientation over time."""
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)
        
        t = np.arange(len(rotations)) / 240.0
        
        # Extract Euler angles from rotation matrices
        yaws = []
        pitches = []
        rolls = []
        
        for R in rotations:
            sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            pitch = np.arctan2(-R[2, 0], sy)
            roll = np.arctan2(R[2, 1], R[2, 2])
            
            yaws.append(np.degrees(yaw))
            pitches.append(np.degrees(pitch))
            rolls.append(np.degrees(roll))
        
        ax.plot(t, yaws, label='Yaw (Spin)', linewidth=2)
        ax.plot(t, pitches, label='Pitch', linewidth=2)
        ax.plot(t, rolls, label='Roll', linewidth=2)
        
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Angle (degrees)')
        ax.set_title(f"{title} - Disc Orientation")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _plot_analysis_summary(disc_analysis: Dict, throw_name: str) -> Figure:
        """Plot analysis summary dashboard."""
        fig = Figure(figsize=(12, 8))
        
        # Flight parameters
        ax = fig.add_subplot(2, 3, 1)
        metrics = {
            'Distance\n(m)': disc_analysis.get('flight_distance', 0) / 1000,
            'Height\n(m)': disc_analysis.get('max_height', 0) / 1000,
            'Duration\n(s)': disc_analysis.get('flight_duration', 0)
        }
        ax.bar(metrics.keys(), metrics.values(), color=['blue', 'green', 'orange'])
        ax.set_title('Flight Parameters')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Velocity profile
        ax = fig.add_subplot(2, 3, 2)
        velocities = {
            'Release': disc_analysis.get('release_velocity', 0) / 1000,
            'Peak': disc_analysis.get('peak_velocity', 0) / 1000,
            'Average': disc_analysis.get('average_velocity', 0) / 1000
        }
        ax.bar(velocities.keys(), velocities.values(), color=['red', 'darkred', 'maroon'])
        ax.set_ylabel('Velocity (m/s)')
        ax.set_title('Velocity Profile')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Disc dynamics
        ax = fig.add_subplot(2, 3, 3)
        ax.text(0.5, 0.5, "DISC DYNAMICS\n" +
               f"Spin: {disc_analysis.get('spin_rate', 0):.0f} RPM\n" +
               f"Stability: {disc_analysis.get('stability_index', 0):.2f}\n" +
               f"Wobble: {disc_analysis.get('wobble_frequency', 0):.1f} Hz",
               ha='center', va='center', fontsize=11, family='monospace',
               bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('Flight Dynamics')
        
        # Angle and glide ratio
        ax = fig.add_subplot(2, 3, 4)
        props = {
            'Angle (°)': disc_analysis.get('flight_angle', 0),
            'Glide Ratio': disc_analysis.get('glide_ratio', 0)
        }
        ax.bar(props.keys(), props.values(), color=['cyan', 'purple'])
        ax.set_title('Flight Characteristics')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Angular velocities
        ax = fig.add_subplot(2, 3, 5)
        ang_vel = {
            'Yaw': disc_analysis.get('angular_velocity_yaw', 0),
            'Pitch': disc_analysis.get('angular_velocity_pitch', 0),
            'Roll': disc_analysis.get('angular_velocity_roll', 0)
        }
        ax.bar(ang_vel.keys(), [abs(v) if v else 0 for v in ang_vel.values()], 
              color=['red', 'green', 'blue'])
        ax.set_ylabel('Angular Velocity (rad/s)')
        ax.set_title('Rotational Motion')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Title
        ax = fig.add_subplot(2, 3, 6)
        ax.text(0.5, 0.5, f"{throw_name}\nFlight Analysis",
               ha='center', va='center', fontsize=14, weight='bold',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        fig.suptitle('Disc Flight Analysis Summary', fontsize=16, weight='bold')
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _format_analysis_summary(disc_analysis) -> str:
        """Format analysis results as text."""
        text = "DISC FLIGHT ANALYSIS RESULTS\n" + "="*50 + "\n\n"
        
        text += "FLIGHT CHARACTERISTICS:\n"
        text += f"  Flight Duration: {disc_analysis.get('flight_duration', 0):.2f}s\n"
        text += f"  Flight Distance: {disc_analysis.get('flight_distance', 0):.0f} mm\n"
        text += f"  Max Height: {disc_analysis.get('max_height', 0):.0f} mm\n"
        text += f"  Flight Angle: {disc_analysis.get('flight_angle', 0):.1f}°\n"
        text += f"  Glide Ratio: {disc_analysis.get('glide_ratio', 0):.2f}\n\n"
        
        text += "VELOCITY PROFILE:\n"
        text += f"  Release Velocity: {disc_analysis.get('release_velocity', 0):.0f} mm/s\n"
        text += f"  Peak Velocity: {disc_analysis.get('peak_velocity', 0):.0f} mm/s\n"
        text += f"  Average Velocity: {disc_analysis.get('average_velocity', 0):.0f} mm/s\n\n"
        
        text += "DISC DYNAMICS:\n"
        if disc_analysis.get('spin_rate'):
            text += f"  Spin Rate: {disc_analysis['spin_rate']:.0f} RPM\n"
        if disc_analysis.get('stability_index'):
            text += f"  Stability Index: {disc_analysis['stability_index']:.2f}\n"
        if disc_analysis.get('wobble_frequency'):
            text += f"  Wobble Frequency: {disc_analysis['wobble_frequency']:.1f} Hz\n"
        if disc_analysis.get('wobble_amplitude'):
            text += f"  Wobble Amplitude: {disc_analysis['wobble_amplitude']:.1f} mm\n\n"
        
        text += "RELEASE CHARACTERISTICS:\n"
        text += f"  Release Height: {disc_analysis.get('release_z_position', 0):.0f} mm\n"
        if disc_analysis.get('release_tilt_angle'):
            text += f"  Release Tilt: {disc_analysis['release_tilt_angle']:.1f}°\n"
        
        return text


def main():
    """Run application."""
    app = QApplication(sys.argv)
    window = DiscGolfAnalysisApp()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
