"""PyQt6-based GUI application for disc golf throw analysis."""

import sys
import re
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


def natural_sort_key(path_or_str):
    """
    Generate a natural sort key for paths/strings with numeric components.
    
    Examples:
        Throw_2.json  -> ['Throw_', 2, '.json']
        Throw_10.json -> ['Throw_', 10, '.json']
        
    This ensures Throw_2 sorts before Throw_10 (numeric sort, not lexicographic).
    """
    name = str(path_or_str).lower()
    # Split the string into text and numeric parts
    return [int(text) if text.isdigit() else text for text in re.split(r'(\d+)', name)]


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
        # On start, ask for a folder to watch for JSON exports
        QTimer.singleShot(100, self.select_folder_on_start)

        # Files already processed
        self.processed_files = set()
        # Map of available throw name -> path
        self.available_throws = {}
        # Poll timer for new files
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(2000)
        self.poll_timer.timeout.connect(self.poll_folder)
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

        folder_btn = QPushButton("Select Session Folder")
        folder_btn.clicked.connect(self.select_folder_dialog)
        control_layout.addWidget(folder_btn)
        
        # Removed separate "Load Throw Data" and "Analyze" buttons
        control_layout.addWidget(QLabel("Select Throw:"))
        self.throw_combo = QComboBox()
        # Will be populated from selected session folder
        self.throw_combo.addItems([])
        self.throw_combo.currentIndexChanged.connect(self.on_throw_selected)
        control_layout.addWidget(self.throw_combo)
        
        layout.addLayout(control_layout)
        
        # Progress bar
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        
        # Tab widget for visualizations (left) + metrics panel (right)
        content_layout = QHBoxLayout()

        self.tabs = QTabWidget()
        self.trajectory_3d = FigureCanvas(Figure(figsize=(6, 5), dpi=100))
        self.tabs.addTab(self.trajectory_3d, "Disc Path")

        # Inspector tab only (velocity + filtered accel)
        self.inspector_plot = FigureCanvas(Figure(figsize=(6, 5), dpi=100))
        self.tabs.addTab(self.inspector_plot, "Inspector")

        content_layout.addWidget(self.tabs, stretch=3)

        # Metrics panel: show the six key parameters
        self.metrics_label = QLabel()
        self.metrics_label.setMinimumWidth(340)
        self.metrics_label.setWordWrap(True)
        # Prominent boxed styling for key parameters
        self.metrics_label.setStyleSheet(
            "background: #f2f2f2; border: 1px solid #888; border-radius: 6px; "
            "padding: 12px; color: #111; font-family: 'Consolas', 'Courier New', monospace; "
            "font-size: 14pt;"
        )
        self.metrics_label.setText('<div style="font-family:monospace; font-size:14pt;">No analysis yet</div>')
        self.metrics_label.setAlignment(Qt.AlignmentFlag.AlignTop)
        content_layout.addWidget(self.metrics_label, stretch=1)

        layout.addLayout(content_layout)
        
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
            if hasattr(self.qtm_loader, 'load_from_json') and self.selected_folder:
                # No-op: using folder polling to load JSON files
                pass
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
        
        # Generate synthetic marker data (Mid marker = same as body center)
        markers = {
            'Mid': positions.copy(),
            'Center': positions + np.random.normal(0, 2, positions.shape),
            'Wide': positions + np.random.normal(0, 3, positions.shape),
            'Close': positions + np.random.normal(0, 2, positions.shape)
        }
        
        self.body_data = {
            'position': positions,
            'rotation': rotations,
            'residual': residuals,
            'markers': markers
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

        # Extract 6DOF data for visualization
        positions = self.body_data['position']
        rotations = self.body_data['rotation']

        # Get net impact frame if detected
        net_impact_frame = disc_analysis.get('net_impact_frame')

        # Disc Path - Multiple views (top, side, back)
        fig = self._plot_body_trajectory_multiview(
            positions, 
            disc_analysis.get('release_frame'), 
            net_impact_frame,
            self.current_throw_name
        )
        self.trajectory_3d.figure = fig
        self.trajectory_3d.draw()

        # Inspector: velocity and filtered acceleration
        fig = self._plot_inspector(positions, disc_analysis, net_impact_frame)
        self.inspector_plot.figure = fig
        self.inspector_plot.draw()

        # Update metrics panel with the six key parameters
        metrics = {
            'Disc Speed (km/h)': disc_analysis.get('disc_speed'),
            'Spin (RPM)': disc_analysis.get('spin'),
            'Hyzer (+hyzer)': disc_analysis.get('hyzer_angle'),
            'Launch (°)': disc_analysis.get('launch_angle'),
            'Nose (+up)': disc_analysis.get('nose_angle'),
            'Wobble RMS (°)': disc_analysis.get('wobble_amplitude'),
        }
        lines = []
        import numbers
        for k, v in metrics.items():
            if v is None:
                lines.append(f"{k:<16} --")
            elif isinstance(v, numbers.Number):
                # format numeric values (handles numpy types)
                if 'km/h' in k:
                    lines.append(f"{k:<16} {float(v):.1f}")
                else:
                    lines.append(f"{k:<16} {float(v):.2f}")
            else:
                lines.append(f"{k:<16} {v}")

        # Render metrics as preformatted text for alignment
        html = "<div style='font-family:monospace; font-size:14pt;'><b>Key Parameters</b><pre>" + "\n".join(lines) + "</pre></div>"
        self.metrics_label.setText(html)
    
    @staticmethod
    def _plot_body_trajectory_multiview(
        positions: np.ndarray, 
        release_frame: int = None, 
        net_impact_frame: int = None,
        title: str = ""
    ) -> Figure:
        """Plot disc trajectory from three views: top (X-Y), side (Z-X), back (Z-Y).

        Args:
            positions: (N,3) array of 6DOF body positions
            release_frame: optional index into positions for release marker
            net_impact_frame: optional index for net impact (truncates visualization)
            title: plot title
        """
        fig = Figure(figsize=(8, 12))
        
        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_pos = positions[valid_mask]
        
        # Truncate at net impact if detected
        if net_impact_frame is not None and net_impact_frame < len(valid_pos):
            viz_pos = valid_pos[:net_impact_frame]
        else:
            viz_pos = valid_pos
        
        # Helper function to plot a single view
        def plot_view(ax, x_data, y_data, x_label, y_label, view_name):
            ax.plot(x_data, y_data, 'b-', linewidth=2, label='Flight Path')
            
            # Start marker
            if len(viz_pos) > 0:
                ax.scatter(x_data[0], y_data[0], s=80, c='green', marker='o', 
                          label='Start', edgecolors='darkgreen', linewidths=2)
            
            # Release marker if available
            if release_frame is not None and 0 <= int(release_frame) < len(viz_pos):
                ax.scatter(x_data[int(release_frame)], y_data[int(release_frame)], s=120, c='orange', 
                          marker='D', label='Release', edgecolors='darkorange', linewidths=2)
            
            # Net impact marker
            if net_impact_frame is not None and 0 < net_impact_frame < len(valid_pos):
                net_x = valid_pos[net_impact_frame, 0] if x_label == 'X (mm)' else valid_pos[net_impact_frame, 2] if x_label == 'Z (mm)' else valid_pos[net_impact_frame, 0]
                net_y = valid_pos[net_impact_frame, 1] if y_label == 'Y (mm)' else valid_pos[net_impact_frame, 2] if y_label == 'Z (mm)' else valid_pos[net_impact_frame, 1]
                ax.scatter(net_x, net_y, s=150, c='red', marker='X', 
                          label='Net Impact', edgecolors='darkred', linewidths=2)
            
            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            ax.set_aspect('equal', adjustable='box')
            ax.grid(True, alpha=0.3)
            ax.set_title(f"{view_name}")
            ax.legend(loc='best', fontsize=8)
        
        # Top view (X-Y plane)
        ax1 = fig.add_subplot(311)
        plot_view(ax1, viz_pos[:, 0], viz_pos[:, 1], 'X (mm)', 'Y (mm)', 'Top View (X-Y)')
        
        # Side view (Z-X plane, looking from the right)
        ax2 = fig.add_subplot(312)
        plot_view(ax2, viz_pos[:, 0], viz_pos[:, 2], 'X (mm)', 'Z (mm)', 'Side View (Z-X)')
        
        # Back view (Z-Y plane, looking from behind)
        ax3 = fig.add_subplot(313)
        plot_view(ax3, viz_pos[:, 1], viz_pos[:, 2], 'Y (mm)', 'Z (mm)', 'Back View (Z-Y)')
        
        if net_impact_frame is not None:
            fig.suptitle(f"{title} - Disc Path (to Net Impact)", fontsize=14, fontweight='bold')
        else:
            fig.suptitle(f"{title} - Disc Path", fontsize=14, fontweight='bold')
        
        fig.tight_layout()
        return fig
    
    @staticmethod
    def _plot_body_trajectory_3d(
        positions: np.ndarray, 
        release_frame: int = None, 
        net_impact_frame: int = None,
        title: str = ""
    ) -> Figure:
        """Plot top-down disc trajectory (X-Y plane) using 6DOF body position with start, release, net impact markers.

        Args:
            positions: (N,3) array of 6DOF body positions
            release_frame: optional index into positions for release marker
            net_impact_frame: optional index for net impact (truncates visualization)
            title: plot title
        """
        fig = Figure(figsize=(8, 6))
        ax = fig.add_subplot(111)

        valid_mask = ~np.isnan(positions).any(axis=1)
        valid_pos = positions[valid_mask]
        
        # Truncate at net impact if detected
        if net_impact_frame is not None and net_impact_frame < len(valid_pos):
            viz_pos = valid_pos[:net_impact_frame]
        else:
            viz_pos = valid_pos

        ax.plot(viz_pos[:, 0], viz_pos[:, 1],
               'b-', linewidth=2, label='Flight Path')
        
        # Start marker (first valid point)
        if len(viz_pos) > 0:
            ax.scatter(viz_pos[0, 0], viz_pos[0, 1], s=80, c='green', marker='o', label='Start', edgecolors='darkgreen', linewidths=2)
        
        # Release marker if available
        if release_frame is not None and 0 <= int(release_frame) < len(viz_pos):
            ax.scatter(viz_pos[int(release_frame), 0], viz_pos[int(release_frame), 1], s=120, c='orange', marker='D', label='Release', edgecolors='darkorange', linewidths=2)
        
        # Net impact marker if available
        if net_impact_frame is not None and 0 < net_impact_frame < len(valid_pos):
            # Use full valid_pos to get the actual impact point
            ax.scatter(valid_pos[net_impact_frame, 0], valid_pos[net_impact_frame, 1], s=150, c='red', marker='X', label='Net Impact', edgecolors='darkred', linewidths=2)

        ax.set_xlabel('X (mm)')
        ax.set_ylabel('Y (mm)')
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, alpha=0.3)
        if net_impact_frame is not None:
            ax.set_title(f"{title} - Disc Path - Top View (to Net Impact)")
        else:
            ax.set_title(f"{title} - Disc Path - Top View")
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

    def _plot_inspector(self, positions: np.ndarray, disc_analysis: Dict, net_impact_frame: int = None) -> Figure:
        """Plot disc velocity and filtered acceleration with release and net impact markers.
        
        Args:
            positions: Position array
            disc_analysis: Analysis results dictionary
            net_impact_frame: Optional frame index for net impact (truncates visualization)
        """
        fig = Figure(figsize=(8, 6))
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212, sharex=ax1)

        dt = 1.0 / 240.0
        velocities = np.diff(positions, axis=0) / dt
        speeds = np.linalg.norm(velocities, axis=1)

        # Acceleration from speeds
        if len(speeds) >= 2:
            acc = np.diff(speeds) / dt
        else:
            acc = np.array([])

        # Filter acceleration with simple moving average
        window = 5
        if len(acc) >= window and window > 1:
            padded = np.concatenate([np.full(window-1, acc[0]), acc])
            acc_filt = np.convolve(padded, np.ones(window)/window, mode='valid')[:len(acc)]
        else:
            acc_filt = acc

        # Truncate at net impact if detected
        if net_impact_frame is not None:
            speeds = speeds[:net_impact_frame]
            acc = acc[:net_impact_frame] if net_impact_frame <= len(acc) else acc
            acc_filt = acc_filt[:net_impact_frame] if net_impact_frame <= len(acc_filt) else acc_filt

        # X axes (frame indices) - align speeds to positions by using frame index = position index +1
        sf = np.arange(1, len(speeds) + 1)
        af = np.arange(2, 2 + len(acc))  # acceleration corresponds to frames starting at 2

        ax1.plot(sf, speeds, color='tab:blue', label='Speed (mm/s)', linewidth=1.5)
        rel = disc_analysis.get('release_frame')
        if rel is not None and rel >= 0 and len(sf) > 0 and rel <= sf[-1]:
            ax1.axvline(rel, color='orange', linestyle='--', linewidth=2, label='Release')
        if net_impact_frame is not None and net_impact_frame >= 0 and len(sf) > 0:
            ax1.axvline(net_impact_frame, color='red', linestyle='--', linewidth=2, label='Net Impact')
        ax1.set_ylabel('Speed (mm/s)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(af, acc, color='tab:gray', label='Acceleration', linewidth=1)
        if len(acc_filt) > 0:
            ax2.plot(af, acc_filt, color='tab:green', label='Filtered Accel', linewidth=1.5)
        if rel is not None and rel >= 2 and len(af) > 0 and rel <= af[-1]:
            ax2.axvline(rel, color='orange', linestyle='--', linewidth=2)
        if net_impact_frame is not None and net_impact_frame >= 2 and len(af) > 0:
            ax2.axvline(net_impact_frame, color='red', linestyle='--', linewidth=2)
        ax2.set_ylabel('Acceleration (mm/s²)')
        ax2.set_xlabel('Frame index')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        fig.tight_layout()
        return fig

    def select_folder_on_start(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select session folder', '')
        if folder:
            self.selected_folder = folder
            self.processed_files = set()
            self.poll_timer.start()
            self.statusBar().showMessage(f'Watching: {folder}')
            # populate available throws
            self.update_throw_list()
        else:
            self.selected_folder = None

    def select_folder_dialog(self):
        folder = QFileDialog.getExistingDirectory(self, 'Select session folder', '')
        if folder:
            self.selected_folder = folder
            self.processed_files = set()
            self.poll_timer.start()
            self.statusBar().showMessage(f'Watching: {folder}')
            self.update_throw_list()

    def update_throw_list(self):
        """Scan the selected folder for JSON export files and populate the throw combo."""
        from pathlib import Path
        if not getattr(self, 'selected_folder', None):
            return
        p = Path(self.selected_folder)
        jsons = sorted(p.rglob('*.json'), key=natural_sort_key)
        self.available_throws = {j.name: str(j) for j in jsons}

        # Update combo while preserving current selection if possible
        current = self.throw_combo.currentText()
        self.throw_combo.blockSignals(True)
        self.throw_combo.clear()
        self.throw_combo.addItems(list(self.available_throws.keys()))
        # Restore selection if it exists
        if current and current in self.available_throws:
            idx = list(self.available_throws.keys()).index(current)
            self.throw_combo.setCurrentIndex(idx)
        self.throw_combo.blockSignals(False)

    def on_throw_selected(self, index: int):
        """Load and analyze the selected throw (by combo index)."""
        try:
            if index < 0:
                return
            name = self.throw_combo.itemText(index)
            path = self.available_throws.get(name)
            if not path:
                return

            loader = QTMLoader()
            ok = loader.load_from_json(path)
            if not ok:
                self.statusBar().showMessage(f'Failed to load: {name}')
                return
            body = loader.extract_disc_data()
            if body is None:
                self.statusBar().showMessage(f'No body data in: {name}')
                return

            analyzer = DiscAnalyzer(frame_rate=loader.frame_rate or 240.0)
            res = analyzer.analyze_disc_trajectory(body)
            self.body_data = body
            self.current_throw_name = name
            self.update_visualizations(res)
            self.statusBar().showMessage(f'Loaded: {name}')
        except Exception as e:
            import logging as _logging
            _logging.getLogger(__name__).exception('Failed to load selected throw: %s', e)

    def poll_folder(self):
        """Poll the selected folder for new JSON files (typically from QTM exports)."""
        if not getattr(self, 'selected_folder', None):
            return
        
        from pathlib import Path
        import logging as _logging
        logger = _logging.getLogger(__name__)
        
        p = Path(self.selected_folder)
        
        # Look for JSON files in the directory (not recursive to avoid subdirectories)
        jsons = sorted(p.glob('*.json'), key=natural_sort_key)
        
        if not jsons:
            # Optionally try one level deep (e.g., Data/2026-02-12/Throw_1.json)
            jsons = sorted(p.glob('*/*.json'), key=natural_sort_key)
        
        new_files_found = 0
        latest_throw_name = None
        
        for j in jsons:
            if str(j) in self.processed_files:
                continue
            
            new_files_found += 1
            print(f"[Polling] New file detected: {j.name}")
            
            # Process new file
            try:
                print(f"  Loading {j.name}...")
                loader = QTMLoader()
                ok = loader.load_from_json(str(j))
                
                if not ok:
                    msg = f"Failed to load JSON: {j.name}"
                    print(f"  Error: {msg}")
                    self.statusBar().showMessage(f"Error: {msg}")
                    continue
                
                print(f"  Extracting 6DOF data from {j.name}...")
                body = loader.extract_disc_data()
                
                if body is None:
                    msg = f"No 6DOF body data in: {j.name}"
                    print(f"  Error: {msg}")
                    self.statusBar().showMessage(f"Error: {msg}")
                    continue
                
                print(f"  Analyzing trajectory...")
                analyzer = DiscAnalyzer(frame_rate=loader.frame_rate or 240.0)
                res = analyzer.analyze_disc_trajectory(body)
                
                print(f"  Updating visualization...")
                self.body_data = body
                self.current_throw_name = j.name
                latest_throw_name = j.name  # Track the latest throw processed
                self.update_visualizations(res)
                self.processed_files.add(str(j))
                
                # Update available throws list so user can manually select them
                try:
                    self.available_throws[j.name] = str(j)
                except Exception as e:
                    logger.warning('Failed to add throw to available list: %s', e)
                
                msg = f'✓ Processed: {j.name}'
                print(f"  {msg}")
                self.statusBar().showMessage(msg)
                
            except Exception as e:
                msg = f'Failed to process {j.name}: {str(e)}'
                print(f"  ERROR: {msg}")
                self.statusBar().showMessage(f"Error: {msg}")
                logger.exception('Failed to process %s', j, exc_info=True)
        
        # Update throw list and select the latest throw if one was processed
        if new_files_found > 0:
            self.update_throw_list()
            # Select the newest throw in the combo box to display it
            if latest_throw_name:
                found_index = -1
                for i in range(self.throw_combo.count()):
                    if self.throw_combo.itemText(i) == latest_throw_name:
                        found_index = i
                        break
                if found_index >= 0:
                    self.throw_combo.blockSignals(False)  # Ensure signals are not blocked
                    self.throw_combo.setCurrentIndex(found_index)
                    print(f"[Polling] Selected newest throw: {latest_throw_name}")
        
        if new_files_found == 0 and jsons:
            # Just print to console, don't clutter status bar
            print(f"[Polling] Watching {self.selected_folder}: {len(jsons)} file(s), all processed")
    
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
