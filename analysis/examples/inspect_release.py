"""Plot velocity and acceleration traces and annotate detected release frame.

Usage:
  python inspect_release.py [path/to/throw.json]

Generates a PNG in the same folder named `<throw>_inspect.png` and also
attempts to show an interactive window.
"""
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt
import logging

sys.path.insert(0, str((Path(__file__).parent).parent))
from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def smooth_acceleration(accel: np.ndarray, window_size: int = 5) -> np.ndarray:
    if len(accel) <= window_size:
        return accel
    padded = np.concatenate([np.full(window_size - 1, accel[0]), accel])
    filtered = np.convolve(padded, np.ones(window_size) / window_size, mode='valid')
    return filtered[: len(accel)]


def inspect_file(path: Path):
    loader = QTMLoader()
    ok = loader.load_from_json(str(path))
    if not ok:
        logging.error('Failed to load %s', path)
        return 1

    body = loader.extract_disc_data()
    if body is None:
        logging.error('No body data in %s', path)
        return 1

    frame_rate = loader.frame_rate or 240.0
    analyzer = DiscAnalyzer(frame_rate=frame_rate)

    positions = body['position']
    if positions.shape[0] < 2:
        logging.error('Not enough frames in %s', path)
        return 1

    dt = 1.0 / frame_rate
    velocities = np.diff(positions, axis=0) / dt
    speeds = np.linalg.norm(velocities, axis=1)
    accelerations = np.diff(speeds) / dt

    # For plotting alignment, accelerations correspond to frames 1..N-2
    accel_frames = np.arange(len(accelerations))
    speed_frames = np.arange(len(speeds))

    # Smoothed acceleration used by detection
    accel_filtered = smooth_acceleration(np.diff(speeds), window_size=5)

    # Jerk
    jerk = None
    if len(accel_filtered) > 1:
        jerk = np.diff(accel_filtered)

    # Detect release using analyzer internal method
    release_idx = analyzer._detect_release_frame(speeds)
    logging.info('Detected release frame (speeds index): %s', release_idx)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)

    axes[0].plot(speed_frames, speeds, label='Speed (mm/s)')
    axes[0].axvline(release_idx, color='red', linestyle='--', label='Detected release')
    axes[0].set_ylabel('Speed (mm/s)')
    axes[0].legend()

    axes[1].plot(accel_frames, accelerations, label='Acceleration (mm/s²)')
    if len(accel_filtered) == len(accelerations):
        axes[1].plot(accel_frames, accel_filtered, label='Filtered accel (5-frame MA)')
    axes[1].axvline(release_idx, color='red', linestyle='--')
    axes[1].set_ylabel('Acceleration (mm/s²)')
    axes[1].legend()

    if jerk is not None and len(jerk) > 0:
        jerk_frames = np.arange(len(jerk))
        axes[2].plot(jerk_frames, jerk, label='Jerk (mm/s³)')
        axes[2].axvline(max(0, release_idx - 1), color='red', linestyle='--')
        axes[2].set_ylabel('Jerk (mm/s³)')
        axes[2].legend()
    else:
        axes[2].text(0.1, 0.5, 'Not enough data for jerk plot', transform=axes[2].transAxes)
        axes[2].set_ylabel('Jerk')

    axes[-1].set_xlabel('Frame index (relative to speeds)')

    title = f'Inspect: {path.name}  (frame_rate={frame_rate} Hz)'
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    out_png = path.parent / (path.stem + '_inspect.png')
    fig.savefig(out_png, dpi=150)
    logging.info('Saved inspection plot: %s', out_png)

    try:
        plt.show()
    except Exception:
        logging.info('Interactive show failed (headless); PNG saved instead')

    return 0


if __name__ == '__main__':
    arg = sys.argv[1] if len(sys.argv) > 1 else 'Data/throw2.json'
    path = Path(arg)
    raise SystemExit(inspect_file(path))
