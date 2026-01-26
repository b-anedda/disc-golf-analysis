"""Analyze all QTM JSON throws in a session folder and save CSV summary.

Usage:
    - Run without arguments to open a folder selection dialog.
    - Or pass a folder path as the first argument to run headless:
            python session_analyzer_gui.py "C:/.../Data/subject/session"

The script analyzes each `.json` file in the selected folder using
`QTMLoader` and `DiscAnalyzer`, collects the key release parameters,
and saves a summary `session_analysis.csv` in the same folder.
"""
from pathlib import Path
import sys
import traceback
import logging

try:
    import tkinter as tk
    from tkinter import filedialog, messagebox
except Exception:
    tk = None

try:
    import pandas as pd
except ImportError:
    pd = None

# Make local package importable when running from examples/
from pathlib import Path as _P
import sys as _sys
_sys.path.insert(0, str((_P(__file__).parent).parent))
from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer

# Use logging so example scripts stay quiet when imported
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def analyze_session_folder(folder: Path) -> Path:
    folder = Path(folder)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    # Search for JSON files directly in folder; if none, search recursively
    json_files = sorted(folder.glob('*.json'))
    if not json_files:
        json_files = sorted(folder.rglob('*.json'))
    if not json_files:
        raise FileNotFoundError(f"No .json files found in {folder} (including subfolders)")

    results = []
    for jf in json_files:
        try:
            loader = QTMLoader()
            ok = loader.load_from_json(str(jf))
            if not ok:
                logging.warning("Failed to load %s; skipping", jf.name)
                continue

            body = loader.extract_disc_data()
            if body is None:
                logging.warning("No body data in %s; skipping", jf.name)
                continue

            analyzer = DiscAnalyzer(frame_rate=loader.frame_rate or 240.0)
            res = analyzer.analyze_disc_trajectory(body)

            row = {
                'file': jf.name,
                'disc_speed_kmh': res.get('disc_speed'),
                'spin_rpm': res.get('spin'),
                'hyzer_deg': res.get('hyzer_angle'),
                'launch_deg': res.get('launch_angle'),
                'nose_deg': res.get('nose_angle'),
                'release_frame': res.get('release_frame'),
                'wobble_deg_rms': res.get('wobble_amplitude'),
                'wobble_deg_range': res.get('wobble_range'),
                'release_z_mm': res.get('release_z_position'),
                'release_tilt_deg': res.get('release_tilt_angle'),
            }
            results.append(row)
            logging.info("Processed %s", jf.name)
        except Exception as e:
            logging.exception("Error processing %s: %s", jf.name, e)

    df = pd.DataFrame(results)
    out_path = folder / 'session_analysis.csv'
    df.to_csv(out_path, index=False)
    return out_path


def main(argv=None):
    argv = argv or sys.argv[1:]

    if pd is None:
        logging.error("Missing dependency: pandas (required to write CSV).")
        logging.info("Install with: pip install pandas")
        return 1

    folder = None
    if argv:
        folder = Path(argv[0])
    else:
        if tk is None:
            print("tkinter not available; pass a folder path as argument")
            return 1
        root = tk.Tk()
        root.withdraw()
        folder_path = filedialog.askdirectory(title='Select session folder containing JSON files')
        if not folder_path:
            logging.info("No folder selected")
            return 1
        folder = Path(folder_path)

    try:
        out = analyze_session_folder(folder)
        logging.info("Saved summary to: %s", out)
        if tk is not None and not argv:
            messagebox.showinfo('Done', f'Saved summary to:\n{out}')
        return 0
    except Exception as e:
        logging.exception("Error: %s", e)
        if tk is not None and not argv:
            messagebox.showerror('Error', str(e))
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
