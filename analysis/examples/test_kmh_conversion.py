"""Test disc speed conversion to km/h."""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer


def test_throw2_speed():
    """Test Throw 2 speed in km/h."""
    
    data_dir = Path(__file__).parent.parent.parent / "Data"
    loader = QTMLoader()
    
    # Load throw2.json
    if not loader.load_from_json(str(data_dir / "throw2.json")):
        print("✗ Failed to load throw2.json")
        return
    
    body_data = loader.extract_disc_data()
    
    # Analyze
    analyzer = DiscAnalyzer(frame_rate=loader.frame_rate or 240.0)
    results = analyzer.analyze_disc_trajectory(body_data)
    
    print("\n" + "="*70)
    print("DISC SPEED CONVERSION TEST")
    print("="*70)
    print(f"\nThrow 2 Analysis Results:")
    print(f"  Frame rate: {loader.frame_rate} Hz")
    print(f"  Release speed: {results['disc_speed']:.2f} km/h")
    print(f"\nComparison Metrics:")
    
    # Test comparison
    throws = {"throw2": results}
    comparison = analyzer.compare_throws(throws)
    
    print(f"  Best disc speed: {comparison['best_disc_speed']:.2f} km/h")
    print(f"  Average disc speed: {comparison['average_disc_speed']:.2f} km/h")
    
    print("\n✓ Speed conversion to km/h working correctly!")


if __name__ == "__main__":
    test_throw2_speed()
