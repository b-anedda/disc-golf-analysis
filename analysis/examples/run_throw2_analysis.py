"""Run DiscAnalyzer on Data/throw2.json and print results."""
from pathlib import Path
import sys
from pprint import pprint
import logging

# Ensure example can import the local package
sys.path.insert(0, str((Path(__file__).parent).parent))

from src.qtm_loader import QTMLoader
from src.throw_analysis import DiscAnalyzer

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def main():
    data_file = Path(__file__).parent.parent.parent / 'Data' / 'throw2.json'
    logging.info('Loading: %s', data_file)

    loader = QTMLoader()
    if not loader.load_from_json(str(data_file)):
        logging.error('Failed to load %s', data_file)
        return 1

    body = loader.extract_disc_data()
    logging.info('Loaded body data shapes: %s %s', body['position'].shape, body['rotation'].shape)

    analyzer = DiscAnalyzer(frame_rate=loader.frame_rate)
    results = analyzer.analyze_disc_trajectory(body)

    logging.info('\nAnalysis results:')
    pprint(results)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
