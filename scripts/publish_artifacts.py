"""Publish v3 research assets; never restore retired RL artifacts."""
import argparse
from chessengine.common import read_json
from chessengine.evolution_report import report, release
from chessengine.research_catalog import publish

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config', default='configs/evolutionary.json')
parser.add_argument('--report-only', action='store_true')
args = parser.parse_args()
config = read_json(args.config)
report(config)
if not args.report_only:
    print(release(config))
print(f'Published {len(publish())} research entries; no retired RL weights copied')
