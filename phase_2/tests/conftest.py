"""Put this phase's own package root on sys.path.

Each phase ships an independent `driftforge`, so the suites are run one phase
at a time from inside that phase's folder:  python -m pytest tests -q
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
