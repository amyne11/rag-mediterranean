"""pytest config — makes src/ importable when running `pytest` from repo root."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
