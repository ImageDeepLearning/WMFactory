from __future__ import annotations

import sys
from pathlib import Path

# worldfmEnd/server.py -> parents[1] == WMFactory root
WMFACTORY_ROOT = Path(__file__).resolve().parents[1]
WORLDFM_END_ROOT = Path(__file__).resolve().parent


def ensure_worldfm_on_path() -> None:
    """Allow `import app` from services/worldfm (same as uvicorn cwd there)."""
    wf = WMFACTORY_ROOT / "services" / "worldfm"
    s = str(wf)
    if s not in sys.path:
        sys.path.insert(0, s)
