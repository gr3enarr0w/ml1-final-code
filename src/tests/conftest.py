from __future__ import annotations

import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[1] / "ncaa_pred"
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))
