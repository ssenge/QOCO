from __future__ import annotations

import importlib
import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

for _name in ("core", "optimizers", "converters", "pipeline", "preproc", "utils", "examples", "tests"):
    _module = importlib.import_module(_name)
    sys.modules[f"qoco.{_name}"] = _module
    setattr(sys.modules[__name__], _name, _module)
