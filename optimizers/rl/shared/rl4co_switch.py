from __future__ import annotations

"""Runtime switch to use a local RL4CO source checkout.

We want to be able to tweak RL4CO internals (NN arch, decoding, etc.) while keeping
the rest of this repo unchanged. This helper supports a simple workflow:

- Default: use the installed `rl4co` package from the active Python environment.
- If `USE_RL4CO_SOURCE=1`: prepend the vendored RL4CO repo to `sys.path` so that
  `import rl4co` resolves to local source code.

Important:
- This must run **before** the first `import rl4co` anywhere in the process.
- If `rl4co` is already imported, we intentionally raise (internal repo, fail fast).
"""

import os
import sys
from pathlib import Path


def maybe_enable_local_rl4co() -> None:
    """Enable local RL4CO source import if `USE_RL4CO_SOURCE=1`."""
    flag = os.environ.get("USE_RL4CO_SOURCE", "").strip().lower()
    if flag not in {"1", "true", "yes", "y"}:
        return

    # `.../src/rl/rl4co_switch.py` -> repo root is `parents[2]` (repo/src/rl).
    repo_root = Path(__file__).resolve().parents[2]
    base = repo_root / "graveyard" / "rl4co-github"

    # Prefer a real git clone if present (easier to `git pull`), otherwise fall back to ZIP snapshot.
    candidates = [
        base / "rl4co-upstream",
        base / "rl4co-main",
    ]
    src_root = next((p for p in candidates if p.exists()), None)
    if src_root is None:
        raise FileNotFoundError(
            "No local RL4CO source checkout found. Expected one of:\n"
            + "\n".join(f"  - {p}" for p in candidates)
        )

    # If rl4co is already imported, be idempotent:
    # - if it already comes from local source: do nothing
    # - otherwise: fail fast (can't reliably swap after import)
    if "rl4co" in sys.modules:
        mod = sys.modules["rl4co"]
        mod_file = str(getattr(mod, "__file__", "") or "")
        if str(src_root) in mod_file:
            return
        raise RuntimeError(
            "USE_RL4CO_SOURCE=1 but `rl4co` is already imported from a non-local path:\n"
            f"  rl4co.__file__ = {mod_file}\n"
            "Start a fresh process and ensure local RL4CO is enabled before importing `rl4co`."
        )

    # Prepend so it takes priority over installed site-packages.
    if sys.path and sys.path[0] == str(src_root):
        return
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

