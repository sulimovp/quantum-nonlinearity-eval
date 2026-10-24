"""Robust matplotlib figure export for headless / sweep runs (Windows-safe)."""

from __future__ import annotations

import os
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent


def repo_figures_dir() -> Path:
    """Shared ``figures/`` directory at repo root (created if missing)."""
    d = _REPO_ROOT / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def safe_savefig(path: str | Path, *, max_attempts: int = 5, **kwargs) -> bool:
    """Save the current pyplot figure; return False instead of raising on failure."""
    import matplotlib.pyplot as plt

    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    last_err: OSError | None = None
    for attempt in range(max_attempts):
        tmp = dest.with_name(f"{dest.stem}.tmp{attempt}{dest.suffix}")
        try:
            plt.savefig(tmp, **kwargs)
            if dest.exists():
                dest.unlink()
            os.replace(tmp, dest)
            return True
        except OSError as exc:
            last_err = exc
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            time.sleep(0.25 * (attempt + 1))
    print(f"WARNING: safe_savefig failed for {dest}: {last_err}")
    return False
