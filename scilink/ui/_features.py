"""Feature-flag probes for the Streamlit UI.

Today there is one flag: whether simulate mode (Simulate mode button,
HPC sidebar, Simulations tab) is exposed. The flag is True iff the
optional `[sim]` extras are installed — `paramiko` (HPC connectivity)
and `ase` (structure handling).

Without `[sim]`, the UI reverts to the analyze/plan-only shape that
predates the simulate-mode work.
"""
from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def simulate_enabled() -> bool:
    try:
        import paramiko  # noqa: F401
        import ase  # noqa: F401
    except ImportError:
        return False
    return True
