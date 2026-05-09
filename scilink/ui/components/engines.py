"""Engine registry for the Generate tab in the simulations UI.

Each entry maps to a Streamlit-rendering callback that owns its own
configure -> review -> submit workflow inside the Generate tab. The list
is hardcoded today; a future capability-inventory agent (see CLAUDE.md,
"the meta agent" section) will populate it from declared software +
introspected agents/skills/tools, replacing the static list without
changing the consumer (`simulations.py`).
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, List, Tuple


@dataclass(frozen=True)
class Engine:
    """An engine the Generate tab can drive.

    `render_workflow` is a no-arg Streamlit-rendering callback that owns
    the Generate-tab content for this engine, typically a state machine
    over configure -> review -> monitoring -> results phases.
    """
    key: str
    label: str
    icon: str
    render_workflow: Callable[[], None]


@lru_cache(maxsize=1)
def _engines() -> Tuple[Engine, ...]:
    # Imports are local to dodge any future circulars (sim_workflow /
    # vasp_workflow have no dependency on this module today).
    from scilink.ui.components import sim_workflow, vasp_workflow

    return (
        Engine(
            key="lammps",
            label="LAMMPS",
            icon="🧪",
            render_workflow=sim_workflow.render_agent_workflow,
        ),
        Engine(
            key="vasp",
            label="VASP",
            icon="⚛️",
            render_workflow=vasp_workflow.render_agent_workflow,
        ),
    )


def get_engines() -> List[Engine]:
    """Registered engines in display order."""
    return list(_engines())


def get_engine(key: str) -> Engine:
    """Look up by key. Raises KeyError if not registered."""
    for e in _engines():
        if e.key == key:
            return e
    raise KeyError(f"Unknown engine: {key!r}")
