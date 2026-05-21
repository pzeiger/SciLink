"""Configurable framework-level quality gate for analyze-mode agents.

Historically the only gate was R² ≥ 0.95 (a curve-fitting default that was
hardcoded across :mod:`curve_fitting_controllers`). For workflow-style
skills like ``structure_matching/xrd`` the natural quality metric is a
figure-of-merit or a normalized cost, not R² — so the gate is now a
data-class that the agent resolves from four sources, in priority order:

1. Explicit ``quality_gate=`` keyword on :meth:`CurveFittingAgent.analyze`.
2. Explicit ``quality_gate=`` on :meth:`CurveFittingAgent.__init__`.
3. Skill frontmatter ``quality_gate:`` block (parsed by
   :func:`scilink.skills.loader.load_skill`).
4. Framework default — :data:`R_SQUARED_DEFAULT` (R² ≥ 0.95; preserves
   pre-existing behavior for every curve-fit skill that does not
   declare its own gate).

A legacy ``r2_threshold=`` keyword still works as a shortcut that
constructs :class:`QualityGate` with ``metric='r_squared'``.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional


# How to interpret the threshold:
#   higher_is_better → accept if value >= accept_threshold; hard-reject if value < hard_reject_threshold
#   lower_is_better  → accept if value <= accept_threshold; hard-reject if value > hard_reject_threshold
_DIRECTIONS = {"higher_is_better", "lower_is_better"}


@dataclass(frozen=True)
class QualityGate:
    """Framework-level accept / reject criterion for a single fit result.

    The gate reads ``fit_result["fit_quality"][metric]`` from the script's
    ``FIT_RESULTS_JSON`` output and applies threshold + direction logic.
    A missing or non-numeric value is treated as hard-reject — same
    behavior as the original R² gate when ``fit_quality`` was missing.
    """

    metric: str = "r_squared"
    accept_threshold: float = 0.95
    hard_reject_threshold: float = 0.90
    direction: str = "higher_is_better"

    def __post_init__(self) -> None:
        if self.direction not in _DIRECTIONS:
            raise ValueError(
                f"QualityGate.direction must be one of {sorted(_DIRECTIONS)}; "
                f"got {self.direction!r}"
            )
        if self.direction == "higher_is_better":
            if self.hard_reject_threshold > self.accept_threshold:
                raise ValueError(
                    "higher_is_better: hard_reject_threshold "
                    f"({self.hard_reject_threshold}) must be ≤ accept_threshold "
                    f"({self.accept_threshold})"
                )
        else:
            if self.hard_reject_threshold < self.accept_threshold:
                raise ValueError(
                    "lower_is_better: hard_reject_threshold "
                    f"({self.hard_reject_threshold}) must be ≥ accept_threshold "
                    f"({self.accept_threshold})"
                )

    # -- evaluation -------------------------------------------------------

    def extract(self, fit_quality: Optional[dict]) -> Optional[float]:
        """Read this gate's metric value from a ``fit_quality`` dict.

        Returns ``None`` when missing, non-numeric, or NaN — callers
        treat ``None`` as a hard reject.
        """
        if not isinstance(fit_quality, dict):
            return None
        v = fit_quality.get(self.metric)
        try:
            f = float(v)
        except (TypeError, ValueError):
            return None
        if f != f:  # NaN
            return None
        return f

    def is_accept(self, value: Optional[float]) -> bool:
        if value is None:
            return False
        if self.direction == "higher_is_better":
            return value >= self.accept_threshold
        return value <= self.accept_threshold

    def is_hard_reject(self, value: Optional[float]) -> bool:
        if value is None:
            return True
        if self.direction == "higher_is_better":
            return value < self.hard_reject_threshold
        return value > self.hard_reject_threshold

    def is_soft_band(self, value: Optional[float]) -> bool:
        """Between hard-reject and accept — verifier gets the final call."""
        if value is None:
            return False
        return (not self.is_accept(value)) and (not self.is_hard_reject(value))

    # -- display helpers --------------------------------------------------

    def describe(self) -> str:
        """One-line human-readable summary for logs and prompts."""
        cmp = "≥" if self.direction == "higher_is_better" else "≤"
        return (
            f"{self.metric} {cmp} {self.accept_threshold:.3f} accepts; "
            f"hard-reject at {self.metric} "
            f"{'<' if self.direction == 'higher_is_better' else '>'} "
            f"{self.hard_reject_threshold:.3f}"
        )

    def short_label(self) -> str:
        """Just the metric name, used in log lines like '✅ Fit approved (...)'."""
        return self.metric.replace("_", " ").upper() if self.metric == "r_squared" else self.metric

    # -- builders ---------------------------------------------------------

    def with_accept_threshold(self, value: float) -> "QualityGate":
        """Return a copy with a different accept_threshold; preserves direction.

        Adjusts hard_reject_threshold proportionally to keep the soft band
        width relative to the original gate.
        """
        soft_width = abs(self.accept_threshold - self.hard_reject_threshold)
        if self.direction == "higher_is_better":
            return replace(self, accept_threshold=value,
                           hard_reject_threshold=value - soft_width)
        return replace(self, accept_threshold=value,
                       hard_reject_threshold=value + soft_width)


# The framework-level default — preserves the pre-existing R² ≥ 0.95
# behavior for every skill / call that doesn't override.
R_SQUARED_DEFAULT = QualityGate(
    metric="r_squared",
    accept_threshold=0.95,
    hard_reject_threshold=0.90,
    direction="higher_is_better",
)


def resolve_gate(
    *,
    call_override: Optional[Any] = None,
    agent_default: Optional[Any] = None,
    skill_meta: Optional[dict] = None,
    legacy_threshold: Optional[float] = None,
) -> QualityGate:
    """Resolve the effective gate from all four sources.

    Priority (highest first):
      1. ``call_override`` — explicit ``analyze(quality_gate=...)`` arg
      2. ``agent_default`` — ``CurveFittingAgent(quality_gate=...)`` arg
      3. ``skill_meta['quality_gate']`` — frontmatter block from the
         active skill (or the first loaded skill in a multi-skill set).
      4. ``legacy_threshold`` — when set, builds a QualityGate(metric=
         'r_squared', accept_threshold=value). This handles the
         pre-existing ``r2_threshold`` kwarg on analyze() / __init__.
      5. :data:`R_SQUARED_DEFAULT`.

    Each layer can be either a :class:`QualityGate` instance or a plain
    dict (matching the dataclass fields). Dicts are coerced via
    :func:`from_mapping`. ``None`` at any layer falls through.
    """
    for source in (call_override, agent_default):
        gate = _coerce(source)
        if gate is not None:
            return gate
    if skill_meta:
        gate = _coerce(skill_meta.get("quality_gate"))
        if gate is not None:
            return gate
    if legacy_threshold is not None:
        return R_SQUARED_DEFAULT.with_accept_threshold(float(legacy_threshold))
    return R_SQUARED_DEFAULT


def from_mapping(data: dict) -> QualityGate:
    """Construct a QualityGate from a frontmatter-style mapping."""
    return QualityGate(
        metric=str(data.get("metric", R_SQUARED_DEFAULT.metric)),
        accept_threshold=float(data.get("accept_threshold", R_SQUARED_DEFAULT.accept_threshold)),
        hard_reject_threshold=float(data.get(
            "hard_reject_threshold", R_SQUARED_DEFAULT.hard_reject_threshold
        )),
        direction=str(data.get("direction", R_SQUARED_DEFAULT.direction)),
    )


def _coerce(value: Any) -> Optional[QualityGate]:
    if value is None:
        return None
    if isinstance(value, QualityGate):
        return value
    if isinstance(value, dict):
        return from_mapping(value)
    raise TypeError(
        f"Expected QualityGate, dict, or None; got {type(value).__name__}"
    )
