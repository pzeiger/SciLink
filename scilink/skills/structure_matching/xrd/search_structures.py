"""``search_structures`` tool — multi-backend structure-database query.

Dispatches a :class:`QuerySpec` across the available backends, dedupes
across sources, ranks, materializes CIF files into ``output_dir``, and
returns a JSON-serializable dict.

The tool is callable from any analysis script (the skill registry exposes
it once ``structure_matching/xrd`` is in ``active_skills``). It also
prints a ``DB_MATCHES_JSON:`` line on stdout so the curve-fitting
stdout-parser (extended in commit 7) can lift the candidate list into
``fit_results['db_matches']`` for downstream synthesis and reporting.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Optional

from .._backends import (
    LocalCIFBackend,
    MaterialsProjectBackend,
    QuerySpec,
    StructureBackend,
    StructureCandidate,
)
from .._backends.cod import CODBackend
from ..._shared._spec import ToolSpec

_logger = logging.getLogger(__name__)

# Source rank order for dedup: when the same structure appears in multiple
# backends, prefer the more authoritative source.
_SOURCE_PREFERENCE = {"mp": 0, "cod": 1, "local": 2}

_BACKEND_FACTORY = {
    "mp": MaterialsProjectBackend,
    "local": LocalCIFBackend,
    "cod": CODBackend,
}


TOOL_SPEC = ToolSpec(
    name="search_structures",
    description=(
        "Query crystal-structure databases for candidate structures matching a "
        "chemistry + symmetry spec. Dispatches across Materials Project, local "
        "CIF, and COD; dedupes across sources; returns a ranked list of "
        "candidates with materialized CIF paths."
    ),
    import_line="from scilink.skills.structure_matching.xrd.search_structures import search_structures",
    signature="search_structures(query: dict, sources: list[str] | None = None, output_dir: str = './candidates') -> dict",
    parameters={
        "query": {
            "type": "dict",
            "description": (
                "Spec: {chemistry: list[str] (required), space_group_hints?: "
                "list[int], lattice_param_ranges?: {a: (min,max), ...}, "
                "max_e_above_hull?: float, top_n?: int (default 10)}"
            ),
        },
        "sources": {
            "type": "list",
            "description": (
                "Backend names in priority order: 'mp', 'local', 'cod'. None "
                "auto-detects whichever backends are available in the current "
                "environment (MP_API_KEY set → 'mp'; SCILINK_LOCAL_CIF_DIR "
                "exists → 'local')."
            ),
        },
        "output_dir": {
            "type": "str",
            "description": (
                "Directory where materialized CIF files are written "
                "(created if absent). Default './candidates'."
            ),
        },
    },
    required=["query"],
    returns=(
        "dict with 'candidates' (list of {id, source, formula, space_group, "
        "structure_path, rank_score, metadata}), 'sources_queried' (list of "
        "backend names), 'warnings' (list of human-readable strings)."
    ),
    when_to_use=(
        "Whenever a candidate-structure list is needed — pre-fit (chemistry "
        "hypothesized up front) or post-fit (extracted lattice/symmetry "
        "filters the DB). Combine with simulate_xrd_pattern and "
        "score_xrd_match for end-to-end identification."
    ),
)


def search_structures(
    query: dict,
    sources: Optional[list[str]] = None,
    output_dir: str = "./candidates",
) -> dict[str, Any]:
    """Query structure databases. See ``TOOL_SPEC`` for full contract."""
    spec = _coerce_query_spec(query)
    warnings: list[str] = []

    backends = _build_backends(sources, warnings)
    if not backends:
        result = {"candidates": [], "sources_queried": [], "warnings": warnings or ["No backends available"]}
        _emit_db_matches_marker(result)
        return result

    accumulated: list[StructureCandidate] = []
    sources_queried: list[str] = []
    for backend in backends:
        try:
            results = backend.query(spec)
        except Exception as e:
            warnings.append(f"{backend.name} backend raised: {e}")
            _logger.warning("Backend %s failed: %s", backend.name, e)
            continue
        sources_queried.append(backend.name)
        accumulated.extend(results)

    deduped = _dedupe(accumulated)
    deduped.sort(key=lambda c: -c.rank_score)
    deduped = deduped[:spec.top_n]

    _materialize_cifs(deduped, Path(output_dir), warnings)

    result = {
        "candidates": [_candidate_to_dict(c) for c in deduped],
        "sources_queried": sources_queried,
        "warnings": warnings,
    }
    _emit_db_matches_marker(result)
    return result


# --- Helpers ------------------------------------------------------------------

def _coerce_query_spec(query: dict) -> QuerySpec:
    chemistry = query.get("chemistry")
    if not chemistry:
        raise ValueError("query.chemistry is required (non-empty list of element symbols)")
    lattice_ranges = query.get("lattice_param_ranges")
    if lattice_ranges:
        # JSON serialization turns tuples into lists; normalize back.
        lattice_ranges = {k: tuple(v) for k, v in lattice_ranges.items()}
    return QuerySpec(
        chemistry=list(chemistry),
        space_group_hints=query.get("space_group_hints"),
        lattice_param_ranges=lattice_ranges,
        max_e_above_hull=query.get("max_e_above_hull"),
        top_n=int(query.get("top_n", 10)),
    )


def _build_backends(
    sources: Optional[list[str]],
    warnings: list[str],
) -> list[StructureBackend]:
    """Construct backend instances for ``sources`` (or auto-detect)."""
    if sources is None:
        sources = list(_BACKEND_FACTORY.keys())

    backends: list[StructureBackend] = []
    for name in sources:
        factory = _BACKEND_FACTORY.get(name)
        if factory is None:
            warnings.append(f"Unknown source '{name}' — skipping")
            continue
        try:
            b = factory()
        except Exception as e:
            warnings.append(f"Failed to construct {name} backend: {e}")
            continue
        if not b.is_available():
            warnings.append(f"Backend '{name}' not available (missing key/dir/dependency)")
            continue
        backends.append(b)
    return backends


def _dedupe(candidates: Iterable[StructureCandidate]) -> list[StructureCandidate]:
    """Collapse duplicates across backends, preferring the more authoritative source."""
    bucket: dict[tuple, StructureCandidate] = {}
    for cand in candidates:
        key = (cand.formula, cand.space_group)
        existing = bucket.get(key)
        if existing is None:
            bucket[key] = cand
            continue
        if _SOURCE_PREFERENCE.get(cand.source, 99) < _SOURCE_PREFERENCE.get(existing.source, 99):
            bucket[key] = cand
    return list(bucket.values())


def _materialize_cifs(
    candidates: list[StructureCandidate],
    output_dir: Path,
    warnings: list[str],
) -> None:
    """Write CIFs to ``output_dir`` for candidates that don't already have a path on disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for cand in candidates:
        if cand.structure_path and Path(cand.structure_path).is_file():
            continue
        struct = cand.metadata.get("_structure")
        if struct is None:
            warnings.append(f"No structure object for {cand.source}:{cand.id} — cannot materialize CIF")
            continue
        path = output_dir / f"{cand.source}_{cand.id}.cif"
        try:
            with open(path, "w") as f:
                f.write(struct.to(fmt="cif"))
            cand.structure_path = str(path)
        except Exception as e:
            warnings.append(f"Failed to write CIF for {cand.source}:{cand.id}: {e}")


def _candidate_to_dict(cand: StructureCandidate) -> dict[str, Any]:
    """JSON-safe dict (drops the in-memory pymatgen Structure)."""
    d = asdict(cand)
    d["metadata"] = {k: v for k, v in d["metadata"].items() if not k.startswith("_")}
    return d


def _emit_db_matches_marker(result: dict[str, Any]) -> None:
    """Emit a ``DB_MATCHES_JSON:`` line on stdout for the framework parser."""
    safe = {
        "candidates": result["candidates"],
        "sources_queried": result["sources_queried"],
        "warnings": result["warnings"],
    }
    try:
        line = "DB_MATCHES_JSON: " + json.dumps(safe, default=str)
    except Exception as e:
        _logger.debug("Skipping DB_MATCHES_JSON emit: %s", e)
        return
    print(line, file=sys.stdout, flush=True)
