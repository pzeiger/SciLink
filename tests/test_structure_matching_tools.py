"""Offline tests for the structure_matching/xrd tools.

Tools are pure Python and fully unit-testable without an LLM or network.
MP-related paths are mocked; local-only paths use a tmp CIF directory.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

pymatgen = pytest.importorskip("pymatgen")
from pymatgen.core import Lattice, Structure  # noqa: E402

from scilink.skills.structure_matching._backends import StructureCandidate
from scilink.skills.structure_matching.xrd.search_structures import (
    TOOL_SPEC,
    _candidate_to_dict,
    _dedupe,
    search_structures,
)


# --- TOOL_SPEC shape ----------------------------------------------------------

def test_tool_spec_renders_prompt_block():
    block = TOOL_SPEC.to_prompt()
    assert "search_structures" in block
    assert "query" in block
    assert "Returns" in block


# --- Fixture helpers ----------------------------------------------------------

def _silicon() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(5.43), ["Si"], [[0, 0, 0]])


def _diamond() -> Structure:
    return Structure.from_spacegroup("Fd-3m", Lattice.cubic(3.57), ["C"], [[0, 0, 0]])


def _write_cif(path: Path, structure: Structure) -> None:
    path.write_text(structure.to(fmt="cif"))


def _capture_stdout(func, *args, **kwargs):
    buf = io.StringIO()
    saved = sys.stdout
    sys.stdout = buf
    try:
        result = func(*args, **kwargs)
    finally:
        sys.stdout = saved
    return result, buf.getvalue()


# --- search_structures end-to-end (local only) --------------------------------

def test_search_local_only_returns_matching_candidates(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())
    _write_cif(cif_dir / "c.cif", _diamond())

    out_dir = tmp_path / "candidates"

    monkey_env = {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}
    with patch.dict("os.environ", monkey_env):
        result = search_structures(
            query={"chemistry": ["Si"]},
            sources=["local"],
            output_dir=str(out_dir),
        )

    assert result["sources_queried"] == ["local"]
    assert len(result["candidates"]) == 1
    cand = result["candidates"][0]
    assert cand["formula"] == "Si"
    assert cand["source"] == "local"
    assert Path(cand["structure_path"]).is_file()


def test_search_warns_when_no_backends_available(tmp_path):
    with patch.dict("os.environ", {}, clear=False):
        # Unset env vars that might enable a backend
        with patch.dict("os.environ", {
            "MP_API_KEY": "",
            "MATERIALS_PROJECT_API_KEY": "",
            "SCILINK_LOCAL_CIF_DIR": "",
        }):
            result = search_structures(
                query={"chemistry": ["Si"]},
                sources=["local"],
                output_dir=str(tmp_path / "out"),
            )
    assert result["candidates"] == []
    assert any("not available" in w for w in result["warnings"])


def test_search_unknown_source_is_warned(tmp_path):
    result = search_structures(
        query={"chemistry": ["Si"]},
        sources=["not_a_real_backend"],
        output_dir=str(tmp_path / "out"),
    )
    assert any("Unknown source" in w for w in result["warnings"])


def test_search_query_requires_chemistry(tmp_path):
    with pytest.raises(ValueError, match="chemistry"):
        search_structures(
            query={"top_n": 5},
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )


def test_search_emits_db_matches_json(tmp_path):
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si.cif", _silicon())

    with patch.dict("os.environ", {"SCILINK_LOCAL_CIF_DIR": str(cif_dir)}):
        result, captured = _capture_stdout(
            search_structures,
            query={"chemistry": ["Si"]},
            sources=["local"],
            output_dir=str(tmp_path / "out"),
        )

    assert "DB_MATCHES_JSON:" in captured
    json_line = [
        line[len("DB_MATCHES_JSON:"):].strip()
        for line in captured.splitlines()
        if line.startswith("DB_MATCHES_JSON:")
    ][0]
    parsed = json.loads(json_line)
    assert parsed["candidates"] == result["candidates"]


# --- Dedup --------------------------------------------------------------------

def test_dedup_prefers_mp_over_local():
    cands = [
        StructureCandidate(id="local_x", source="local", formula="Si", space_group="Fd-3m", rank_score=0.5),
        StructureCandidate(id="mp-149", source="mp", formula="Si", space_group="Fd-3m", rank_score=1.0),
    ]
    result = _dedupe(cands)
    assert len(result) == 1
    assert result[0].source == "mp"


def test_dedup_keeps_distinct_space_groups():
    cands = [
        StructureCandidate(id="mp-1", source="mp", formula="C", space_group="Fd-3m"),
        StructureCandidate(id="mp-2", source="mp", formula="C", space_group="P6_3/mmc"),
    ]
    assert len(_dedupe(cands)) == 2


def test_dedup_keeps_distinct_formulas():
    cands = [
        StructureCandidate(id="mp-1", source="mp", formula="TiO2"),
        StructureCandidate(id="mp-2", source="mp", formula="Ti2O3"),
    ]
    assert len(_dedupe(cands)) == 2


# --- Candidate serialization --------------------------------------------------

def test_candidate_to_dict_strips_private_metadata():
    cand = StructureCandidate(
        id="mp-1", source="mp", formula="Si",
        metadata={"energy_above_hull": 0.0, "_structure": "pretend-pymatgen-obj"},
    )
    d = _candidate_to_dict(cand)
    assert "_structure" not in d["metadata"]
    assert d["metadata"]["energy_above_hull"] == 0.0


# --- MP + local combined (mocked MP) ------------------------------------------

def _mp_record(material_id, formula, sg_symbol, sg_number, e_hull, structure):
    return SimpleNamespace(
        material_id=material_id,
        formula_pretty=formula,
        symmetry=SimpleNamespace(symbol=sg_symbol, number=sg_number),
        energy_above_hull=e_hull,
        structure=structure,
    )


@patch("scilink.skills.structure_matching._backends.materials_project.MPRester")
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_search_dedupes_mp_and_local_same_phase(mock_mprester, tmp_path):
    # Local CIF for Si
    cif_dir = tmp_path / "cifs"
    cif_dir.mkdir()
    _write_cif(cif_dir / "si_local.cif", _silicon())

    # Mock MP returning same Si (Fd-3m)
    si_struct = _silicon()
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = [
        _mp_record("mp-149", "Si", "Fd-3m", 227, 0.0, si_struct),
    ]

    with patch.dict("os.environ", {
        "MP_API_KEY": "fake",
        "SCILINK_LOCAL_CIF_DIR": str(cif_dir),
    }):
        result = search_structures(
            query={"chemistry": ["Si"]},
            sources=["mp", "local"],
            output_dir=str(tmp_path / "out"),
        )

    assert result["sources_queried"] == ["mp", "local"]
    assert len(result["candidates"]) == 1
    # Dedup keeps MP (preferred source)
    assert result["candidates"][0]["source"] == "mp"
    assert result["candidates"][0]["id"] == "mp-149"


@patch("scilink.skills.structure_matching._backends.materials_project.MPRester")
@patch(
    "scilink.skills.structure_matching._backends.materials_project.MP_API_AVAILABLE",
    True,
)
def test_search_truncates_to_top_n(mock_mprester, tmp_path):
    records = [
        _mp_record(f"mp-{i}", "TiO2", "Pbca", 61, 0.01 * i, _silicon())  # synthetic, rank by e_hull
        for i in range(20)
    ]
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = records

    # Dedup will collapse same-formula+sg entries, but each "mp-i" has the same
    # formula+sg so dedup collapses to 1. Use distinct sgs to test top_n.
    records = [
        _mp_record(f"mp-{i}", "TiO2", f"sg-{i}", 100 + i, 0.01 * i, _silicon())
        for i in range(20)
    ]
    mock_mprester.return_value.__enter__.return_value.materials.summary.search.return_value = records

    with patch.dict("os.environ", {"MP_API_KEY": "fake"}):
        result = search_structures(
            query={"chemistry": ["Ti", "O"], "top_n": 5},
            sources=["mp"],
            output_dir=str(tmp_path / "out"),
        )

    assert len(result["candidates"]) == 5
    # Should be the ones with lowest e_hull (best rank_score)
    assert result["candidates"][0]["id"] == "mp-0"
    assert result["candidates"][-1]["id"] == "mp-4"
