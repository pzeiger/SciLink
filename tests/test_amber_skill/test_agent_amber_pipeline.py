"""
Full integration test: ForceFieldAgent + AMBER skill + AmberTools.

Requires:
  - AmberTools (antechamber, tleap, parmchk2)
  - ParmEd
  - LLM API key (set SCILINK_API_KEY or similar)

Run with: pytest test_agent_amber_pipeline.py -v -s
"""

import pytest
import os
import json

from scilink.tools.amber_tools import check_amber_tools

_tools = check_amber_tools()
requires_full_stack = pytest.mark.skipif(
    not _tools["available"],
    reason=f"Missing: {_tools.get('missing', [])}"
)

# Also skip if no API key
_has_api_key = bool(
    os.environ.get("SCILINK_API_KEY") or
    os.environ.get("OPENAI_API_KEY") or
    os.environ.get("GOOGLE_API_KEY")
)
requires_api_key = pytest.mark.skipif(
    not _has_api_key,
    reason="No LLM API key found in environment"
)


@requires_full_stack
@requires_api_key
class TestAgentAmberPipeline:

    @pytest.fixture
    def alanine_pdb(self, tmp_path):
        """Write a simple alanine dipeptide PDB."""
        pdb = """ATOM      1  N   ALA A   1       1.000   1.000   1.000  1.00  0.00           N
ATOM      2  CA  ALA A   1       2.450   1.000   1.000  1.00  0.00           C
ATOM      3  C   ALA A   1       3.000   2.420   1.000  1.00  0.00           C
ATOM      4  O   ALA A   1       2.220   3.390   1.000  1.00  0.00           O
ATOM      5  CB  ALA A   1       3.000   0.320  -0.260  1.00  0.00           C
ATOM      6  H   ALA A   1       0.600   0.600   1.850  1.00  0.00           H
ATOM      7  HA  ALA A   1       2.850   0.400   1.850  1.00  0.00           H
ATOM      8  HB1 ALA A   1       4.090   0.320  -0.260  1.00  0.00           H
ATOM      9  HB2 ALA A   1       2.600  -0.700  -0.260  1.00  0.00           H
ATOM     10  HB3 ALA A   1       2.600   0.850  -1.100  1.00  0.00           H
END
"""
        path = str(tmp_path / "alanine.pdb")
        with open(path, "w") as f:
            f.write(pdb)
        return path

    def test_select_force_field_loads_skill(self, alanine_pdb, tmp_path):
        """select_force_field should auto-load the amber skill for a protein."""
        from scilink.agents.force_field.force_field_agent import ForceFieldAgent

        agent = ForceFieldAgent(
            working_dir=str(tmp_path / "ff_work"),
            skill="amber",
        )

        selection = agent.select_force_field(
            pdb_file=alanine_pdb,
            research_goal="Study protein folding dynamics",
        )

        assert agent.skill_name == "amber"
        assert "force_field" in selection
        assert selection.get("skill_used") == "amber"

        ff = selection["force_field"]["force_field"].lower()
        # For a protein, it should select an AMBER FF
        assert any(kw in ff for kw in ["amber", "ff14sb", "ff19sb", "gaff"])

    def test_complete_parameterization_amber_pipeline(self, alanine_pdb, tmp_path):
        """Full complete_parameterization should use AMBER pipeline."""
        from scilink.agents.force_field.force_field_agent import ForceFieldAgent

        working_dir = str(tmp_path / "complete_test")
        agent = ForceFieldAgent(
            working_dir=working_dir,
            skill="amber",
        )

        # Create a dummy data file (the pipeline may replace it)
        dummy_data = os.path.join(working_dir, "input.data")
        with open(dummy_data, "w") as f:
            f.write("LAMMPS data\n0 atoms\n")

        results = agent.complete_parameterization(
            pdb_file=alanine_pdb,
            data_file=dummy_data,
            research_goal="Study protein stability",
        )

        assert results["status"] == "success"
        assert results.get("skill_used") == "amber"

        # Check that output files exist
        output_files = results.get("output_files", {})
        if "charged_data_file" in output_files:
            assert os.path.exists(output_files["charged_data_file"])

        # Check for errors
        assert len(results.get("errors", [])) == 0
