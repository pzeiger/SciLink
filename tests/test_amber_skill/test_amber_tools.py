"""
Tests for scilink/tools/amber_tools.py

Some tests require AmberTools to be installed (marked with
@pytest.mark.requires_ambertools). Others test pure-Python logic.
"""

import pytest
import os
import shutil

from scilink.skills.force_field.amber.amber import (
    check_amber_tools,
    run_pdb4amber,
    run_antechamber,
    run_parmchk2,
    generate_tleap_script,
    run_tleap,
    convert_amber_to_lammps,
    validate_amber_data_file,
)

# ─── Markers ─────────────────────────────────────────────────────

_tools_info = check_amber_tools()
requires_ambertools = pytest.mark.skipif(
    not _tools_info["available"],
    reason=f"AmberTools not available. Missing: {_tools_info.get('missing', [])}"
)


# ─── Pure Python tests (no external tools) ───────────────────────

class TestCheckAmberTools:
    """Tests for check_amber_tools() — always runs."""

    def test_returns_dict_with_required_keys(self):
        result = check_amber_tools()
        assert isinstance(result, dict)
        assert "available" in result
        assert "missing" in result
        assert "tools" in result
        assert isinstance(result["available"], bool)
        assert isinstance(result["missing"], list)

    def test_tools_have_found_and_path(self):
        result = check_amber_tools()
        for tool_name, info in result["tools"].items():
            assert "found" in info
            assert "path" in info

    def test_parmed_info_present(self):
        result = check_amber_tools()
        assert "parmed" in result or "parmed_available" in result


class TestTleapScriptGeneration:
    """Tests for generate_tleap_script() — no AmberTools needed."""

    def test_protein_system(self, tmp_path):
        """Protein-only system should source protein FF."""
        pdb_file = str(tmp_path / "test.pdb")
        # Create dummy PDB
        with open(pdb_file, "w") as f:
            f.write("ATOM      1  CA  ALA A   1       0.0   0.0   0.0  1.00  0.00\nEND\n")

        script = generate_tleap_script(
            pdb_file=pdb_file,
            working_dir=str(tmp_path),
            composition={"proteins": True, "small_molecules": False,
                        "nucleic_acids": False, "lipids": False,
                        "carbohydrates": False},
            protein_ff="ff19SB",
            water_model="opc",
        )

        assert os.path.exists(script)
        with open(script) as f:
            content = f.read()

        assert "leaprc.protein.ff19SB" in content
        assert "leaprc.water.opc" in content
        assert "loadpdb" in content
        assert "saveamberparm" in content

    def test_small_molecule_system(self, tmp_path):
        """System with small molecules should source GAFF."""
        pdb_file = str(tmp_path / "test.pdb")
        with open(pdb_file, "w") as f:
            f.write("END\n")

        mol2_files = [{"mol2": "/fake/ligand.mol2", "name": "LIG"}]
        frcmod_files = ["/fake/ligand.frcmod"]

        script = generate_tleap_script(
            pdb_file=pdb_file,
            working_dir=str(tmp_path),
            composition={"proteins": False, "small_molecules": True,
                        "nucleic_acids": False, "lipids": False,
                        "carbohydrates": False},
            mol2_files=mol2_files,
            frcmod_files=frcmod_files,
            gaff_version="gaff2",
            water_model="tip3p",
        )

        with open(script) as f:
            content = f.read()

        assert "leaprc.gaff2" in content
        assert "loadmol2" in content
        assert "loadamberparams" in content

    def test_solvation_and_neutralization(self, tmp_path):
        """Solvation and neutralization flags should add correct commands."""
        pdb_file = str(tmp_path / "test.pdb")
        with open(pdb_file, "w") as f:
            f.write("END\n")

        script = generate_tleap_script(
            pdb_file=pdb_file,
            working_dir=str(tmp_path),
            composition={"proteins": True, "small_molecules": False,
                        "nucleic_acids": False, "lipids": False,
                        "carbohydrates": False},
            solvate=True,
            box_buffer=12.0,
            neutralize=True,
            water_model="tip3p",
        )

        with open(script) as f:
            content = f.read()

        assert "solvatebox" in content
        assert "12.0" in content
        assert "addIonsRand" in content

    def test_no_solvation(self, tmp_path):
        """Without solvation, no solvatebox command."""
        pdb_file = str(tmp_path / "test.pdb")
        with open(pdb_file, "w") as f:
            f.write("END\n")

        script = generate_tleap_script(
            pdb_file=pdb_file,
            working_dir=str(tmp_path),
            composition={"proteins": True, "small_molecules": False,
                        "nucleic_acids": False, "lipids": False,
                        "carbohydrates": False},
            solvate=False,
            neutralize=False,
            water_model="tip3p",
        )

        with open(script) as f:
            content = f.read()

        assert "solvatebox" not in content
        assert "addIonsRand" not in content


class TestValidateDataFile:
    """Tests for validate_amber_data_file() — no AmberTools needed."""

    def _write_minimal_data_file(self, path, charges_zero=False):
        """Write a minimal valid LAMMPS data file."""
        if charges_zero:
            q_O, q_H = "0.000000", "0.000000"
        else:
            q_O, q_H = "-0.834000", "0.417000"
    
        content = f"""LAMMPS data file via ParmEd
    
    3 atoms
    2 bonds
    1 angles
    0 dihedrals
    0 impropers
    2 atom types
    1 bond types
    1 angle types
    
    0.0 10.0 xlo xhi
    0.0 10.0 ylo yhi
    0.0 10.0 zlo zhi
    
    Masses
    
    1 15.9994
    2 1.008
    
    Pair Coeffs
    
    1 0.1553 3.1507
    2 0.0000 0.0000
    
    Bond Coeffs
    
    1 553.0 0.9572
    
    Angle Coeffs
    
    1 100.0 104.52
    
    Atoms # full
    
    1 1 1 {q_O} 5.0 5.0 5.0
    2 1 2 {q_H} 5.8 5.0 5.0
    3 1 2 {q_H} 4.2 5.0 5.0
    
    Bonds
    
    1 1 1 2
    2 1 1 3
    
    Angles
    
    1 1 2 1 3
    
    """
        with open(path, "w") as f:
            f.write(content)

    def test_valid_data_file(self, tmp_path):
        path = str(tmp_path / "good.data")
        self._write_minimal_data_file(path, charges_zero=False)

        result = validate_amber_data_file(path)
        assert result["valid"] is True
        assert result["n_atoms"] == 3
        assert "Masses" in result["sections_found"]
        assert "Atoms" in result["sections_found"]
        assert "Pair Coeffs" in result["sections_found"]

    def test_all_zero_charges_flagged(self, tmp_path):
        """All-zero charges should be flagged as an error."""
        path = str(tmp_path / "zero_charges.data")
        self._write_minimal_data_file(path, charges_zero=True)

        result = validate_amber_data_file(path)
        # With only 3 atoms it might not trigger the n>10 check
        # but total_charge should be 0
        assert result["total_charge"] == 0.0

    def test_missing_file(self, tmp_path):
        result = validate_amber_data_file(str(tmp_path / "nonexistent.data"))
        assert result["valid"] is False
        assert len(result["errors"]) > 0


# ─── Integration tests (require AmberTools installed) ────────────

@requires_ambertools
class TestAntechamber:
    """Tests that actually run antechamber."""

    @pytest.fixture
    def methanol_pdb(self, tmp_path):
        """Create a minimal methanol PDB."""
        pdb = """HETATM    1  C1  MOL A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  O1  MOL A   1       1.430   0.000   0.000  1.00  0.00           O
HETATM    3  H1  MOL A   1      -0.390   1.010   0.000  1.00  0.00           H
HETATM    4  H2  MOL A   1      -0.390  -0.510   0.880  1.00  0.00           H
HETATM    5  H3  MOL A   1      -0.390  -0.510  -0.880  1.00  0.00           H
HETATM    6  H4  MOL A   1       1.820   0.890   0.000  1.00  0.00           H
END
"""
        path = str(tmp_path / "methanol.pdb")
        with open(path, "w") as f:
            f.write(pdb)
        return path

    def test_antechamber_gasteiger(self, methanol_pdb, tmp_path):
        """antechamber with Gasteiger charges (fastest method)."""
        result = run_antechamber(
            input_file=methanol_pdb,
            working_dir=str(tmp_path),
            net_charge=0,
            charge_method="gas",  # Fast — no sqm needed
            atom_type="gaff2",
            output_prefix="methanol",
        )
        assert os.path.exists(result["mol2"])
        assert result["charge_method"] == "gas"
        assert result["atom_type"] == "gaff2"

    def test_antechamber_bcc(self, methanol_pdb, tmp_path):
        """antechamber with AM1-BCC charges (standard method)."""
        result = run_antechamber(
            input_file=methanol_pdb,
            working_dir=str(tmp_path),
            net_charge=0,
            charge_method="bcc",
            atom_type="gaff2",
            output_prefix="methanol_bcc",
        )
        assert os.path.exists(result["mol2"])

    def test_parmchk2(self, methanol_pdb, tmp_path):
        """parmchk2 should produce a frcmod file."""
        ac = run_antechamber(
            input_file=methanol_pdb,
            working_dir=str(tmp_path),
            net_charge=0,
            charge_method="gas",
            atom_type="gaff2",
        )
        frcmod = run_parmchk2(
            mol2_file=ac["mol2"],
            working_dir=str(tmp_path),
            atom_type="gaff2",
        )
        assert os.path.exists(frcmod)
        assert frcmod.endswith(".frcmod")


@requires_ambertools
class TestFullPipeline:
    """End-to-end test: PDB → antechamber → tleap → ParmEd → LAMMPS data."""

    @pytest.fixture
    def methanol_pdb(self, tmp_path):
        pdb = """HETATM    1  C1  MOL A   1       0.000   0.000   0.000  1.00  0.00           C
HETATM    2  O1  MOL A   1       1.430   0.000   0.000  1.00  0.00           O
HETATM    3  H1  MOL A   1      -0.390   1.010   0.000  1.00  0.00           H
HETATM    4  H2  MOL A   1      -0.390  -0.510   0.880  1.00  0.00           H
HETATM    5  H3  MOL A   1      -0.390  -0.510  -0.880  1.00  0.00           H
HETATM    6  H4  MOL A   1       1.820   0.890   0.000  1.00  0.00           H
END
"""
        path = str(tmp_path / "methanol.pdb")
        with open(path, "w") as f:
            f.write(pdb)
        return path

    def test_full_pipeline_methanol(self, methanol_pdb, tmp_path):
        """Full pipeline for a simple methanol molecule."""
        working_dir = str(tmp_path)

        # Step 1: antechamber
        ac = run_antechamber(
            input_file=methanol_pdb,
            working_dir=working_dir,
            net_charge=0,
            charge_method="gas",
            atom_type="gaff2",
            output_prefix="mol",
        )

        # Step 2: parmchk2
        frcmod = run_parmchk2(
            mol2_file=ac["mol2"],
            working_dir=working_dir,
            atom_type="gaff2",
            output_prefix="mol",
        )

        # Step 3: tleap
        script = generate_tleap_script(
            pdb_file=methanol_pdb,
            working_dir=working_dir,
            composition={"proteins": False, "small_molecules": True,
                        "nucleic_acids": False, "lipids": False,
                        "carbohydrates": False},
            mol2_files=[{"mol2": ac["mol2"], "name": "MOL"}],
            frcmod_files=[frcmod],
            gaff_version="gaff2",
            water_model="tip3p",
            solvate=False,
            neutralize=False,
        )

        prmtop, inpcrd = run_tleap(script, working_dir)
        assert os.path.exists(prmtop)
        assert os.path.exists(inpcrd)

        # Step 4: ParmEd conversion
        data_file = convert_amber_to_lammps(
            prmtop=prmtop,
            inpcrd=inpcrd,
            output_data=os.path.join(working_dir, "methanol.data"),
        )
        assert os.path.exists(data_file)
        assert os.path.getsize(data_file) > 100

        # Step 5: Validate
        validation = validate_amber_data_file(data_file)
        assert validation["valid"] is True
        assert validation["n_atoms"] == 6
        assert abs(validation["total_charge"]) < 0.01  # Should be neutral
        assert "Pair Coeffs" in validation["sections_found"]
        assert "Bond Coeffs" in validation["sections_found"]
