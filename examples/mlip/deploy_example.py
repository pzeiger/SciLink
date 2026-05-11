"""
MLIPAgent deploy-pretrained example.

Runs the same code path twice (Cu bulk and MoS2) and writes everything
into a timestamped subdirectory so reruns don't clobber each other.

This script gracefully degrades:
  - With MACE installed (cluster): runs the full deploy_pretrained
    pipeline, including model selection and weight download/load.
  - Without MACE (local dev): exercises agent assembly, skill loading,
    and direct LAMMPS input generation with a placeholder model path.

Both paths produce a runnable LAMMPS input file you can inspect.

Usage:
  python examples/mlip/deploy_example.py
  python examples/mlip/deploy_example.py --systems cu mos2
  python examples/mlip/deploy_example.py --temperature 600 --pressure 1.0
"""

import argparse
import datetime
import json
import os
import sys
import tempfile
import textwrap
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io.lammpsdata import write_lammps_data


def cu_bulk_system():
    atoms = bulk("Cu", "fcc", a=3.615, cubic=True)
    return {
        "name": "cu_bulk",
        "atoms": atoms,
        "elements": ["Cu"],
        "research_goal": (
            "Equilibrate FCC Cu at 300 K and report a stable lattice "
            "constant for downstream surface-energy work."
        ),
    }


def mos2_monolayer():
    """A 2H-MoS2 monolayer with vacuum along z."""
    a = 3.16
    c_vac = 20.0
    cell = np.array([
        [a, 0.0, 0.0],
        [-a / 2, a * np.sqrt(3) / 2, 0.0],
        [0.0, 0.0, c_vac],
    ])
    positions = [
        (0.0, 0.0, c_vac / 2),
        (a / 2, a * np.sqrt(3) / 6, c_vac / 2 + 1.59),
        (a / 2, a * np.sqrt(3) / 6, c_vac / 2 - 1.59),
    ]
    atoms = Atoms("MoS2", positions=positions, cell=cell, pbc=[True, True, True])
    return {
        "name": "mos2_monolayer",
        "atoms": atoms,
        "elements": ["Mo", "S"],
        "research_goal": (
            "Verify mace-mp-0 reproduces the MoS2 monolayer lattice "
            "constant within 2% at room temperature."
        ),
    }


SYSTEMS = {"cu": cu_bulk_system, "mos2": mos2_monolayer}


def print_section(title):
    print()
    print("=" * 70)
    print(f"  {title}")
    print("=" * 70)


def write_data_file(atoms, out_dir):
    data_path = out_dir / "system.data"
    elements = sorted(set(atoms.get_chemical_symbols()))
    write_lammps_data(
        str(data_path), atoms, atom_style="atomic",
        specorder=elements, masses=True,
    )
    return data_path


def report_skill_context(agent):
    print(f"  skill loaded:    {agent.skill_name}")
    if agent.skill_sections is None:
        print("  WARNING: skill_sections is None — agent will get empty LLM context")
        return
    sections = [
        k for k in agent.skill_sections
        if k not in ("name", "meta", "extras") and agent.skill_sections.get(k)
    ]
    print(f"  populated:       {sections}")
    validation = agent.skill_sections.get("validation", "")
    print(f"  validation len:  {len(validation)} chars")


def check_mace_availability():
    from scilink.skills._shared import mlip_tools
    backends = mlip_tools.check_backends()
    mace_info = backends.get("mace", {})
    return mace_info.get("available", False), mace_info


def run_with_mace(
    agent, system_info_dict, out_dir, temperature, pressure,
    runner, structure_file, n_steps, device,
):
    print(f"  MACE available — running deploy_pretrained (runner={runner}).")
    if runner == "lammps":
        sim_params = {
            "timestep": 0.5,            # ps (LAMMPS metal units)
            "temperature": temperature,
            "pressure": pressure,
        }
        kwargs = dict(runner="lammps")
    else:  # ase
        sim_params = {
            "timestep": 1.0,            # fs (ASE convention)
            "temperature": temperature,
            "pressure": pressure,
            "n_steps": n_steps,
            "device": device,
        }
        kwargs = dict(runner="ase", structure_file=str(structure_file))

    return agent.deploy_pretrained(
        system_info=system_info_dict,
        research_goal=system_info_dict["research_goal"],
        simulation_params=sim_params,
        **kwargs,
    )


def run_without_mace(
    agent, system, out_dir, temperature, pressure,
    runner, structure_file, n_steps, device,
):
    """
    Demonstrate the orchestration without MACE installed: drive
    skill context lookup + direct script/input generation with a
    placeholder model path. Documents what the cluster run would do.
    """
    print(f"  MACE not installed locally — exercising deterministic")
    print(f"  pieces only (skill context + {runner} script generation).")

    agent._load_backend_skill("mace")
    planning = agent._get_skill_context(section="planning")

    organic_elements = {"C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"}
    sys_elements = set(system["elements"])
    is_organic_only = sys_elements and sys_elements.issubset(organic_elements)

    if is_organic_only and "mace-off23" in planning:
        chosen = "mace-off23"
        rationale = f"all elements {sorted(sys_elements)} are organic"
    elif "mace-mp-0" in planning:
        chosen = "mace-mp-0"
        rationale = (
            f"system contains inorganic elements ({sorted(sys_elements)}) "
            f"→ mace-mp-0 per MACE skill rules"
        )
    else:
        chosen = "mace-mp-0"
        rationale = "default — MACE backend skill did not load"
    print(f"  pretrained pick: {chosen}")
    print(f"  rationale:       {rationale}")

    from scilink.skills._shared import mlip_tools
    if runner == "lammps":
        out_path = mlip_tools.generate_lammps_input(
            backend="mace",
            model_file="/CLUSTER_PATH/to/mace-mp-0.model",
            elements=system["elements"],
            working_dir=str(out_dir),
            timestep=0.5,
            temperature=temperature,
            pressure=pressure,
        )
        return {
            "backend": "mace", "model_name": chosen,
            "elements": system["elements"], "runner": "lammps",
            "lammps_input": out_path,
            "note": "PLACEHOLDER MODEL PATH — update on the cluster",
        }
    else:  # ase
        out_path = mlip_tools.generate_ase_script(
            backend="mace",
            model_name=chosen,
            elements=system["elements"],
            working_dir=str(out_dir),
            structure_file=Path(structure_file).name,
            timestep=1.0,
            temperature=temperature,
            pressure=pressure,
            n_steps=n_steps,
            device=device,
        )
        return {
            "backend": "mace", "model_name": chosen,
            "elements": system["elements"], "runner": "ase",
            "ase_script": out_path,
            "note": "MACE not installed locally — script needs MACE+torch to run",
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--systems", nargs="+", default=["cu", "mos2"],
        choices=list(SYSTEMS.keys()),
    )
    parser.add_argument("--temperature", type=float, default=300.0)
    parser.add_argument("--pressure", type=float, default=None,
                        help="If set, NPT; otherwise NVT")
    parser.add_argument(
        "--api-key", default=os.environ.get("SCILINK_API_KEY", ""),
        help="Falls back to SCILINK_API_KEY env var",
    )
    parser.add_argument("--model-name", default="gemini-2.5-pro")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument(
        "--runner", choices=["lammps", "ase"], default="lammps",
        help="lammps: write a LAMMPS+MACE input file. "
             "ase: write a runnable Python MD script using mace-torch.",
    )
    parser.add_argument(
        "--n-steps", type=int, default=200,
        help="ASE only: number of MD steps in the generated script.",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="ASE only: device the generated script will run on.",
    )
    parser.add_argument(
        "--run", action="store_true",
        help="ASE only: after generation, also execute the script "
             "in-process (requires MACE installed). Useful for "
             "single-command end-to-end demos on a GPU node.",
    )
    args = parser.parse_args()

    mace_available, mace_info = check_mace_availability()
    print_section("ENVIRONMENT")
    print(f"  MACE installed:  {mace_available}")
    if mace_info.get("version"):
        print(f"  MACE version:    {mace_info['version']}")
    if mace_available:
        n_pretrained = len(mace_info.get("pretrained", []))
        print(f"  pretrained avail: {n_pretrained}")
    if not args.api_key:
        if mace_available:
            print("  WARNING: no SCILINK_API_KEY — LLM model selection will fail")
        else:
            print("  no SCILINK_API_KEY (fine for local-only walkthrough)")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_out = Path(
        args.out_dir or Path(__file__).parent / f"mlip_run_{timestamp}"
    ).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    print(f"  output dir:      {base_out}")

    from scilink.agents.sim_agents.mlip_agent import MLIPAgent

    summary = {"systems": {}, "mace_available": mace_available}

    for system_key in args.systems:
        system = SYSTEMS[system_key]()
        print_section(f"SYSTEM: {system['name']}  ({len(system['atoms'])} atoms)")

        system_dir = base_out / system["name"]
        system_dir.mkdir(exist_ok=True)
        data_path = write_data_file(system["atoms"], system_dir)
        print(f"  data file:       {data_path}")
        print(f"  elements:        {system['elements']}")
        print(f"  research goal:   {system['research_goal']}")

        agent = MLIPAgent(
            working_dir=str(system_dir),
            api_key=args.api_key or "sk-no-llm-needed",
            model_name=args.model_name,
        )
        report_skill_context(agent)

        if mace_available and args.api_key:
            system_info = {
                "elements": {e: system["atoms"].get_chemical_symbols().count(e)
                             for e in system["elements"]},
                "n_atoms": len(system["atoms"]),
            }
            try:
                result = run_with_mace(
                    agent, system_info, system_dir,
                    args.temperature, args.pressure,
                    args.runner, data_path,
                    args.n_steps, args.device,
                )
            except Exception as exc:
                print(f"  deploy_pretrained failed: {exc}")
                result = run_without_mace(
                    agent, system, system_dir,
                    args.temperature, args.pressure,
                    args.runner, data_path,
                    args.n_steps, args.device,
                )
        else:
            result = run_without_mace(
                agent, system, system_dir,
                args.temperature, args.pressure,
                args.runner, data_path,
                args.n_steps, args.device,
            )

        output_file = result.get("lammps_input") or result.get("ase_script")
        label = "LAMMPS input" if args.runner == "lammps" else "ASE MD script"
        print()
        print(f"  --- generated {label} (head) ---")
        with open(output_file) as f:
            head = "".join(f.readlines()[:14])
        print(textwrap.indent(head, "    "))

        if args.run and args.runner == "ase" and mace_available:
            print()
            print("  --- executing ASE MD script in-process ---")
            import subprocess
            r = subprocess.run(
                [sys.executable, output_file],
                cwd=str(system_dir),
                env={**os.environ, "MACE_DEVICE": args.device},
            )
            print(f"  script exit code: {r.returncode}")
            for artifact in ("thermo.log", "traj.traj"):
                p = system_dir / artifact
                if p.exists():
                    print(f"  produced: {p} ({p.stat().st_size} bytes)")
        elif args.run and args.runner == "lammps":
            print("  --run is ASE-only; for LAMMPS runs, submit `lmp -in in.lammps` separately.")
        elif args.run and not mace_available:
            print("  --run skipped: MACE not installed.")

        summary["systems"][system["name"]] = {
            "data_file": str(data_path),
            "runner": args.runner,
            "output_file": output_file,
            "backend": result.get("backend"),
            "model_name": result.get("model_name"),
        }

    summary_path = base_out / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print_section("DONE")
    print(f"  summary: {summary_path}")
    if not mace_available:
        print()
        print("  Next step: copy this directory to a node with MACE installed")
        print("  and rerun with the same command — the script will detect MACE")
        print("  and run deploy_pretrained end-to-end.")


if __name__ == "__main__":
    main()
