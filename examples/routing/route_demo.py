"""
End-to-end demo of the SimulationRouter.

Walks the router through three contrasting user goals and shows how
the (scale, engine) decision changes with the physics. Uses a fake
``AvailableSoftware`` so it runs anywhere without needing real VASP /
LAMMPS / MACE installations; only an LLM key is required.

Run:
    export SCILINK_API_KEY=...        # or GOOGLE_API_KEY, OPENAI_API_KEY, etc.
    python examples/routing/route_demo.py

Or pick a single scenario:
    python examples/routing/route_demo.py --scenario slab
    python examples/routing/route_demo.py --scenario md
    python examples/routing/route_demo.py --scenario mlip
"""

import argparse
import os
import sys
import textwrap
from typing import Optional

from scilink.agents.sim_agents.simulation_router import (
    SimulationRouter,
    discover_scale_agents,
)
from scilink.utils.available_software import AvailableSoftware


SCENARIOS = {
    "slab": {
        "goal": (
            "Relax a Cu(111) slab with an adsorbed CO molecule and "
            "report the binding energy in the atop configuration."
        ),
        "system": "metallic surface slab, 16 Cu atoms + 1 CO; periodic in plane",
        "expect_hint": (
            "Should route to periodic_dft + vasp — small metallic slab "
            "with adsorbate, ~17 atoms, accuracy matters for binding."
        ),
    },
    "md": {
        "goal": (
            "Run a 100 ns NPT simulation of a fully solvated lysozyme "
            "protein in water with physiological ions and report RMSD "
            "vs the crystal structure."
        ),
        "system": (
            "solvated protein, ~100k atoms total, AMBER FF14SB available "
            "for the protein, TIP3P for water"
        ),
        "expect_hint": (
            "Should route to molecular_dynamics + lammps — large "
            "biomolecular system, long timescale, well-parameterized."
        ),
    },
    "mlip": {
        "goal": (
            "Equilibrate a 500-atom amorphous a-Si:H sample at 300 K and "
            "characterize the bonding-environment distribution. Need "
            "near-DFT accuracy on bond statistics; classical FF for "
            "Si:H aren't reliable enough."
        ),
        "system": (
            "500 amorphous-Si:H atoms, periodic, no good classical FF, "
            "DFT too slow for ensemble averaging"
        ),
        "expect_hint": (
            "Should route to machine_learning_potentials + mace — "
            "system size beyond DFT, accuracy beyond classical FF; "
            "perfect MLIP fit."
        ),
    },
}


def build_fake_available_software() -> AvailableSoftware:
    """Pretend the user has VASP, LAMMPS, and MACE installed.

    Keeps the demo independent of the host's actual installs.
    """
    avail = AvailableSoftware()
    avail.set("periodic_dft", "vasp", True,
              source="demo-fake", user_confirmed=True)
    avail.set("molecular_dynamics", "lammps", True,
              source="demo-fake", user_confirmed=True)
    avail.set("machine_learning_potentials", "mace", True,
              source="demo-fake", user_confirmed=True)
    return avail


def build_model(api_key: str, model_name: str):
    """Construct an LLM model client the router can call."""
    from scilink.wrappers.litellm_wrapper import LiteLLMGenerativeModel
    return LiteLLMGenerativeModel(model=model_name, api_key=api_key)


def print_banner(title: str) -> None:
    print()
    print("=" * 72)
    print(f"  {title}")
    print("=" * 72)


def print_candidate_grid(router: SimulationRouter) -> None:
    print_banner("WHAT THE ROUTER SEES")
    scale_agents = discover_scale_agents()
    print("Agent classes available in this build:")
    for scale, info in sorted(scale_agents.items()):
        print(f"  - {scale:35s} -> {info['agent_class'].__name__}")
        print(f"    supported_software: {info['supported']}")
    print()
    print("User-available software (faked for this demo):")
    for scale in sorted(router.available_software.domains()):
        engines = router.available_software.list_available(domain=scale)
        if engines:
            print(f"  - {scale:35s} -> {engines}")
    print()
    candidates = router.candidate_engines()
    print("CANDIDATES (agent_supports ∩ user_available):")
    if not candidates:
        print("  (empty — would short-circuit before LLM call)")
    for scale, engines in sorted(candidates.items()):
        print(f"  - {scale:35s} -> {engines}")


def run_scenario(router: SimulationRouter, name: str) -> int:
    scenario = SCENARIOS[name]
    print_banner(f"SCENARIO: {name}")
    print("User goal:")
    print(textwrap.indent(textwrap.fill(scenario["goal"], 65), "  "))
    print()
    print("System description:")
    print(textwrap.indent(textwrap.fill(scenario["system"], 65), "  "))
    print()
    print("Expected:")
    print(textwrap.indent(textwrap.fill(scenario["expect_hint"], 65), "  "))
    print()
    print("Calling router (this hits the LLM)…")

    decision = router.route(
        user_goal=scenario["goal"],
        system_description=scenario["system"],
    )

    print()
    print("ROUTING DECISION:")
    if decision.get("scale") is None:
        print(f"  ERROR: {decision.get('error')}")
        return 1
    print(f"  scale  : {decision['scale']}")
    print(f"  engine : {decision['engine']}")
    print(f"  reasoning:")
    print(textwrap.indent(
        textwrap.fill(decision.get("reasoning", ""), 65),
        "    ",
    ))
    alts = decision.get("alternatives") or []
    if alts:
        print("  alternatives:")
        for a in alts:
            tradeoff = a.get("tradeoff", "")
            print(f"    - {a.get('scale')}/{a.get('engine')}: {tradeoff}")
    print()
    if decision["engine"] == "vasp":
        print("  DISPATCH PATH: orchestrator can run this end-to-end —")
        print("    generate_structure -> generate_vasp_inputs -> validate_incar")
        print("    -> submit_vasp_job (when HPC connection active).")
    else:
        print("  DISPATCH PATH: routing matched, but the orchestrator")
        print(f"    doesn't have integrated dispatch tools for {decision['engine']}")
        print(f"    yet. For now, point the user at the corresponding agent:")
        print(f"      {decision['scale']} -> MDSimulationAgent / MLIPAgent")
        print("    The router's job is to make this routing decision visible.")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scenario", choices=list(SCENARIOS) + ["all"],
        default="all",
        help="Which scenario to run. Default: all three.",
    )
    parser.add_argument(
        "--api-key",
        default=(
            os.environ.get("SCILINK_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        ),
        help="LLM API key. Defaults to SCILINK_API_KEY / GOOGLE_API_KEY / etc.",
    )
    parser.add_argument(
        "--model-name", default="gemini-2.5-pro",
        help="LiteLLM model identifier (default: gemini-2.5-pro).",
    )
    parser.add_argument(
        "--show-candidates-only", action="store_true",
        help=(
            "Print the candidate grid (agent_supports ∩ user_available) "
            "and exit without calling the LLM."
        ),
    )
    args = parser.parse_args()

    avail = build_fake_available_software()

    if args.show_candidates_only:
        # Still construct a router but don't make LLM calls
        from unittest.mock import MagicMock
        router = SimulationRouter(model=MagicMock(), available_software=avail)
        print_candidate_grid(router)
        return 0

    if not args.api_key:
        print(
            "ERROR: no LLM API key. Set SCILINK_API_KEY (or "
            "GOOGLE_API_KEY / OPENAI_API_KEY / ANTHROPIC_API_KEY), or "
            "pass --api-key, or use --show-candidates-only to skip the "
            "LLM call.",
            file=sys.stderr,
        )
        return 2

    model = build_model(args.api_key, args.model_name)
    router = SimulationRouter(model=model, available_software=avail)

    print_candidate_grid(router)

    scenarios_to_run = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    rc = 0
    for name in scenarios_to_run:
        rc |= run_scenario(router, name)
    print()
    print("=" * 72)
    print("  Demo complete.")
    print("=" * 72)
    return rc


if __name__ == "__main__":
    sys.exit(main())
