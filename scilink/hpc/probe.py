"""Probe a remote HPC system for available tools and paths."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scilink.hpc.connection import HPCConnection


@dataclass
class HPCEnvironment:
    """Cached snapshot of what's available on the remote system."""
    home: str = ""
    scratch: str = ""                         # $SCRATCH, $WORK, etc.
    lammps_binaries: list[str] = field(default_factory=list)
    lammps_modules: list[str] = field(default_factory=list)
    vasp_binaries: list[str] = field(default_factory=list)
    vasp_modules: list[str] = field(default_factory=list)
    container_runtimes: list[str] = field(default_factory=list)  # podman, singularity, …
    python_path: str = ""
    scheduler_name: str = ""


def probe_remote(conn: HPCConnection) -> HPCEnvironment:
    """Probe the remote host for available tools. ~5 SSH round-trips."""
    env = HPCEnvironment()

    # ── Home + scratch ────────────────────────────────────
    out, _, _ = conn.run(
        'echo "$HOME"; '
        'echo "${SCRATCH:-${WORK:-${CENTER_SCRATCH:-__none__}}}"',
        timeout=10,
    )
    lines = out.strip().splitlines()
    env.home = lines[0].strip() if lines else ""
    if len(lines) > 1 and lines[1].strip() != "__none__":
        env.scratch = lines[1].strip()

    # ── LAMMPS binaries ───────────────────────────────────
    candidates = [
        "lmp", "lmp_mpi", "lmp_serial", "lmp_omp",
        "lmp_gpu", "lmp_kokkos", "lmp_rocky9",
    ]
    out, _, _ = conn.run(
        " ; ".join(f'command -v {c} 2>/dev/null && echo "FOUND:{c}"' for c in candidates),
        timeout=10,
    )
    for line in out.splitlines():
        if line.startswith("FOUND:"):
            env.lammps_binaries.append(line.split(":", 1)[1])

    # ── LAMMPS modules ────────────────────────────────────
    # module avail output goes to stderr on most systems
    _, err, _ = conn.run(
        'module avail lammps 2>&1 | grep -i lammps || true',
        timeout=10,
    )
    out_combined = err  # module avail writes to stderr
    # Also try stdout
    out2, _, _ = conn.run(
        'module avail lammps 2>&1 | grep -i lammps || true',
        timeout=10,
    )
    for text in (out_combined, out2):
        for tok in text.split():
            cleaned = tok.strip("()/")
            if "lammps" in cleaned.lower() and cleaned not in env.lammps_modules:
                env.lammps_modules.append(cleaned)

    # ── VASP binaries ─────────────────────────────────────
    candidates = [
        "vasp", "vasp_std", "vasp_gam", "vasp_ncl", "vasp_gpu",
    ]
    out, _, _ = conn.run(
        " ; ".join(f'command -v {c} 2>/dev/null && echo "FOUND:{c}"' for c in candidates),
        timeout=10,
    )
    for line in out.splitlines():
        if line.startswith("FOUND:"):
            env.vasp_binaries.append(line.split(":", 1)[1])

    # ── VASP modules ──────────────────────────────────────
    _, err, _ = conn.run(
        'module avail vasp 2>&1 | grep -i vasp || true',
        timeout=10,
    )
    out_combined = err
    out2, _, _ = conn.run(
        'module avail vasp 2>&1 | grep -i vasp || true',
        timeout=10,
    )
    for text in (out_combined, out2):
        for tok in text.split():
            cleaned = tok.strip("()/")
            if "vasp" in cleaned.lower() and cleaned not in env.vasp_modules:
                env.vasp_modules.append(cleaned)

    # ── Container runtimes ────────────────────────────────
    for rt in ("podman", "docker", "singularity", "apptainer"):
        _, _, rc = conn.run(f"command -v {rt}", timeout=5)
        if rc == 0:
            env.container_runtimes.append(rt)

    # ── Python ────────────────────────────────────────────
    out, _, rc = conn.run("command -v python3 || command -v python", timeout=5)
    if rc == 0:
        env.python_path = out.strip().splitlines()[0]

    return env
