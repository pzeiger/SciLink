"""Run refinement-loop phases on an HPC scheduler.

``ClusterExecutor`` is the cluster counterpart to ``LocalExecutor``: it
satisfies the same :class:`~scilink.agents.sim_agents.refinement.Executor`
contract, so the refinement loop drives it without knowing a run happens on a
remote scheduler rather than a local subprocess. A ``run`` uploads the phase's
input files, submits a batch job, polls until the job reaches a terminal state,
downloads the outputs back into the local ``run_dir``, and persists the same
engine-neutral ``stdout`` / ``stderr`` / ``returncode`` files the local executor
writes — so the post-run critic's snapshot is identical regardless of executor.

This lives with the loop (not in ``scilink.hpc``) because it implements the
loop's ``Executor`` contract; ``scilink.hpc`` stays self-contained and is
imported lazily, so importing the loop does not pull in paramiko.

v1 runs one job per ``run`` call and blocks until it finishes. The refinement
loop calls ``run`` serially per phase (and per fan-out member), so a campaign's
parallel members are submitted one after another. Concurrent submission across
members is a future extension (a batch ``run_many`` seam) and does not change
this contract.
"""
from __future__ import annotations

import logging
import stat as _stat
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from .refinement import Executor, LocalExecutor

if TYPE_CHECKING:
    from scilink.hpc.connection import HPCConnection
    from scilink.hpc.scheduler import Scheduler

logger = logging.getLogger(__name__)

# Reuse the local executor's engine-neutral output filenames so a cluster run's
# snapshot is byte-for-byte the same shape the critic already reads.
_STDOUT = LocalExecutor.STDOUT_FILE
_STDERR = LocalExecutor.STDERR_FILE
_RC = LocalExecutor.RETURNCODE_FILE


class ClusterExecutor(Executor):
    """Execute inputs as a scheduler batch job on a remote host.

    Attributes:
        conn: A connected (or connectable) ``HPCConnection``.
        remote_root: Base remote directory under which per-run directories are
            created. Defaults to the remote home directory.
        resources: Engine-neutral resource request for the batch directives
            (partition, nodes, ntasks, time, account, gres, extra_directives).
        setup: Shell lines run before the command (module loads, env setup).
        poll_interval: Seconds between job-status polls.
        timeout: Overall wall-clock cap (seconds) before the job is cancelled.
    """

    def __init__(
        self,
        connection: "HPCConnection",
        *,
        scheduler: "Scheduler | None" = None,
        remote_root: Optional[str] = None,
        resources: Optional[dict] = None,
        setup: Optional[List[str]] = None,
        poll_interval: int = 30,
        timeout: int = 86400,
        job_script_name: str = "scilink_job.sh",
    ):
        self.conn = connection
        self._scheduler = scheduler
        self.remote_root = remote_root
        self.resources = resources or {}
        self.setup = setup or []
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.job_script_name = job_script_name

    # ── internals ─────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if not self.conn.is_connected:
            self.conn.connect()

    def _resolve_scheduler(self) -> "Scheduler":
        if self._scheduler is None:
            from scilink.hpc import detect_scheduler
            self._scheduler = detect_scheduler(self.conn)
            if self._scheduler is None:
                raise RuntimeError(
                    "No supported scheduler (SLURM/PBS/LSF) detected on the "
                    "remote host."
                )
        return self._scheduler

    def _remote_dir_for(self, run_dir: str) -> str:
        base = (self.remote_root or self.conn.home_dir()).rstrip("/")
        return f"{base}/{Path(run_dir).name}"

    def _download_outputs(self, remote_dir: str, local: Path) -> None:
        """Download every regular file in the remote run dir to ``local``.

        Subdirectories are skipped — a phase's run directory is flat (the loop
        gives each phase / fan-out member its own directory).
        """
        try:
            entries = self.conn.listdir(remote_dir)
        except Exception as exc:  # noqa: BLE001 — best-effort retrieval
            logger.warning("ClusterExecutor: could not list %s: %s", remote_dir, exc)
            return
        for attr in entries:
            name = attr.filename
            if name in (".", ".."):
                continue
            if _stat.S_ISDIR(attr.st_mode):
                continue
            try:
                self.conn.download(f"{remote_dir}/{name}", str(local / name))
            except Exception as exc:  # noqa: BLE001 — one bad file shouldn't abort
                logger.warning("ClusterExecutor: could not download %s: %s", name, exc)

    # ── Executor contract ─────────────────────────────────────

    def run(
        self, input_files: Dict[str, str], run_command: str, run_dir: str
    ) -> Dict[str, Any]:
        from scilink.hpc.batch import build_batch_script

        run_path = Path(run_dir)
        run_path.mkdir(parents=True, exist_ok=True)

        # Materialize inputs locally too — gives the local snapshot the inputs
        # and serves as the upload source.
        for name, contents in (input_files or {}).items():
            (run_path / name).write_text(contents)

        self._ensure_connected()
        sched = self._resolve_scheduler()
        remote_dir = self._remote_dir_for(run_dir)
        self.conn.mkdir_p(remote_dir)

        for name in (input_files or {}):
            self.conn.upload(str(run_path / name), f"{remote_dir}/{name}")

        script = build_batch_script(
            sched, run_command,
            resources=self.resources, setup=self.setup,
            stdout_file=_STDOUT, stderr_file=_STDERR, rc_file=_RC,
        )
        (run_path / self.job_script_name).write_text(script)
        self.conn.upload(
            str(run_path / self.job_script_name),
            f"{remote_dir}/{self.job_script_name}",
        )

        try:
            job_id = sched.submit(self.job_script_name, work_dir=remote_dir)
        except Exception as exc:  # noqa: BLE001 — surface as an executor error
            (run_path / _STDERR).write_text(f"Job submission failed: {exc}")
            (run_path / _RC).write_text("submit_error")
            return {
                "status": "error", "output_dir": str(run_path),
                "returncode": None, "error": f"Job submission failed: {exc}",
            }
        logger.info("ClusterExecutor: submitted job %s in %s", job_id, remote_dir)

        elapsed = 0
        while True:
            job = sched.status(job_id)
            if job.status.is_terminal:
                break
            if elapsed >= self.timeout:
                sched.cancel(job_id)
                self._download_outputs(remote_dir, run_path)
                (run_path / _STDERR).write_text(
                    f"Job {job_id} exceeded {self.timeout}s wall-clock; cancelled."
                )
                (run_path / _RC).write_text("timeout")
                return {
                    "status": "error", "output_dir": str(run_path),
                    "returncode": None,
                    "error": f"Job timed out after {self.timeout}s",
                }
            time.sleep(self.poll_interval)
            elapsed += self.poll_interval

        self._download_outputs(remote_dir, run_path)

        # Prefer the returncode the wrapper recorded; fall back to the
        # scheduler's reported exit code.
        returncode: Optional[int] = None
        rc_path = run_path / _RC
        if rc_path.exists():
            try:
                returncode = int(rc_path.read_text().strip())
            except (ValueError, OSError):
                returncode = None
        if returncode is None:
            returncode = job.exit_code

        return {
            "status": "completed",
            "output_dir": str(run_path),
            "returncode": returncode,
        }
