"""Low-level wrappers around Slurm CLI commands.

All subprocess calls to Slurm are centralized here, making it easy to mock
for testing or adapt to different Slurm versions.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

# Retry parameters for transient Slurm failures
MAX_RETRIES = 3
RETRY_DELAYS = [5, 15, 45]  # seconds


class SlurmError(Exception):
    """Raised when a Slurm command fails."""


class TaskState(Enum):
    """Slurm job/task states we care about."""

    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    TIMEOUT = "TIMEOUT"
    CANCELLED = "CANCELLED"
    NODE_FAIL = "NODE_FAIL"
    OUT_OF_MEMORY = "OUT_OF_MEMORY"
    UNKNOWN = "UNKNOWN"

    @classmethod
    def from_sacct(cls, state_str: str) -> TaskState:
        """Parse a sacct state string into a TaskState.

        sacct can return states like "FAILED" or "CANCELLED by 12345".
        """
        s = state_str.strip().split()[0].upper()
        # Handle compound states like "CANCELLED+"
        s = s.rstrip("+")
        try:
            return cls(s)
        except ValueError:
            return cls.UNKNOWN

    @property
    def is_terminal(self) -> bool:
        return self in (
            TaskState.COMPLETED,
            TaskState.FAILED,
            TaskState.TIMEOUT,
            TaskState.CANCELLED,
            TaskState.NODE_FAIL,
            TaskState.OUT_OF_MEMORY,
        )

    @property
    def is_success(self) -> bool:
        return self == TaskState.COMPLETED

    @property
    def is_retriable_failure(self) -> bool:
        return self in (
            TaskState.FAILED,
            TaskState.TIMEOUT,
            TaskState.CANCELLED,
            TaskState.NODE_FAIL,
            TaskState.OUT_OF_MEMORY,
        )


@dataclass
class TaskStatus:
    """Status of a single array task."""

    state: TaskState
    exit_code: int


def _run_command(
    cmd: List[str],
    description: str,
    retries: int = MAX_RETRIES,
) -> subprocess.CompletedProcess:
    """Run a command with retry logic for transient failures."""
    for attempt in range(retries):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                return result
            # Some Slurm errors are transient (e.g., slurmdbd connection)
            if attempt < retries - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                log.warning(
                    "%s failed (attempt %d/%d), retrying in %ds: %s",
                    description, attempt + 1, retries, delay,
                    result.stderr.strip(),
                )
                time.sleep(delay)
            else:
                raise SlurmError(
                    f"{description} failed after {retries} attempts: "
                    f"{result.stderr.strip()}"
                )
        except subprocess.TimeoutExpired:
            if attempt < retries - 1:
                delay = RETRY_DELAYS[min(attempt, len(RETRY_DELAYS) - 1)]
                log.warning(
                    "%s timed out (attempt %d/%d), retrying in %ds",
                    description, attempt + 1, retries, delay,
                )
                time.sleep(delay)
            else:
                raise SlurmError(
                    f"{description} timed out after {retries} attempts"
                )
    # Should not reach here, but satisfy type checker
    raise SlurmError(f"{description} failed")


def sbatch(script_path: str) -> str:
    """Submit a job via sbatch. Returns the Slurm job ID."""
    result = _run_command(
        ["sbatch", "--parsable", script_path],
        f"sbatch {script_path}",
    )
    # --parsable output is "jobid" or "jobid;cluster"
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise SlurmError(f"Unexpected sbatch output: {result.stdout.strip()}")
    log.info("Submitted %s -> job %s", script_path, job_id)
    return job_id


def sbatch_wrap(wrap_cmd: str, sbatch_flags: List[str]) -> str:
    """Submit a job via sbatch --wrap. Returns the Slurm job ID."""
    result = _run_command(
        ["sbatch", "--parsable"] + sbatch_flags + [f"--wrap={wrap_cmd}"],
        "sbatch --wrap",
    )
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise SlurmError(f"Unexpected sbatch output: {result.stdout.strip()}")
    log.info("Submitted monitor job -> job %s", job_id)
    return job_id


def sacct_query(job_ids: List[str]) -> Dict[str, TaskStatus]:
    """Query sacct for array task statuses.

    Returns a dict mapping "jobid_arrayidx" to TaskStatus.
    Only returns entries for individual array tasks (not the parent job).
    """
    if not job_ids:
        return {}

    result = _run_command(
        [
            "sacct",
            "--jobs=" + ",".join(job_ids),
            "--format=JobID,State,ExitCode",
            "--parsable2",
            "--noheader",
        ],
        "sacct query",
    )

    log.debug("sacct raw output: %s", result.stdout)

    statuses: Dict[str, TaskStatus] = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.split("|")
        if len(parts) < 3:
            continue
        job_id_str, state_str, exit_str = parts[0], parts[1], parts[2]

        # We only care about array task lines, which look like "12345_0".
        # Skip:
        #   - Parent job lines ("12345")
        #   - Batch/extern steps ("12345_0.batch", "12345_0.extern")
        #   - Summarized ranges ("12345_[0-4]")
        if "_" not in job_id_str or "." in job_id_str:
            continue
        # Extract the part after "_" and check it's a plain integer
        _, array_part = job_id_str.split("_", 1)
        if not array_part.isdigit():
            continue

        state = TaskState.from_sacct(state_str)
        # Exit code format is "exitcode:signal", e.g., "0:0" or "1:0"
        try:
            exit_code = int(exit_str.split(":")[0])
        except (ValueError, IndexError):
            exit_code = -1

        statuses[job_id_str] = TaskStatus(state=state, exit_code=exit_code)

    return statuses


def squeue_query(job_ids: List[str]) -> Dict[str, str]:
    """Fallback query via squeue for running/pending jobs.

    Returns a dict mapping "jobid_arrayidx" to state string.
    """
    if not job_ids:
        return {}

    result = _run_command(
        [
            "squeue",
            "--jobs=" + ",".join(job_ids),
            "--format=%i|%T",
            "--noheader",
        ],
        "squeue query",
    )

    statuses: Dict[str, str] = {}
    for line in result.stdout.strip().splitlines():
        if not line:
            continue
        parts = line.strip().split("|")
        if len(parts) < 2:
            continue
        statuses[parts[0]] = parts[1]

    return statuses


def scancel(job_ids: List[str]) -> None:
    """Cancel Slurm jobs."""
    if not job_ids:
        return
    _run_command(
        ["scancel"] + job_ids,
        "scancel",
    )
    log.info("Cancelled jobs: %s", ", ".join(job_ids))


def get_user_job_count() -> Optional[int]:
    """Count total running+pending tasks for the current user across all jobs.

    Returns None if the query fails (callers should treat this as unknown).
    """
    user = os.environ.get("USER") or os.environ.get("LOGNAME")
    if not user:
        log.warning("Could not determine current user for squeue headroom check")
        return None
    try:
        result = _run_command(
            ["squeue", "--noheader", f"--user={user}"],
            "squeue user job count",
            retries=1,
        )
        lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
        return len(lines)
    except (SlurmError, OSError) as e:
        log.warning("Could not query user job count: %s", e)
        return None


def get_max_array_size() -> int:
    """Query Slurm for MaxArraySize. Returns 1001 as default on failure."""
    try:
        result = _run_command(
            ["scontrol", "show", "config"],
            "scontrol show config",
            retries=1,
        )
        for line in result.stdout.splitlines():
            if "MaxArraySize" in line:
                match = re.search(r"MaxArraySize\s*=\s*(\d+)", line)
                if match:
                    val = int(match.group(1))
                    log.info("Detected MaxArraySize = %d", val)
                    return val
    except SlurmError:
        log.warning("Could not detect MaxArraySize, using default 1001")
    return 1001
