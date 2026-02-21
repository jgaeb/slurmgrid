"""State persistence for slurmgrid.

State is stored as JSON with atomic writes (write to temp file, then rename)
to prevent corruption if the process is interrupted mid-write.
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# Chunk lifecycle: pending -> submitted -> running -> completed
#                                                  \-> partial_failure -> (retry chunks created)
#                              \-> submit_failed -> (retry submission next poll)
CHUNK_STATUSES = {
    "pending",
    "submitted",
    "running",
    "completed",
    "partial_failure",
    "submit_failed",
}


@dataclass
class FailureRecord:
    """A single failed task that may be retried."""

    global_index: int  # Row index in the original manifest (0-based, excluding header)
    chunk_id: str  # Chunk it most recently ran in
    array_index: int  # Array index within that chunk
    exit_code: int
    retries: int = 0
    permanently_failed: bool = False


@dataclass
class ChunkState:
    """Status of a single chunk."""

    chunk_id: str
    status: str = "pending"
    slurm_job_id: Optional[str] = None
    submitted_at: Optional[str] = None
    size: int = 0
    # Number of tasks that completed successfully so far (updated each poll).
    completed_tasks: int = 0
    # Mapping from global manifest row index to the array index in this chunk.
    # Populated at chunking time so we can trace failures back to the original row.
    row_mapping: Dict[str, int] = field(default_factory=dict)


@dataclass
class State:
    """Master state for a slurmgrid run."""

    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    total_jobs: int = 0
    chunk_size: int = 0
    max_concurrent: int = 0
    max_retries: int = 0
    chunks: Dict[str, ChunkState] = field(default_factory=dict)
    failures: Dict[str, FailureRecord] = field(default_factory=dict)

    # --- Chunk queries ---

    def pending_chunks(self) -> List[ChunkState]:
        """Chunks waiting to be submitted, in order."""
        return [
            c for c in self.chunks.values()
            if c.status in ("pending", "submit_failed")
        ]

    def active_chunks(self) -> List[ChunkState]:
        """Chunks that have been submitted and are not yet done."""
        return [
            c for c in self.chunks.values()
            if c.status in ("submitted", "running")
        ]

    def active_job_count(self) -> int:
        """Total number of in-flight tasks across all active chunks."""
        return sum(c.size for c in self.active_chunks())

    def is_done(self) -> bool:
        """True if all chunks are completed or permanently failed."""
        return all(
            c.status in ("completed", "partial_failure")
            for c in self.chunks.values()
        ) and all(
            f.permanently_failed for f in self.failures.values()
        ) if self.failures else all(
            c.status == "completed" for c in self.chunks.values()
        )

    def all_retries_resolved(self) -> bool:
        """True if no failures are pending retry."""
        return all(
            f.permanently_failed or f.retries >= self.max_retries
            for f in self.failures.values()
        )

    # --- Mutations ---

    def add_chunk(self, chunk_id: str, size: int,
                  row_mapping: Dict[str, int]) -> None:
        self.chunks[chunk_id] = ChunkState(
            chunk_id=chunk_id, size=size, row_mapping=row_mapping,
        )

    def mark_submitted(self, chunk_id: str, slurm_job_id: str) -> None:
        c = self.chunks[chunk_id]
        c.status = "submitted"
        c.slurm_job_id = slurm_job_id
        c.submitted_at = _now()

    def mark_running(self, chunk_id: str) -> None:
        self.chunks[chunk_id].status = "running"

    def mark_completed(self, chunk_id: str) -> None:
        self.chunks[chunk_id].status = "completed"

    def mark_partial_failure(self, chunk_id: str) -> None:
        self.chunks[chunk_id].status = "partial_failure"

    def mark_submit_failed(self, chunk_id: str) -> None:
        self.chunks[chunk_id].status = "submit_failed"

    def record_failure(self, global_index: int, chunk_id: str,
                       array_index: int, exit_code: int) -> None:
        key = str(global_index)
        if key in self.failures:
            rec = self.failures[key]
            rec.retries += 1
            rec.chunk_id = chunk_id
            rec.array_index = array_index
            rec.exit_code = exit_code
            if rec.retries >= self.max_retries:
                rec.permanently_failed = True
        else:
            perm = self.max_retries == 0
            self.failures[key] = FailureRecord(
                global_index=global_index,
                chunk_id=chunk_id,
                array_index=array_index,
                exit_code=exit_code,
                retries=0,
                permanently_failed=perm,
            )

    # --- Summary ---

    def summary(self) -> Dict[str, int]:
        completed_chunks = sum(
            1 for c in self.chunks.values()
            if c.status in ("completed", "partial_failure")
        )
        # Per-task completion counts from all chunks (including active ones)
        completed_tasks = sum(c.completed_tasks for c in self.chunks.values())
        active = sum(
            c.size - c.completed_tasks for c in self.active_chunks()
        )
        pending_chunks = len(self.pending_chunks())
        pending_tasks = sum(c.size for c in self.pending_chunks())
        failed_retry = sum(
            1 for f in self.failures.values() if not f.permanently_failed
        )
        failed_final = sum(
            1 for f in self.failures.values() if f.permanently_failed
        )
        return {
            "total_jobs": self.total_jobs,
            "completed_tasks": completed_tasks,
            "active_tasks": active,
            "pending_tasks": pending_tasks,
            "completed_chunks": completed_chunks,
            "active_chunks": len(self.active_chunks()),
            "pending_chunks": pending_chunks,
            "total_chunks": len(self.chunks),
            "failed_retry": failed_retry,
            "failed_final": failed_final,
        }


def new_state(total_jobs: int, chunk_size: int, max_concurrent: int,
              max_retries: int) -> State:
    """Create a fresh State."""
    now = _now()
    return State(
        created_at=now,
        updated_at=now,
        total_jobs=total_jobs,
        chunk_size=chunk_size,
        max_concurrent=max_concurrent,
        max_retries=max_retries,
    )


def save_state(state: State, state_dir: str) -> None:
    """Atomically write state to state_dir/state.json."""
    state.updated_at = _now()
    path = os.path.join(state_dir, "state.json")
    data = _serialize(state)
    # Write to temp file in the same directory, then rename (atomic on POSIX)
    fd, tmp_path = tempfile.mkstemp(dir=state_dir, suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
            f.write("\n")
        os.rename(tmp_path, path)
    except BaseException:
        # Clean up temp file on any failure
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_state(state_dir: str) -> State:
    """Load state from state_dir/state.json."""
    path = os.path.join(state_dir, "state.json")
    with open(path) as f:
        data = json.load(f)
    return _deserialize(data)


def state_exists(state_dir: str) -> bool:
    """Check if a state file exists."""
    return os.path.isfile(os.path.join(state_dir, "state.json"))


# --- Serialization helpers ---

def _serialize(state: State) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "version": state.version,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
        "total_jobs": state.total_jobs,
        "chunk_size": state.chunk_size,
        "max_concurrent": state.max_concurrent,
        "max_retries": state.max_retries,
        "chunks": {},
        "failures": {},
    }
    for cid, cs in state.chunks.items():
        data["chunks"][cid] = asdict(cs)
    for key, fr in state.failures.items():
        data["failures"][key] = asdict(fr)
    return data


def _deserialize(data: Dict[str, Any]) -> State:
    if data.get("version", 1) != 1:
        raise ValueError(f"Unsupported state version: {data.get('version')}")
    state = State(
        version=data["version"],
        created_at=data["created_at"],
        updated_at=data["updated_at"],
        total_jobs=data["total_jobs"],
        chunk_size=data["chunk_size"],
        max_concurrent=data["max_concurrent"],
        max_retries=data["max_retries"],
    )
    for cid, cs_data in data.get("chunks", {}).items():
        state.chunks[cid] = ChunkState(**cs_data)
    for key, fr_data in data.get("failures", {}).items():
        state.failures[key] = FailureRecord(**fr_data)
    return state


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
