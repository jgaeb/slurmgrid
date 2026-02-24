"""Main monitoring loop: submission, polling, and retry orchestration."""

from __future__ import annotations

import csv
import logging
import os
import signal
import time
from typing import Dict, List, Optional, Tuple

from . import slurm
from .config import RunConfig
from .manifest import (
    CHUNK_DELIMITER,
    ChunkInfo,
    chunk_manifest,
    load_headers,
)
from .script import (
    ensure_log_dir,
    generate_sbatch_script,
    script_path_for_chunk,
    write_sbatch_script,
)
from .state import (
    ChunkState,
    FailureRecord,
    State,
    load_state,
    save_state,
)

log = logging.getLogger(__name__)


class GracefulExit(Exception):
    """Raised when a shutdown signal is received."""


def run(state: State, config: RunConfig) -> State:
    """Main entry point: submit initial chunks and run the monitor loop.

    Returns the final state when all work is done or a signal is caught.
    """
    _install_signal_handlers()
    start_time = time.monotonic()

    try:
        # Wait for upstream dependency if specified
        if config.after_run:
            _wait_for_upstream(config)

        # Initial submission to fill available slots
        _submit_to_fill(state, config)
        save_state(state, config.state_dir)

        # Polling loop
        while not state.is_done():
            if config.max_runtime and (
                time.monotonic() - start_time > config.max_runtime
            ):
                log.warning(
                    "Max runtime of %ds reached, saving state and exiting",
                    config.max_runtime,
                )
                break

            time.sleep(config.poll_interval)
            _poll_once(state, config)
            save_state(state, config.state_dir)
            _log_progress(state)

    except GracefulExit:
        log.info("Caught shutdown signal, saving state and exiting")

    save_state(state, config.state_dir)
    if not state.is_done():
        log.info("State saved. Running Slurm jobs will continue independently.")
        log.info("Resume with: python -m slurmgrid resume --state-dir %s",
                 config.state_dir)

    return state


def _wait_for_upstream(config: RunConfig) -> None:
    """Poll the upstream state directory until that run reports done."""
    log.info("Waiting for upstream run at %s to complete...", config.after_run)
    while True:
        try:
            prev = load_state(config.after_run)
            if prev.is_done():
                log.info("Upstream run complete, proceeding.")
                return
            s = prev.summary()
            log.info(
                "Upstream: %d/%d completed, %d active, %d pending",
                s["completed_tasks"], s["total_jobs"],
                s["active_tasks"], s["pending_tasks"],
            )
        except (OSError, ValueError) as e:
            log.warning("Could not read upstream state: %s", e)
        time.sleep(config.poll_interval)


def _poll_once(state: State, config: RunConfig) -> None:
    """Single iteration of the monitor loop."""
    # 1. Query Slurm for status of all active chunks
    active = state.active_chunks()
    if active:
        _update_chunk_statuses(active, state, config)

    # 2. Submit more chunks if we have capacity.
    #    If no regular chunks are pending, this will batch any accumulated
    #    failures into a retry chunk before submitting.
    _submit_to_fill(state, config)


def _update_chunk_statuses(
    active: List[ChunkState],
    state: State,
    config: RunConfig,
) -> None:
    """Query sacct and update the status of active chunks."""
    job_ids = [c.slurm_job_id for c in active if c.slurm_job_id]
    if not job_ids:
        return

    try:
        statuses = slurm.sacct_query(job_ids)
    except slurm.SlurmError as e:
        log.warning("sacct query failed, will retry next poll: %s", e)
        return

    for chunk in active:
        if not chunk.slurm_job_id:
            continue
        _update_single_chunk(chunk, chunk.slurm_job_id, statuses, state)


def _update_single_chunk(
    chunk: ChunkState,
    job_id: str,
    all_statuses: Dict[str, slurm.TaskStatus],
    state: State,
) -> None:
    """Update a single chunk's status based on sacct results."""
    # Collect statuses for this chunk's array tasks
    chunk_tasks: Dict[int, slurm.TaskStatus] = {}
    for array_idx in range(chunk.size):
        key = f"{job_id}_{array_idx}"
        if key in all_statuses:
            chunk_tasks[array_idx] = all_statuses[key]

    if not chunk_tasks:
        # No sacct results yet — job may still be queued
        return

    # Update per-task completion count (for progress reporting)
    succeeded = sum(1 for ts in chunk_tasks.values() if ts.state.is_success)
    state.chunks[chunk.chunk_id].completed_tasks = succeeded

    # Check if all tasks have reached a terminal state
    all_terminal = (
        len(chunk_tasks) == chunk.size
        and all(ts.state.is_terminal for ts in chunk_tasks.values())
    )

    # Check if any are running (transition from submitted -> running)
    any_running = any(
        not ts.state.is_terminal for ts in chunk_tasks.values()
    )
    if any_running and chunk.status == "submitted":
        state.mark_running(chunk.chunk_id)

    if not all_terminal:
        return

    # All tasks are done — classify the chunk
    all_success = all(ts.state.is_success for ts in chunk_tasks.values())

    if all_success:
        state.mark_completed(chunk.chunk_id)
        # Clear failure records for tasks that succeeded on retry
        for array_idx in range(chunk.size):
            global_idx = _array_idx_to_global(chunk, array_idx)
            if global_idx is not None:
                key = str(global_idx)
                if key in state.failures:
                    del state.failures[key]
        log.info("Chunk %s completed successfully (%d tasks)",
                 chunk.chunk_id, chunk.size)
    else:
        # Record individual failures and clear records for retried tasks
        # that succeeded this time
        state.mark_partial_failure(chunk.chunk_id)
        failed_count = 0
        for array_idx, ts in chunk_tasks.items():
            global_idx = _array_idx_to_global(chunk, array_idx)
            if ts.state.is_retriable_failure:
                if global_idx is not None:
                    state.record_failure(
                        global_idx, chunk.chunk_id, array_idx, ts.exit_code,
                    )
                    failed_count += 1
            elif ts.state.is_success and global_idx is not None:
                # Task succeeded on retry — clear its failure record
                key = str(global_idx)
                if key in state.failures:
                    del state.failures[key]
            elif ts.state == slurm.TaskState.CANCELLED:
                log.warning("Chunk %s task %d was cancelled",
                            chunk.chunk_id, array_idx)

        succeeded = sum(1 for ts in chunk_tasks.values() if ts.state.is_success)
        log.info("Chunk %s finished: %d succeeded, %d failed",
                 chunk.chunk_id, succeeded, failed_count)


def _submit_to_fill(state: State, config: RunConfig) -> None:
    """Submit pending chunks to fill available task capacity.

    Submits chunks until adding another would exceed max_concurrent total
    active tasks. Always submits at least one chunk if none are active
    (ensures forward progress). For example, with max_concurrent=10000
    and chunk_size=5000, up to 2 chunks run simultaneously.

    If no regular chunks are pending but there are retriable failures,
    batch them into retry chunks and submit those (only when all active
    chunks have finished to avoid premature retries).
    """
    pending = state.pending_chunks()
    if not pending and not state.active_chunks():
        # No regular pending chunks and nothing active — try retry batches.
        _create_retry_batch(state, config)
        pending = state.pending_chunks()

    while pending:
        active_tasks = state.active_job_count()
        next_chunk = pending[0]
        # Always submit if nothing is active (ensures forward progress).
        # Otherwise only submit if we have remaining capacity.
        if active_tasks > 0 and active_tasks + next_chunk.size > config.max_concurrent:
            break
        _submit_chunk(next_chunk, state, config)
        pending = state.pending_chunks()


def _create_retry_batch(state: State, config: RunConfig) -> None:
    """Batch all retriable failures into a single retry chunk (or split
    into multiple if there are more than chunk_size failures)."""
    retriable = [
        f for f in state.failures.values()
        if not f.permanently_failed and f.retries < config.max_retries
    ]
    if not retriable:
        return

    # Read the original manifest to extract the failed rows
    delimiter = config.resolved_delimiter
    needed_indices = {ft.global_index for ft in retriable}
    rows_by_global: Dict[int, str] = {}

    with open(config.manifest, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header_row = next(reader)
        for row_idx, row in enumerate(reader):
            if row_idx in needed_indices:
                rows_by_global[row_idx] = CHUNK_DELIMITER.join(row)

    header_line = CHUNK_DELIMITER.join(header_row)
    chunk_size = state.chunk_size

    # Determine retry round number from existing retry chunks
    retry_round = sum(1 for cid in state.chunks if cid.startswith("retry_"))

    # Split into chunks of chunk_size
    sorted_indices = sorted(rows_by_global.keys())
    for batch_start in range(0, len(sorted_indices), chunk_size):
        batch_indices = sorted_indices[batch_start:batch_start + chunk_size]
        chunk_id = f"retry_{retry_round:03d}"
        retry_round += 1

        # Write the chunk file
        chunks_dir = os.path.join(config.state_dir, "chunks")
        os.makedirs(chunks_dir, exist_ok=True)
        chunk_path = os.path.join(chunks_dir, f"{chunk_id}.chunk")
        with open(chunk_path, "w") as f:
            f.write(header_line + "\n")
            for gi in batch_indices:
                f.write(rows_by_global[gi] + "\n")

        # Build row mapping (global_index -> array_index in this chunk)
        row_mapping = {
            str(gi): array_idx
            for array_idx, gi in enumerate(batch_indices)
        }

        # Register in state
        state.add_chunk(chunk_id, len(batch_indices), row_mapping)

        # Generate sbatch script
        chunk_info = ChunkInfo(
            chunk_id=chunk_id, path=chunk_path, size=len(batch_indices),
        )
        throttle = config.max_concurrent
        script_content = generate_sbatch_script(
            chunk_info, config, config.state_dir, throttle=throttle,
        )
        spath = script_path_for_chunk(chunk_info, config.state_dir)
        write_sbatch_script(script_content, spath)
        ensure_log_dir(chunk_info, config.state_dir)

        log.info("Created retry chunk %s with %d tasks",
                 chunk_id, len(batch_indices))


def _submit_chunk(chunk: ChunkState, state: State, config: RunConfig) -> None:
    """Submit a single chunk via sbatch."""
    script = os.path.join(
        config.state_dir, "scripts", f"{chunk.chunk_id}.sh",
    )
    if not os.path.isfile(script):
        log.error("Script not found for chunk %s: %s", chunk.chunk_id, script)
        state.mark_submit_failed(chunk.chunk_id)
        return

    if config.dry_run:
        log.info("[DRY RUN] Would submit %s", script)
        return

    try:
        job_id = slurm.sbatch(script)
        state.mark_submitted(chunk.chunk_id, job_id)
        log.info("Submitted chunk %s -> Slurm job %s (%d tasks)",
                 chunk.chunk_id, job_id, chunk.size)
    except slurm.SlurmError as e:
        log.error("Failed to submit chunk %s: %s", chunk.chunk_id, e)
        state.mark_submit_failed(chunk.chunk_id)


def _array_idx_to_global(chunk: ChunkState, array_idx: int) -> Optional[int]:
    """Map an array index back to the global manifest row index."""
    for global_str, aidx in chunk.row_mapping.items():
        if aidx == array_idx:
            return int(global_str)
    return None


def _log_progress(state: State) -> None:
    """Log a one-line progress summary."""
    s = state.summary()
    log.info(
        "Progress: %d/%d completed, %d active, %d pending, "
        "%d failed (retrying), %d failed (final)",
        s["completed_tasks"], s["total_jobs"], s["active_tasks"],
        s["pending_tasks"], s["failed_retry"], s["failed_final"],
    )


def _install_signal_handlers() -> None:
    """Install handlers for graceful shutdown on SIGINT and SIGHUP."""
    def handler(signum, frame):
        raise GracefulExit()

    signal.signal(signal.SIGINT, handler)
    signal.signal(signal.SIGHUP, handler)
