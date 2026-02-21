"""Integration test: full submit -> monitor -> retry cycle with mocked Slurm."""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from slurmgrid.config import RunConfig, SlurmConfig
from slurmgrid.manifest import chunk_manifest, load_headers, count_rows
from slurmgrid.script import (
    ensure_log_dir,
    generate_sbatch_script,
    script_path_for_chunk,
    write_sbatch_script,
)
from slurmgrid.manifest import ChunkInfo
from slurmgrid.slurm import TaskState, TaskStatus
from slurmgrid.state import new_state, save_state, load_state
from slurmgrid.monitor import _poll_once, _submit_to_fill

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestFullCycle(unittest.TestCase):
    """Simulate a full run: chunk, submit, poll, detect failure, retry."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        self.config = RunConfig(
            manifest=self.manifest,
            command="python train.py --alpha {alpha} --beta {beta} --seed {seed}",
            state_dir=self.tmpdir,
            chunk_size=7,
            max_concurrent=14,
            max_retries=1,
            poll_interval=1,
            slurm=SlurmConfig(partition="test", time="00:10:00"),
        )

    def _setup_chunks_and_state(self):
        """Chunk the manifest, generate scripts, initialize state."""
        delimiter = self.config.resolved_delimiter
        chunks_dir = os.path.join(self.tmpdir, "chunks")
        chunks = chunk_manifest(self.manifest, delimiter, 7, chunks_dir,
                                shuffle=False)

        state = new_state(
            total_jobs=count_rows(self.manifest),
            chunk_size=7,
            max_concurrent=14,
            max_retries=1,
        )

        for chunk in chunks:
            row_mapping = {
                str(orig_idx): array_idx
                for array_idx, orig_idx in enumerate(chunk.original_indices)
            }
            state.add_chunk(chunk.chunk_id, chunk.size, row_mapping)

            script = generate_sbatch_script(chunk, self.config, self.tmpdir)
            spath = script_path_for_chunk(chunk, self.tmpdir)
            write_sbatch_script(script, spath)
            ensure_log_dir(chunk, self.tmpdir)

        save_state(state, self.tmpdir)
        return state, chunks

    @patch("slurmgrid.monitor.slurm")
    def test_submit_and_complete(self, mock_slurm):
        """All jobs succeed on first try."""
        state, chunks = self._setup_chunks_and_state()

        # Mock sbatch to return incrementing job IDs
        job_counter = [100]
        def fake_sbatch(script_path):
            jid = str(job_counter[0])
            job_counter[0] += 1
            return jid
        mock_slurm.sbatch.side_effect = fake_sbatch

        # Submit initial batch — one chunk at a time
        _submit_to_fill(state, self.config)
        self.assertEqual(mock_slurm.sbatch.call_count, 1)
        self.assertEqual(state.active_job_count(), 7)

        # Mock sacct: all tasks completed successfully
        def fake_sacct(job_ids):
            result = {}
            for jid in job_ids:
                for c in state.chunks.values():
                    if c.slurm_job_id == jid:
                        for i in range(c.size):
                            result[f"{jid}_{i}"] = TaskStatus(
                                state=TaskState.COMPLETED, exit_code=0,
                            )
            return result
        mock_slurm.sacct_query.side_effect = fake_sacct

        # First poll: chunk 0 completes, chunk 1 gets submitted
        _poll_once(state, self.config)
        self.assertEqual(state.chunks["chunk_000"].status, "completed")
        self.assertEqual(state.chunks["chunk_001"].status, "submitted")

        # Second poll: chunk 1 completes, chunk 2 gets submitted
        _poll_once(state, self.config)
        self.assertEqual(state.chunks["chunk_001"].status, "completed")
        self.assertEqual(state.chunks["chunk_002"].status, "submitted")

        # Third poll: chunk 2 completes
        _poll_once(state, self.config)
        self.assertEqual(state.chunks["chunk_002"].status, "completed")
        self.assertTrue(state.is_done())
        self.assertEqual(state.summary()["failed_final"], 0)

    @patch("slurmgrid.monitor.slurm")
    def test_failure_and_retry(self, mock_slurm):
        """Some tasks fail, get retried, then succeed."""
        state, chunks = self._setup_chunks_and_state()

        job_counter = [100]
        def fake_sbatch(script_path):
            jid = str(job_counter[0])
            job_counter[0] += 1
            return jid
        mock_slurm.sbatch.side_effect = fake_sbatch

        # One chunk at a time
        _submit_to_fill(state, self.config)
        self.assertEqual(mock_slurm.sbatch.call_count, 1)

        # Mock sacct: tasks 3 and 4 in chunk_000 fail
        def fake_sacct(job_ids):
            result = {}
            for jid in job_ids:
                for c in state.chunks.values():
                    if c.slurm_job_id == jid:
                        for i in range(c.size):
                            key = f"{jid}_{i}"
                            if c.chunk_id == "chunk_000" and i in (3, 4):
                                result[key] = TaskStatus(
                                    state=TaskState.FAILED, exit_code=1,
                                )
                            else:
                                result[key] = TaskStatus(
                                    state=TaskState.COMPLETED, exit_code=0,
                                )
            return result
        mock_slurm.sacct_query.side_effect = fake_sacct

        # Poll: chunk_000 partial failure, chunk_001 submitted next
        # (retries are deferred until all regular chunks are done)
        _poll_once(state, self.config)
        self.assertEqual(state.chunks["chunk_000"].status, "partial_failure")
        self.assertEqual(len(state.failures), 2)
        # No retry chunk yet — regular chunks take priority
        retry_chunks = [c for c in state.chunks if "retry" in c]
        self.assertEqual(len(retry_chunks), 0)
        self.assertEqual(state.chunks["chunk_001"].status, "submitted")

        # Poll: chunk_001 completes, chunk_002 submitted
        _poll_once(state, self.config)
        self.assertEqual(state.chunks["chunk_001"].status, "completed")
        self.assertEqual(state.chunks["chunk_002"].status, "submitted")

        # Poll: chunk_002 completes, now retry batch is created and submitted
        _poll_once(state, self.config)
        self.assertEqual(state.chunks["chunk_002"].status, "completed")
        retry_chunks = [c for c in state.chunks if "retry" in c]
        self.assertEqual(len(retry_chunks), 1)
        submitted = [c for c in state.chunks.values() if c.status == "submitted"]
        self.assertEqual(len(submitted), 1)
        # Retry chunk should contain both failed tasks
        retry_chunk = state.chunks[retry_chunks[0]]
        self.assertEqual(retry_chunk.size, 2)

    @patch("slurmgrid.monitor.slurm")
    def test_permanent_failure(self, mock_slurm):
        """Tasks that fail after max_retries are marked permanently failed."""
        # Use max_retries=0 so first failure is permanent
        self.config.max_retries = 0
        state, chunks = self._setup_chunks_and_state()
        state.max_retries = 0

        job_counter = [100]
        def fake_sbatch(script_path):
            jid = str(job_counter[0])
            job_counter[0] += 1
            return jid
        mock_slurm.sbatch.side_effect = fake_sbatch

        _submit_to_fill(state, self.config)

        # Task 0 in chunk_000 fails
        def fake_sacct(job_ids):
            result = {}
            for jid in job_ids:
                for c in state.chunks.values():
                    if c.slurm_job_id == jid:
                        for i in range(c.size):
                            key = f"{jid}_{i}"
                            if c.chunk_id == "chunk_000" and i == 0:
                                result[key] = TaskStatus(
                                    state=TaskState.FAILED, exit_code=1,
                                )
                            else:
                                result[key] = TaskStatus(
                                    state=TaskState.COMPLETED, exit_code=0,
                                )
            return result
        mock_slurm.sacct_query.side_effect = fake_sacct

        _poll_once(state, self.config)
        self.assertEqual(len(state.failures), 1)
        self.assertTrue(state.failures["0"].permanently_failed)
        # No retry chunks should be created
        retry_chunks = [c for c in state.chunks if "retry" in c]
        self.assertEqual(len(retry_chunks), 0)


if __name__ == "__main__":
    unittest.main()
