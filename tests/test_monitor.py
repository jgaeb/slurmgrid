"""Tests for monitor loop edge cases."""

import os
import signal
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from slurmgrid.config import RunConfig, SlurmConfig
from slurmgrid.manifest import chunk_manifest, count_rows
from slurmgrid.monitor import (
    GracefulExit,
    _install_signal_handlers,
    _log_progress,
    _submit_chunk,
    _submit_to_fill,
    _update_single_chunk,
    run,
)
from slurmgrid.script import (
    ensure_log_dir,
    generate_sbatch_script,
    script_path_for_chunk,
    write_sbatch_script,
)
from slurmgrid.slurm import SlurmError, TaskState, TaskStatus
from slurmgrid.state import ChunkState, new_state, save_state

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestSubmitChunk(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = RunConfig(
            manifest=os.path.join(FIXTURES, "sample_manifest.csv"),
            command="echo {alpha}",
            state_dir=self.tmpdir,
            chunk_size=5,
            max_concurrent=10,
            max_retries=0,
            poll_interval=1,
            slurm=SlurmConfig(),
        )

    @patch("slurmgrid.monitor.slurm")
    def test_submit_missing_script(self, mock_slurm):
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {"0": 0})
        chunk = state.chunks["chunk_000"]
        _submit_chunk(chunk, state, self.config)
        # Should mark as submit_failed
        self.assertEqual(state.chunks["chunk_000"].status, "submit_failed")
        mock_slurm.sbatch.assert_not_called()

    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_submit_sbatch_error(self, mock_sbatch):
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {"0": 0})

        # Create a script file so it passes the existence check
        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")

        mock_sbatch.side_effect = SlurmError("connection refused")
        chunk = state.chunks["chunk_000"]
        _submit_chunk(chunk, state, self.config)
        self.assertEqual(state.chunks["chunk_000"].status, "submit_failed")

    @patch("slurmgrid.monitor.slurm")
    def test_submit_dry_run_no_submit(self, mock_slurm):
        self.config.dry_run = True
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {"0": 0})

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")

        chunk = state.chunks["chunk_000"]
        _submit_chunk(chunk, state, self.config)
        mock_slurm.sbatch.assert_not_called()
        # Should still be pending (not submitted)
        self.assertEqual(state.chunks["chunk_000"].status, "pending")


class TestUpdateSingleChunk(unittest.TestCase):
    def test_running_transition(self):
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 2, {"0": 0, "1": 1})
        state.mark_submitted("chunk_000", "100")
        chunk = state.chunks["chunk_000"]

        # One task running, one pending in sacct
        statuses = {
            "100_0": TaskStatus(state=TaskState.RUNNING, exit_code=0),
        }
        _update_single_chunk(chunk, "100", statuses, state)
        self.assertEqual(chunk.status, "running")

    def test_no_sacct_results(self):
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 2, {"0": 0, "1": 1})
        state.mark_submitted("chunk_000", "100")
        chunk = state.chunks["chunk_000"]

        _update_single_chunk(chunk, "100", {}, state)
        # Should remain submitted
        self.assertEqual(chunk.status, "submitted")

    def test_cancelled_task(self):
        state = new_state(5, 5, 10, 1)
        state.add_chunk("chunk_000", 2, {"0": 0, "1": 1})
        state.mark_submitted("chunk_000", "100")
        chunk = state.chunks["chunk_000"]

        statuses = {
            "100_0": TaskStatus(state=TaskState.COMPLETED, exit_code=0),
            "100_1": TaskStatus(state=TaskState.CANCELLED, exit_code=0),
        }
        _update_single_chunk(chunk, "100", statuses, state)
        self.assertEqual(chunk.status, "partial_failure")
        # Cancelled tasks are not recorded as failures
        self.assertEqual(len(state.failures), 0)


class TestLogProgress(unittest.TestCase):
    def test_log_progress(self):
        state = new_state(10, 5, 10, 1)
        state.add_chunk("chunk_000", 5, {"0": 0})
        state.mark_submitted("chunk_000", "123")
        state.mark_completed("chunk_000")
        # Should not raise
        _log_progress(state)


class TestRunLoop(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manifest = os.path.join(FIXTURES, "sample_manifest.csv")

    def _setup_state_and_config(self):
        config = RunConfig(
            manifest=self.manifest,
            command="echo {alpha}",
            state_dir=self.tmpdir,
            chunk_size=20,
            max_concurrent=100,
            max_retries=0,
            poll_interval=1,
            slurm=SlurmConfig(),
        )

        chunks_dir = os.path.join(self.tmpdir, "chunks")
        chunks = chunk_manifest(self.manifest, ",", 20, chunks_dir)
        state = new_state(
            total_jobs=count_rows(self.manifest),
            chunk_size=20, max_concurrent=100, max_retries=0,
        )
        for chunk in chunks:
            row_mapping = {str(i): i for i in range(chunk.size)}
            state.add_chunk(chunk.chunk_id, chunk.size, row_mapping)
            script = generate_sbatch_script(chunk, config, self.tmpdir)
            spath = script_path_for_chunk(chunk, self.tmpdir)
            write_sbatch_script(script, spath)
            ensure_log_dir(chunk, self.tmpdir)

        save_state(state, self.tmpdir)
        return state, config

    @patch("slurmgrid.monitor.slurm")
    def test_run_completes(self, mock_slurm):
        state, config = self._setup_state_and_config()

        mock_slurm.sbatch.return_value = "100"

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

        final = run(state, config)
        self.assertTrue(final.is_done())

    @patch("slurmgrid.monitor.time.sleep")
    @patch("slurmgrid.monitor.slurm")
    def test_run_max_runtime(self, mock_slurm, mock_sleep):
        state, config = self._setup_state_and_config()
        config.max_runtime = 1  # Exit quickly
        config.poll_interval = 1

        mock_slurm.sbatch.return_value = "100"
        mock_slurm.sacct_query.return_value = {}  # No results yet

        final = run(state, config)
        # Should exit without completing
        self.assertFalse(final.is_done())

    @patch("slurmgrid.monitor.time.sleep")
    @patch("slurmgrid.monitor.slurm.sacct_query")
    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_run_graceful_exit(self, mock_sbatch, mock_sacct, mock_sleep):
        state, config = self._setup_state_and_config()

        mock_sbatch.return_value = "100"
        mock_sleep.side_effect = GracefulExit()

        final = run(state, config)
        self.assertFalse(final.is_done())

    @patch("slurmgrid.monitor.time.sleep")
    @patch("slurmgrid.monitor.slurm.sacct_query")
    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_sacct_error_continues(self, mock_sbatch, mock_sacct, mock_sleep):
        """When sacct fails, monitor should continue polling."""
        state, config = self._setup_state_and_config()
        config.max_runtime = 1  # Exit quickly
        config.poll_interval = 1

        mock_sbatch.return_value = "100"
        # sacct fails
        mock_sacct.side_effect = SlurmError("sacct down")

        final = run(state, config)
        # Should not crash, just exit due to max_runtime


class TestInstallSignalHandlers(unittest.TestCase):
    def test_handlers_installed(self):
        _install_signal_handlers()
        # The handler should raise GracefulExit
        handler = signal.getsignal(signal.SIGINT)
        with self.assertRaises(GracefulExit):
            handler(signal.SIGINT, None)


if __name__ == "__main__":
    unittest.main()
