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
    _remove_lockfile,
    _self_resubmit,
    _submit_chunk,
    _submit_to_fill,
    _update_chunk_statuses,
    _update_single_chunk,
    _wait_for_upstream,
    _write_lockfile,
    run,
)
from slurmgrid.script import (
    ensure_log_dir,
    generate_sbatch_script,
    script_path_for_chunk,
    write_sbatch_script,
)
from slurmgrid.slurm import SlurmError, TaskState, TaskStatus
from slurmgrid.state import ChunkState, new_state, save_state, load_state

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

    def test_cancelled_task_is_retriable(self):
        """CANCELLED tasks are recorded as retriable failures."""
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
        # Cancelled task is recorded as a retriable failure
        self.assertEqual(len(state.failures), 1)
        self.assertIn("1", state.failures)
        self.assertFalse(state.failures["1"].permanently_failed)


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
        mock_slurm.squeue_query.return_value = {"100_0": "PENDING"}  # Job still alive
        mock_slurm.SlurmError = SlurmError

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


class TestWaitForUpstream(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.upstream_dir = tempfile.mkdtemp()
        self.config = RunConfig(
            manifest=os.path.join(FIXTURES, "sample_manifest.csv"),
            command="echo {alpha}",
            state_dir=self.tmpdir,
            chunk_size=5,
            max_concurrent=10,
            max_retries=0,
            poll_interval=0,
            after_run=self.upstream_dir,
            slurm=SlurmConfig(),
        )

    def _make_upstream_state(self, done=False):
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        if done:
            state.mark_submitted("chunk_000", "999")
            state.mark_completed("chunk_000")
            state.chunks["chunk_000"].completed_tasks = 5
        save_state(state, self.upstream_dir)
        return state

    @patch("slurmgrid.monitor.time.sleep")
    def test_returns_immediately_when_done(self, mock_sleep):
        self._make_upstream_state(done=True)
        _wait_for_upstream(self.config)
        mock_sleep.assert_not_called()

    @patch("slurmgrid.monitor.time.sleep")
    def test_polls_until_done(self, mock_sleep):
        upstream = self._make_upstream_state(done=False)

        call_count = [0]
        def complete_on_second_call(_):
            call_count[0] += 1
            if call_count[0] >= 1:
                upstream.mark_submitted("chunk_000", "999")
                upstream.mark_completed("chunk_000")
                save_state(upstream, self.upstream_dir)
        mock_sleep.side_effect = complete_on_second_call

        _wait_for_upstream(self.config)
        self.assertEqual(mock_sleep.call_count, 1)

    @patch("slurmgrid.monitor.time.sleep")
    def test_handles_unreadable_state(self, mock_sleep):
        """If the upstream state file is temporarily unreadable, keep polling."""
        upstream = self._make_upstream_state(done=False)

        call_count = [0]
        def fix_on_second_call(_):
            call_count[0] += 1
            if call_count[0] >= 1:
                upstream.mark_submitted("chunk_000", "999")
                upstream.mark_completed("chunk_000")
                save_state(upstream, self.upstream_dir)
        mock_sleep.side_effect = fix_on_second_call

        # Corrupt the state file temporarily, then fix it on first sleep
        import json
        state_path = os.path.join(self.upstream_dir, "state.json")
        with open(state_path, "w") as f:
            f.write("not valid json")

        # The first read will fail; the sleep side effect will write a valid state
        _wait_for_upstream(self.config)
        self.assertGreaterEqual(mock_sleep.call_count, 1)

    @patch("slurmgrid.monitor.time.sleep")
    @patch("slurmgrid.monitor.slurm")
    def test_run_waits_for_upstream_before_submitting(self, mock_slurm, mock_sleep):
        """run() should not submit any chunks until upstream is done."""
        upstream = self._make_upstream_state(done=False)

        # Set up a minimal runnable state for the current run
        state = new_state(5, 5, 10, 0)
        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")
        save_state(state, self.tmpdir)

        mock_slurm.sbatch.return_value = "101"
        mock_slurm.sacct_query.return_value = {}

        sleep_count = [0]
        def complete_upstream_then_exit(_):
            sleep_count[0] += 1
            if sleep_count[0] == 1:
                # Complete upstream on first sleep (during _wait_for_upstream)
                upstream.mark_submitted("chunk_000", "999")
                upstream.mark_completed("chunk_000")
                save_state(upstream, self.upstream_dir)
            else:
                raise GracefulExit()
        mock_sleep.side_effect = complete_upstream_then_exit

        self.config.max_runtime = None
        run(state, self.config)

        # sbatch should have been called after upstream completed
        mock_slurm.sbatch.assert_called_once()


class TestLockfile(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def test_lockfile_written_and_removed(self):
        """Lockfile is created on write and deleted on remove."""
        _write_lockfile(self.tmpdir)
        lockfile = os.path.join(self.tmpdir, "monitor.lock")
        self.assertTrue(os.path.exists(lockfile))
        content = open(lockfile).read().strip()
        host, pid = content.split(":")
        self.assertEqual(int(pid), os.getpid())
        _remove_lockfile(self.tmpdir)
        self.assertFalse(os.path.exists(lockfile))

    def test_remove_lockfile_tolerates_missing(self):
        """_remove_lockfile does not raise if file is absent."""
        _remove_lockfile(self.tmpdir)  # No error

    def test_stale_lockfile_overwritten(self):
        """A lockfile with a dead PID is silently overwritten."""
        import socket
        lockfile = os.path.join(self.tmpdir, "monitor.lock")
        # Write a lockfile with a PID that definitely doesn't exist
        with open(lockfile, "w") as f:
            f.write(f"{socket.gethostname()}:999999999\n")
        _write_lockfile(self.tmpdir)
        content = open(lockfile).read().strip()
        _, pid = content.split(":")
        self.assertEqual(int(pid), os.getpid())
        _remove_lockfile(self.tmpdir)

    @patch("slurmgrid.monitor.time.sleep")
    @patch("slurmgrid.monitor.slurm.sacct_query")
    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_lockfile_removed_on_graceful_exit(
        self, mock_sbatch, mock_sacct, mock_sleep
    ):
        """Lockfile is cleaned up even when GracefulExit is raised."""
        tmpdir = tempfile.mkdtemp()
        config = RunConfig(
            manifest=os.path.join(FIXTURES, "sample_manifest.csv"),
            command="echo {alpha}",
            state_dir=tmpdir,
            chunk_size=5,
            max_concurrent=10,
            max_retries=0,
            poll_interval=1,
            slurm=SlurmConfig(),
        )
        scripts_dir = os.path.join(tmpdir, "scripts")
        os.makedirs(scripts_dir)
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")
        save_state(state, tmpdir)

        mock_sbatch.return_value = "100"
        mock_sleep.side_effect = GracefulExit()

        run(state, config)
        self.assertFalse(os.path.exists(os.path.join(tmpdir, "monitor.lock")))


class TestSelfResubmit(unittest.TestCase):
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
            max_runtime=1,
            self_resubmit=True,
            slurm=SlurmConfig(partition="gpu", time="02:00:00", account="mylab"),
        )

    @patch("slurmgrid.monitor.slurm.sbatch_wrap")
    def test_self_resubmit_calls_sbatch_wrap(self, mock_wrap):
        """_self_resubmit passes resume command and minimal flags to sbatch_wrap."""
        mock_wrap.return_value = "42"
        _self_resubmit(self.config)
        mock_wrap.assert_called_once()
        wrap_cmd, flags = mock_wrap.call_args[0]
        self.assertIn("slurmgrid resume", wrap_cmd)
        self.assertIn(self.tmpdir, wrap_cmd)
        self.assertIn("--self-resubmit", wrap_cmd)
        self.assertIn("--partition=gpu", flags)
        self.assertIn("--time=02:00:00", flags)
        self.assertIn("--account=mylab", flags)
        self.assertIn("--cpus-per-task=1", flags)

    @patch("slurmgrid.monitor.time.sleep")
    @patch("slurmgrid.monitor.slurm.squeue_query")
    @patch("slurmgrid.monitor.slurm.sacct_query")
    @patch("slurmgrid.monitor.slurm.sbatch_wrap")
    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_run_self_resubmits_on_max_runtime(
        self, mock_sbatch, mock_wrap, mock_sacct, mock_squeue, mock_sleep
    ):
        """run() calls sbatch_wrap to resubmit when max_runtime is hit."""
        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")
        save_state(state, self.tmpdir)

        mock_sbatch.return_value = "100"
        mock_wrap.return_value = "101"
        mock_sacct.return_value = {}
        mock_squeue.return_value = {"100_0": "PENDING"}  # Job still alive

        run(state, self.config)
        mock_wrap.assert_called_once()

    @patch("slurmgrid.monitor.time.sleep")
    @patch("slurmgrid.monitor.slurm.sacct_query")
    @patch("slurmgrid.monitor.slurm.sbatch_wrap")
    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_no_resubmit_on_graceful_exit(
        self, mock_sbatch, mock_wrap, mock_sacct, mock_sleep
    ):
        """run() does NOT resubmit when killed by a signal."""
        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")
        save_state(state, self.tmpdir)

        mock_sbatch.return_value = "100"
        mock_sleep.side_effect = GracefulExit()

        run(state, self.config)
        mock_wrap.assert_not_called()


class TestSqueueFallback(unittest.TestCase):
    """Test that jobs gone from both sacct and squeue are detected as cancelled."""

    def test_missing_tasks_synthesized_as_cancelled(self):
        """When sacct returns nothing and squeue confirms job is gone,
        tasks should be recorded as retriable failures."""
        state = new_state(5, 5, 10, 1)
        state.add_chunk("chunk_000", 2, {"0": 0, "1": 1})
        state.mark_submitted("chunk_000", "100")
        config = RunConfig(
            manifest="/dev/null", command="echo hi",
            state_dir="/tmp", slurm=SlurmConfig(),
        )

        with patch("slurmgrid.monitor.slurm.sacct_query") as mock_sacct, \
             patch("slurmgrid.monitor.slurm.squeue_query") as mock_squeue:
            # sacct returns nothing
            mock_sacct.return_value = {}
            # squeue also returns nothing (job is gone)
            mock_squeue.return_value = {}

            _update_chunk_statuses(state.active_chunks(), state, config)

        # Both tasks should be recorded as failures
        self.assertEqual(state.chunks["chunk_000"].status, "partial_failure")
        self.assertEqual(len(state.failures), 2)
        for f in state.failures.values():
            self.assertFalse(f.permanently_failed)

    def test_squeue_still_alive_no_synthesis(self):
        """When sacct returns nothing but squeue shows the job is alive,
        do NOT synthesize cancellations."""
        state = new_state(5, 5, 10, 1)
        state.add_chunk("chunk_000", 2, {"0": 0, "1": 1})
        state.mark_submitted("chunk_000", "100")
        config = RunConfig(
            manifest="/dev/null", command="echo hi",
            state_dir="/tmp", slurm=SlurmConfig(),
        )

        with patch("slurmgrid.monitor.slurm.sacct_query") as mock_sacct, \
             patch("slurmgrid.monitor.slurm.squeue_query") as mock_squeue:
            mock_sacct.return_value = {}
            # squeue shows the job is still pending
            mock_squeue.return_value = {"100_0": "PENDING", "100_1": "PENDING"}

            _update_chunk_statuses(state.active_chunks(), state, config)

        # Chunk should still be submitted (waiting)
        self.assertEqual(state.chunks["chunk_000"].status, "submitted")
        self.assertEqual(len(state.failures), 0)


class TestCancelledChunkResume(unittest.TestCase):
    """Test that cancelled chunks are re-queued on resume."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manifest = os.path.join(FIXTURES, "sample_manifest.csv")

    @patch("slurmgrid.monitor.slurm")
    def test_cancelled_chunks_requeued_on_run(self, mock_slurm):
        """Cancelled chunks transition to pending and get resubmitted."""
        state = new_state(5, 5, 10, 1)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        state.mark_submitted("chunk_000", "100")
        state.mark_cancelled("chunk_000")

        # Create script file
        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")
        save_state(state, self.tmpdir)

        config = RunConfig(
            manifest=self.manifest, command="echo {alpha}",
            state_dir=self.tmpdir, chunk_size=5, max_concurrent=10,
            max_retries=1, poll_interval=1, slurm=SlurmConfig(),
        )

        # sbatch returns a new job id; sacct shows all completed
        mock_slurm.sbatch.return_value = "200"
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
        # Should have been resubmitted with a new job ID
        mock_slurm.sbatch.assert_called_once()


class TestSlurmOverridesStamped(unittest.TestCase):
    """Test that slurm_overrides on RunConfig get stamped onto ChunkState."""

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
            slurm=SlurmConfig(time="04:00:00"),
            slurm_overrides={"time": "04:00:00"},
        )

    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_submit_chunk_stamps_overrides(self, mock_sbatch):
        mock_sbatch.return_value = "100"
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")

        chunk = state.chunks["chunk_000"]
        _submit_chunk(chunk, state, self.config)
        self.assertEqual(
            state.chunks["chunk_000"].slurm_overrides,
            {"time": "04:00:00"},
        )

    @patch("slurmgrid.monitor.slurm.sbatch")
    def test_submit_chunk_no_overrides(self, mock_sbatch):
        """Without overrides, slurm_overrides stays empty."""
        mock_sbatch.return_value = "100"
        self.config.slurm_overrides = {}
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")

        chunk = state.chunks["chunk_000"]
        _submit_chunk(chunk, state, self.config)
        self.assertEqual(state.chunks["chunk_000"].slurm_overrides, {})

    @patch("slurmgrid.monitor.slurm")
    def test_cancelled_chunks_get_overrides_on_resume(self, mock_slurm):
        """Cancelled chunks should get slurm_overrides stamped when re-queued."""
        state = new_state(5, 5, 10, 1)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        state.mark_submitted("chunk_000", "100")
        state.mark_cancelled("chunk_000")

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")
        save_state(state, self.tmpdir)

        mock_slurm.sbatch.return_value = "200"
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

        final = run(state, self.config)
        self.assertEqual(
            final.chunks["chunk_000"].slurm_overrides,
            {"time": "04:00:00"},
        )


class TestHeadroomCheck(unittest.TestCase):
    """Tests for the user-task headroom check in _submit_to_fill()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = RunConfig(
            manifest=os.path.join(FIXTURES, "sample_manifest.csv"),
            command="echo {alpha}",
            state_dir=self.tmpdir,
            max_concurrent=100,
            max_retries=0,
            poll_interval=1,
            headroom=10,
            slurm=SlurmConfig(),
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("slurmgrid.monitor.get_user_job_count", return_value=95)
    @patch("slurmgrid.monitor.slurm")
    def test_headroom_blocks_submission_when_near_limit(self, mock_slurm, mock_count):
        """If user tasks + chunk size > max_concurrent - headroom, don't submit."""
        state = new_state(10, 5, 100, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        state.add_chunk("chunk_001", 5, {str(i+5): i for i in range(5)})

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        for cid in ("chunk_000", "chunk_001"):
            with open(os.path.join(scripts_dir, f"{cid}.sh"), "w") as f:
                f.write("#!/bin/bash\necho hi\n")

        mock_slurm.sbatch.return_value = "111"

        # active_tasks=0, so first chunk always submits regardless of headroom
        # but second chunk: user_task_count(95) + 5 = 100 > 100-10=90 → blocked
        _submit_to_fill(state, self.config)
        self.assertEqual(mock_slurm.sbatch.call_count, 1)

    @patch("slurmgrid.monitor.get_user_job_count", return_value=None)
    @patch("slurmgrid.monitor.slurm")
    def test_headroom_skipped_when_user_count_unknown(self, mock_slurm, mock_count):
        """If user job count is unavailable, submit anyway."""
        state = new_state(10, 5, 100, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        state.add_chunk("chunk_001", 5, {str(i+5): i for i in range(5)})

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        for cid in ("chunk_000", "chunk_001"):
            with open(os.path.join(scripts_dir, f"{cid}.sh"), "w") as f:
                f.write("#!/bin/bash\necho hi\n")

        mock_slurm.sbatch.return_value = "111"

        _submit_to_fill(state, self.config)
        self.assertEqual(mock_slurm.sbatch.call_count, 2)


class TestSubmitChunkErrorClassification(unittest.TestCase):
    """Tests for queue-limit error logging in _submit_chunk()."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.config = RunConfig(
            manifest="/dev/null",
            command="echo hi",
            state_dir=self.tmpdir,
            max_concurrent=10,
            max_retries=0,
            poll_interval=1,
            slurm=SlurmConfig(),
        )

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    @patch("slurmgrid.monitor.slurm")
    def test_queue_limit_error_logged_at_info(self, mock_slurm):
        mock_slurm.sbatch.side_effect = SlurmError(
            "sbatch failed: QOSMaxJobsPerUser limit reached"
        )
        mock_slurm.SlurmError = SlurmError

        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")

        chunk = state.chunks["chunk_000"]
        import logging
        with self.assertLogs("slurmgrid.monitor", level="INFO") as cm:
            _submit_chunk(chunk, state, self.config)

        self.assertEqual(state.chunks["chunk_000"].status, "submit_failed")
        # Should log at INFO, not ERROR
        self.assertTrue(any("deferred" in msg for msg in cm.output))
        self.assertFalse(any("ERROR" in msg and "Failed to submit" in msg
                             for msg in cm.output))

    @patch("slurmgrid.monitor.slurm")
    def test_unexpected_error_logged_at_error(self, mock_slurm):
        mock_slurm.sbatch.side_effect = SlurmError("connection refused")
        mock_slurm.SlurmError = SlurmError

        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})

        scripts_dir = os.path.join(self.tmpdir, "scripts")
        os.makedirs(scripts_dir)
        with open(os.path.join(scripts_dir, "chunk_000.sh"), "w") as f:
            f.write("#!/bin/bash\necho hi\n")

        chunk = state.chunks["chunk_000"]
        import logging
        with self.assertLogs("slurmgrid.monitor", level="INFO") as cm:
            _submit_chunk(chunk, state, self.config)

        self.assertEqual(state.chunks["chunk_000"].status, "submit_failed")
        self.assertTrue(any("ERROR" in msg and "Failed to submit" in msg
                            for msg in cm.output))


class TestInstallSignalHandlers(unittest.TestCase):
    def test_handlers_installed(self):
        _install_signal_handlers()
        # The handler should raise GracefulExit
        handler = signal.getsignal(signal.SIGINT)
        with self.assertRaises(GracefulExit):
            handler(signal.SIGINT, None)


if __name__ == "__main__":
    unittest.main()
