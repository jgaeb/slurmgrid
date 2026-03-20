"""Tests for state persistence."""

import os
import tempfile
import unittest

from slurmgrid.state import (
    ChunkState,
    FailureRecord,
    State,
    load_state,
    new_state,
    save_state,
    state_exists,
)


class TestState(unittest.TestCase):
    def _make_state(self):
        s = new_state(total_jobs=20, chunk_size=7, max_concurrent=10,
                      max_retries=2)
        s.add_chunk("chunk_000", 7, {"0": 0, "1": 1, "2": 2, "3": 3,
                                      "4": 4, "5": 5, "6": 6})
        s.add_chunk("chunk_001", 7, {"7": 0, "8": 1, "9": 2, "10": 3,
                                      "11": 4, "12": 5, "13": 6})
        s.add_chunk("chunk_002", 6, {"14": 0, "15": 1, "16": 2, "17": 3,
                                      "18": 4, "19": 5})
        return s

    def test_pending_chunks(self):
        s = self._make_state()
        self.assertEqual(len(s.pending_chunks()), 3)

    def test_active_chunks(self):
        s = self._make_state()
        self.assertEqual(len(s.active_chunks()), 0)
        s.mark_submitted("chunk_000", "123")
        self.assertEqual(len(s.active_chunks()), 1)
        self.assertEqual(s.active_job_count(), 7)

    def test_lifecycle(self):
        s = self._make_state()
        s.mark_submitted("chunk_000", "123")
        self.assertEqual(s.chunks["chunk_000"].status, "submitted")
        s.mark_running("chunk_000")
        self.assertEqual(s.chunks["chunk_000"].status, "running")
        s.mark_completed("chunk_000")
        self.assertEqual(s.chunks["chunk_000"].status, "completed")

    def test_is_done(self):
        s = self._make_state()
        self.assertFalse(s.is_done())
        for cid in list(s.chunks):
            s.mark_submitted(cid, "1")
            s.mark_completed(cid)
        self.assertTrue(s.is_done())

    def test_record_failure_and_retry(self):
        s = self._make_state()  # max_retries=2
        s.record_failure(global_index=3, chunk_id="chunk_000",
                         array_index=3, exit_code=1)
        self.assertIn("3", s.failures)
        f = s.failures["3"]
        self.assertEqual(f.retries, 0)
        self.assertFalse(f.permanently_failed)

        # First retry fails
        s.record_failure(global_index=3, chunk_id="retry_000",
                         array_index=0, exit_code=1)
        self.assertEqual(f.retries, 1)
        self.assertFalse(f.permanently_failed)

        # Second retry fails — now permanent
        s.record_failure(global_index=3, chunk_id="retry_001",
                         array_index=0, exit_code=1)
        self.assertEqual(f.retries, 2)
        self.assertTrue(f.permanently_failed)  # max_retries=2

    def test_record_failure_zero_retries(self):
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=0)
        s.record_failure(global_index=0, chunk_id="chunk_000",
                         array_index=0, exit_code=1)
        self.assertTrue(s.failures["0"].permanently_failed)

    def test_summary(self):
        s = self._make_state()
        s.mark_submitted("chunk_000", "123")
        s.chunks["chunk_000"].completed_tasks = 7
        s.mark_completed("chunk_000")
        s.mark_submitted("chunk_001", "124")
        s.chunks["chunk_001"].completed_tasks = 3  # 3 of 7 done so far
        summary = s.summary()
        self.assertEqual(summary["total_jobs"], 20)
        self.assertEqual(summary["completed_tasks"], 10)  # 7 + 3
        self.assertEqual(summary["active_tasks"], 4)  # 7 - 3 still active
        self.assertEqual(summary["pending_tasks"], 6)

    def test_mark_submit_failed(self):
        s = self._make_state()
        s.mark_submit_failed("chunk_000")
        self.assertEqual(s.chunks["chunk_000"].status, "submit_failed")
        # submit_failed chunks should appear in pending (for retry)
        pending_ids = [c.chunk_id for c in s.pending_chunks()]
        self.assertIn("chunk_000", pending_ids)

    def test_all_retries_resolved(self):
        s = self._make_state()  # max_retries=2
        # No failures -> all resolved
        self.assertTrue(s.all_retries_resolved())

        # Add a non-permanent failure
        s.record_failure(0, "chunk_000", 0, 1)
        self.assertFalse(s.all_retries_resolved())

        # First retry fails — still not permanent
        s.record_failure(0, "retry_000", 0, 1)
        self.assertFalse(s.all_retries_resolved())

        # Second retry fails — now permanent
        s.record_failure(0, "retry_001", 0, 1)
        self.assertTrue(s.all_retries_resolved())

    def test_is_done_with_failures(self):
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=0)
        s.add_chunk("chunk_000", 5, {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4})
        s.mark_submitted("chunk_000", "1")
        s.mark_partial_failure("chunk_000")
        s.record_failure(0, "chunk_000", 0, 1)  # permanently failed (max_retries=0)
        self.assertTrue(s.is_done())

    def test_is_done_with_non_permanent_failures(self):
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=2)
        s.add_chunk("chunk_000", 5, {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4})
        s.mark_submitted("chunk_000", "1")
        s.mark_partial_failure("chunk_000")
        s.record_failure(0, "chunk_000", 0, 1)  # not yet permanent
        self.assertFalse(s.is_done())

    def test_is_done_all_retries_succeeded_partial_failure_chunks_remain(self):
        """Regression: is_done() must not loop forever when all retried tasks
        succeed, failures dict is cleared, but original chunks stay in
        partial_failure status.
        """
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=2)
        s.add_chunk("chunk_000", 5, {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4})
        s.mark_submitted("chunk_000", "1")
        s.mark_partial_failure("chunk_000")
        # Record a failure then simulate successful retry by removing it
        s.record_failure(0, "chunk_000", 0, 1)
        del s.failures["0"]  # retry chunk succeeded — record cleared
        # chunk_000 is still in partial_failure, failures dict is now empty
        self.assertTrue(s.is_done())

    def test_is_done_cancelled_not_done(self):
        """Cancelled chunks are not done — they need to be resumed."""
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=0)
        s.add_chunk("chunk_000", 5, {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4})
        s.mark_submitted("chunk_000", "1")
        s.mark_cancelled("chunk_000")
        self.assertFalse(s.is_done())

    def test_active_job_count_excludes_failed_tasks(self):
        s = self._make_state()
        s.mark_submitted("chunk_000", "123")
        # 2 tasks succeeded, 3 failed (in-flight display only)
        s.chunks["chunk_000"].completed_tasks = 2
        s.chunks["chunk_000"].failed_tasks = 3
        # active = size - completed - failed = 7 - 2 - 3 = 2
        self.assertEqual(s.active_job_count(), 2)

    def test_summary_failing_active(self):
        s = self._make_state()
        s.mark_submitted("chunk_000", "123")
        s.chunks["chunk_000"].completed_tasks = 2
        s.chunks["chunk_000"].failed_tasks = 3
        summary = s.summary()
        self.assertEqual(summary["failing_active"], 3)
        self.assertEqual(summary["active_tasks"], 2)  # 7 - 2 - 3


class TestResetFailures(unittest.TestCase):
    def test_reset_clears_permanently_failed(self):
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=0)
        s.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        s.record_failure(0, "chunk_000", 0, 1)
        s.record_failure(1, "chunk_000", 1, 1)
        self.assertTrue(s.failures["0"].permanently_failed)
        self.assertTrue(s.failures["1"].permanently_failed)

        reset_count = s.reset_failures()
        self.assertEqual(reset_count, 2)
        self.assertFalse(s.failures["0"].permanently_failed)
        self.assertFalse(s.failures["1"].permanently_failed)

    def test_reset_bumps_max_retries(self):
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=1)
        s.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        # Fail once (retries=0), then fail again (retries=1, permanently_failed)
        s.record_failure(0, "chunk_000", 0, 1)
        s.record_failure(0, "retry_000", 0, 1)
        self.assertTrue(s.failures["0"].permanently_failed)
        self.assertEqual(s.failures["0"].retries, 1)

        s.reset_failures()
        # max_retries should be bumped to at least max_retries_seen + 1 = 2
        self.assertEqual(s.max_retries, 2)

    def test_reset_no_failures_returns_zero(self):
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=0)
        self.assertEqual(s.reset_failures(), 0)
        self.assertEqual(s.max_retries, 0)

    def test_reset_skips_non_permanent_failures(self):
        s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                      max_retries=2)
        s.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
        s.record_failure(0, "chunk_000", 0, 1)  # retries=0, not permanent
        self.assertFalse(s.failures["0"].permanently_failed)

        reset_count = s.reset_failures()
        self.assertEqual(reset_count, 0)  # Nothing was permanently_failed


class TestChunkStateOverrides(unittest.TestCase):
    def test_slurm_overrides_default_empty(self):
        cs = ChunkState(chunk_id="chunk_000")
        self.assertEqual(cs.slurm_overrides, {})

    def test_slurm_overrides_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                          max_retries=0)
            s.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
            s.chunks["chunk_000"].slurm_overrides = {"time": "04:00:00"}
            save_state(s, tmpdir)

            loaded = load_state(tmpdir)
            self.assertEqual(
                loaded.chunks["chunk_000"].slurm_overrides,
                {"time": "04:00:00"},
            )


class TestStatePersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            s = new_state(total_jobs=10, chunk_size=5, max_concurrent=10,
                          max_retries=1)
            s.add_chunk("chunk_000", 5, {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4})
            s.mark_submitted("chunk_000", "999")
            s.record_failure(2, "chunk_000", 2, 1)

            save_state(s, tmpdir)
            self.assertTrue(state_exists(tmpdir))

            loaded = load_state(tmpdir)
            self.assertEqual(loaded.total_jobs, 10)
            self.assertEqual(loaded.chunks["chunk_000"].slurm_job_id, "999")
            self.assertIn("2", loaded.failures)
            self.assertEqual(loaded.failures["2"].exit_code, 1)

    def test_state_exists_false(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            self.assertFalse(state_exists(tmpdir))

    def test_deserialize_bad_version(self):
        import json
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "state.json")
            data = {
                "version": 99,
                "created_at": "", "updated_at": "",
                "total_jobs": 0, "chunk_size": 0,
                "max_concurrent": 0, "max_retries": 0,
            }
            with open(path, "w") as f:
                json.dump(data, f)
            with self.assertRaises(ValueError) as ctx:
                load_state(tmpdir)
            self.assertIn("version", str(ctx.exception))

    def test_save_state_atomic(self):
        """Verify save_state writes atomically (no partial files left)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            s = new_state(total_jobs=5, chunk_size=5, max_concurrent=5,
                          max_retries=0)
            save_state(s, tmpdir)
            # Should have state.json, no .tmp files
            files = os.listdir(tmpdir)
            self.assertIn("state.json", files)
            self.assertFalse(any(f.endswith(".tmp") for f in files))


if __name__ == "__main__":
    unittest.main()
