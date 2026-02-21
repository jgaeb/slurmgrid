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
