"""Tests for Slurm command wrappers."""

import subprocess
import unittest
from unittest.mock import patch, MagicMock

from slurmgrid.slurm import (
    SlurmError,
    TaskState,
    TaskStatus,
    _run_command,
    get_max_array_size,
    get_user_job_count,
    sacct_query,
    sbatch,
    scancel,
    squeue_query,
)


class TestTaskState(unittest.TestCase):
    def test_from_sacct_basic(self):
        self.assertEqual(TaskState.from_sacct("COMPLETED"), TaskState.COMPLETED)
        self.assertEqual(TaskState.from_sacct("FAILED"), TaskState.FAILED)
        self.assertEqual(TaskState.from_sacct("PENDING"), TaskState.PENDING)
        self.assertEqual(TaskState.from_sacct("RUNNING"), TaskState.RUNNING)

    def test_from_sacct_with_suffix(self):
        self.assertEqual(TaskState.from_sacct("CANCELLED+"), TaskState.CANCELLED)

    def test_from_sacct_with_extra_words(self):
        self.assertEqual(
            TaskState.from_sacct("CANCELLED by 12345"), TaskState.CANCELLED,
        )

    def test_from_sacct_unknown(self):
        self.assertEqual(TaskState.from_sacct("REQUEUED"), TaskState.UNKNOWN)

    def test_is_terminal(self):
        self.assertTrue(TaskState.COMPLETED.is_terminal)
        self.assertTrue(TaskState.FAILED.is_terminal)
        self.assertTrue(TaskState.TIMEOUT.is_terminal)
        self.assertTrue(TaskState.CANCELLED.is_terminal)
        self.assertTrue(TaskState.NODE_FAIL.is_terminal)
        self.assertTrue(TaskState.OUT_OF_MEMORY.is_terminal)
        self.assertFalse(TaskState.PENDING.is_terminal)
        self.assertFalse(TaskState.RUNNING.is_terminal)
        self.assertFalse(TaskState.UNKNOWN.is_terminal)

    def test_is_success(self):
        self.assertTrue(TaskState.COMPLETED.is_success)
        self.assertFalse(TaskState.FAILED.is_success)

    def test_is_retriable_failure(self):
        self.assertTrue(TaskState.FAILED.is_retriable_failure)
        self.assertTrue(TaskState.TIMEOUT.is_retriable_failure)
        self.assertTrue(TaskState.NODE_FAIL.is_retriable_failure)
        self.assertTrue(TaskState.OUT_OF_MEMORY.is_retriable_failure)
        self.assertFalse(TaskState.COMPLETED.is_retriable_failure)
        self.assertTrue(TaskState.CANCELLED.is_retriable_failure)


class TestRunCommand(unittest.TestCase):
    @patch("slurmgrid.slurm.subprocess.run")
    def test_success_first_try(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=0, stdout="ok", stderr="",
        )
        result = _run_command(["cmd"], "test cmd", retries=3)
        self.assertEqual(result.stdout, "ok")
        self.assertEqual(mock_run.call_count, 1)

    @patch("slurmgrid.slurm.time.sleep")
    @patch("slurmgrid.slurm.subprocess.run")
    def test_retry_on_failure(self, mock_run, mock_sleep):
        # Fail twice, succeed on third try
        mock_run.side_effect = [
            subprocess.CompletedProcess(
                args=["cmd"], returncode=1, stdout="", stderr="err1",
            ),
            subprocess.CompletedProcess(
                args=["cmd"], returncode=1, stdout="", stderr="err2",
            ),
            subprocess.CompletedProcess(
                args=["cmd"], returncode=0, stdout="ok", stderr="",
            ),
        ]
        result = _run_command(["cmd"], "test cmd", retries=3)
        self.assertEqual(result.stdout, "ok")
        self.assertEqual(mock_run.call_count, 3)
        self.assertEqual(mock_sleep.call_count, 2)

    @patch("slurmgrid.slurm.time.sleep")
    @patch("slurmgrid.slurm.subprocess.run")
    def test_all_retries_exhausted(self, mock_run, mock_sleep):
        mock_run.return_value = subprocess.CompletedProcess(
            args=["cmd"], returncode=1, stdout="", stderr="persistent error",
        )
        with self.assertRaises(SlurmError) as ctx:
            _run_command(["cmd"], "test cmd", retries=2)
        self.assertIn("persistent error", str(ctx.exception))
        self.assertEqual(mock_run.call_count, 2)

    @patch("slurmgrid.slurm.time.sleep")
    @patch("slurmgrid.slurm.subprocess.run")
    def test_timeout_with_retry(self, mock_run, mock_sleep):
        mock_run.side_effect = [
            subprocess.TimeoutExpired(cmd=["cmd"], timeout=60),
            subprocess.CompletedProcess(
                args=["cmd"], returncode=0, stdout="ok", stderr="",
            ),
        ]
        result = _run_command(["cmd"], "test cmd", retries=2)
        self.assertEqual(result.stdout, "ok")

    @patch("slurmgrid.slurm.time.sleep")
    @patch("slurmgrid.slurm.subprocess.run")
    def test_timeout_all_retries(self, mock_run, mock_sleep):
        mock_run.side_effect = subprocess.TimeoutExpired(cmd=["cmd"], timeout=60)
        with self.assertRaises(SlurmError) as ctx:
            _run_command(["cmd"], "test cmd", retries=2)
        self.assertIn("timed out", str(ctx.exception))


class TestSbatch(unittest.TestCase):
    @patch("slurmgrid.slurm._run_command")
    def test_parse_job_id(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="12345\n", stderr="",
        )
        self.assertEqual(sbatch("/path/to/script.sh"), "12345")

    @patch("slurmgrid.slurm._run_command")
    def test_parse_job_id_with_cluster(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="12345;cluster\n", stderr="",
        )
        self.assertEqual(sbatch("/path/to/script.sh"), "12345")

    @patch("slurmgrid.slurm._run_command")
    def test_bad_output(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="not_a_number\n", stderr="",
        )
        with self.assertRaises(SlurmError):
            sbatch("/path/to/script.sh")


class TestSacctQuery(unittest.TestCase):
    @patch("slurmgrid.slurm._run_command")
    def test_parse_array_tasks(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout=(
                "100|COMPLETED|0:0\n"
                "100_0|COMPLETED|0:0\n"
                "100_0.batch|COMPLETED|0:0\n"
                "100_1|FAILED|1:0\n"
                "100_[2-4]|PENDING|0:0\n"
            ),
            stderr="",
        )
        result = sacct_query(["100"])
        # Should only include 100_0 and 100_1 (not parent, batch, or range)
        self.assertEqual(len(result), 2)
        self.assertEqual(result["100_0"].state, TaskState.COMPLETED)
        self.assertEqual(result["100_0"].exit_code, 0)
        self.assertEqual(result["100_1"].state, TaskState.FAILED)
        self.assertEqual(result["100_1"].exit_code, 1)

    @patch("slurmgrid.slurm._run_command")
    def test_parse_extern_step(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="100_0.extern|COMPLETED|0:0\n100_0|COMPLETED|0:0\n",
            stderr="",
        )
        result = sacct_query(["100"])
        self.assertEqual(len(result), 1)
        self.assertIn("100_0", result)

    @patch("slurmgrid.slurm._run_command")
    def test_bad_exit_code(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="100_0|COMPLETED|bad\n",
            stderr="",
        )
        result = sacct_query(["100"])
        self.assertEqual(result["100_0"].exit_code, -1)

    def test_empty_input(self):
        result = sacct_query([])
        self.assertEqual(result, {})

    @patch("slurmgrid.slurm._run_command")
    def test_short_line_skipped(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="short_line\n100_0|COMPLETED|0:0\n",
            stderr="",
        )
        result = sacct_query(["100"])
        self.assertEqual(len(result), 1)


class TestSqueuQuery(unittest.TestCase):
    @patch("slurmgrid.slurm._run_command")
    def test_parse_output(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="100_0|RUNNING\n100_1|PENDING\n",
            stderr="",
        )
        result = squeue_query(["100"])
        self.assertEqual(result["100_0"], "RUNNING")
        self.assertEqual(result["100_1"], "PENDING")

    def test_empty_input(self):
        result = squeue_query([])
        self.assertEqual(result, {})

    @patch("slurmgrid.slurm._run_command")
    def test_short_line_skipped(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="short\n100_0|RUNNING\n",
            stderr="",
        )
        result = squeue_query(["100"])
        self.assertEqual(len(result), 1)


class TestScancel(unittest.TestCase):
    @patch("slurmgrid.slurm._run_command")
    def test_basic(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0, stdout="", stderr="",
        )
        scancel(["100", "101"])
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        self.assertEqual(args, ["scancel", "100", "101"])

    @patch("slurmgrid.slurm._run_command")
    def test_empty_input(self, mock_run):
        scancel([])
        mock_run.assert_not_called()


class TestGetMaxArraySize(unittest.TestCase):
    @patch("slurmgrid.slurm._run_command")
    def test_parse_max_array_size(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="MaxArraySize          = 4001\nOtherConfig = foo\n",
            stderr="",
        )
        self.assertEqual(get_max_array_size(), 4001)

    @patch("slurmgrid.slurm._run_command")
    def test_default_on_failure(self, mock_run):
        mock_run.side_effect = SlurmError("scontrol failed")
        self.assertEqual(get_max_array_size(), 1001)

    @patch("slurmgrid.slurm._run_command")
    def test_default_when_not_found(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="SomeOtherConfig = 123\n",
            stderr="",
        )
        self.assertEqual(get_max_array_size(), 1001)


class TestGetUserJobCount(unittest.TestCase):
    @patch("slurmgrid.slurm._run_command")
    def test_counts_lines(self, mock_run):
        mock_run.return_value = subprocess.CompletedProcess(
            args=[], returncode=0,
            stdout="12345_0 RUNNING\n12345_1 RUNNING\n12346 PENDING\n",
            stderr="",
        )
        with patch.dict("os.environ", {"USER": "testuser"}):
            count = get_user_job_count()
        self.assertEqual(count, 3)

    @patch("slurmgrid.slurm._run_command")
    def test_returns_none_on_slurm_error(self, mock_run):
        mock_run.side_effect = SlurmError("squeue failed")
        with patch.dict("os.environ", {"USER": "testuser"}):
            count = get_user_job_count()
        self.assertIsNone(count)

    @patch("slurmgrid.slurm._run_command")
    def test_returns_none_on_os_error(self, mock_run):
        mock_run.side_effect = FileNotFoundError("squeue not found")
        with patch.dict("os.environ", {"USER": "testuser"}):
            count = get_user_job_count()
        self.assertIsNone(count)

    def test_returns_none_when_no_user_env(self):
        env = {k: v for k, v in __import__("os").environ.items()
               if k not in ("USER", "LOGNAME")}
        with patch.dict("os.environ", env, clear=True):
            count = get_user_job_count()
        self.assertIsNone(count)


if __name__ == "__main__":
    unittest.main()
