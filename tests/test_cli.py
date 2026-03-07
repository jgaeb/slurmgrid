"""Tests for the CLI interface."""

import logging
import os
import tempfile
import unittest
from io import StringIO
from unittest.mock import patch, MagicMock

from slurmgrid.cli import (
    _build_slurm_config,
    _print_final_report,
    _print_summary,
    cmd_cancel,
    cmd_resume,
    cmd_status,
    cmd_submit,
    main,
)
from slurmgrid.config import RunConfig, SlurmConfig
from slurmgrid.state import State, load_state, new_state, save_state

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestMain(unittest.TestCase):
    def test_version(self):
        with self.assertRaises(SystemExit) as ctx:
            main(["--version"])
        self.assertEqual(ctx.exception.code, 0)

    @patch("slurmgrid.cli.get_max_array_size", return_value=1001)
    def test_yaml_config_file(self, mock_max_array):
        """Options set in a YAML config file are applied."""
        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "run.yaml")
            with open(config_path, "w") as f:
                f.write(
                    f"manifest: {manifest}\n"
                    f"command: echo {{alpha}}\n"
                    f"state-dir: {tmpdir}\n"
                    f"chunk-size: 5\n"
                    f"dry-run: true\n"
                )
            main(["submit", f"--config={config_path}"])
            self.assertTrue(os.path.isfile(os.path.join(tmpdir, "state.json")))
            scripts = os.listdir(os.path.join(tmpdir, "scripts"))
            # 20 rows / 5 per chunk = 4 chunks
            self.assertEqual(len(scripts), 4)

    @patch("slurmgrid.cli.get_max_array_size", return_value=1001)
    def test_cli_overrides_yaml_config(self, mock_max_array):
        """CLI flags take precedence over YAML config file values."""
        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = os.path.join(tmpdir, "run.yaml")
            with open(config_path, "w") as f:
                f.write(
                    f"manifest: {manifest}\n"
                    f"command: echo {{alpha}}\n"
                    f"state-dir: {tmpdir}\n"
                    f"chunk-size: 5\n"
                    f"dry-run: true\n"
                )
            # Override chunk-size via CLI
            main(["submit", f"--config={config_path}", "--chunk-size=10"])
            scripts = os.listdir(os.path.join(tmpdir, "scripts"))
            # 20 rows / 10 per chunk = 2 chunks (CLI value wins)
            self.assertEqual(len(scripts), 2)


class TestBuildSlurmConfig(unittest.TestCase):
    def test_basic(self):
        import argparse

        args = argparse.Namespace(
            partition="gpu", time="01:00:00", mem="4G", mem_per_cpu=None,
            cpus_per_task=1, gpus=None, gres=None, account=None, qos=None,
            constraint=None, exclude=None, job_name_prefix="sc",
            extra_sbatch=[], preamble="module load cuda", preamble_file=None,
        )
        sc = _build_slurm_config(args)
        self.assertEqual(sc.partition, "gpu")
        self.assertEqual(sc.preamble, "module load cuda")

    def test_preamble_file(self):
        import argparse

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                         delete=False) as f:
            f.write("module load python/3.10\nmodule load cuda\n")
            preamble_path = f.name
        try:
            args = argparse.Namespace(
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=1, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None, job_name_prefix="sc",
                extra_sbatch=[], preamble=None, preamble_file=preamble_path,
            )
            sc = _build_slurm_config(args)
            self.assertIn("module load python/3.10", sc.preamble)
            self.assertIn("module load cuda", sc.preamble)
        finally:
            os.unlink(preamble_path)

    def test_preamble_plus_preamble_file(self):
        import argparse

        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh",
                                         delete=False) as f:
            f.write("conda activate myenv\n")
            preamble_path = f.name
        try:
            args = argparse.Namespace(
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=1, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None, job_name_prefix="sc",
                extra_sbatch=[], preamble="module load python",
                preamble_file=preamble_path,
            )
            sc = _build_slurm_config(args)
            # Both preamble and file content should be combined
            self.assertIn("module load python", sc.preamble)
            self.assertIn("conda activate myenv", sc.preamble)
        finally:
            os.unlink(preamble_path)


class TestCmdSubmitDryRun(unittest.TestCase):
    def setUp(self):
        # Suppress logging output during tests
        logging.getLogger("slurmgrid").handlers = []

    @patch("slurmgrid.cli.get_max_array_size", return_value=1001)
    def test_dry_run(self, mock_max_array):
        import argparse

        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                subcommand="submit",
                manifest=manifest,
                command="echo {alpha} {beta}",
                state_dir=tmpdir,
                delimiter=None,
                chunk_size=10,
                max_concurrent=100,
                max_retries=0,
                poll_interval=30,
                max_runtime=None,
                dry_run=True,
                partition=None,
                time=None,
                mem=None,
                mem_per_cpu=None,
                cpus_per_task=1,
                gpus=None,
                gres=None,
                account=None,
                qos=None,
                constraint=None,
                exclude=None,
                job_name_prefix="sc",
                extra_sbatch=[],
                preamble=None,
                preamble_file=None,
                no_shuffle=False,
                after_run=None,
                self_resubmit=False,
                serial_chunks=False,
                config=None,
            )
            cmd_submit(args)

            # Should have created state, chunks, and scripts
            self.assertTrue(os.path.isfile(
                os.path.join(tmpdir, "state.json"),
            ))
            self.assertTrue(os.path.isdir(
                os.path.join(tmpdir, "scripts"),
            ))
            scripts = os.listdir(os.path.join(tmpdir, "scripts"))
            self.assertEqual(len(scripts), 2)  # 20 rows / 10 = 2 chunks

    @patch("slurmgrid.cli.get_max_array_size", return_value=1001)
    def test_submit_existing_state_exits(self, mock_max_array):
        import argparse

        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            # Pre-create a state file
            state = new_state(10, 5, 10, 1)
            save_state(state, tmpdir)

            args = argparse.Namespace(
                subcommand="submit",
                manifest=manifest,
                command="echo {alpha}",
                state_dir=tmpdir,
                delimiter=None,
                chunk_size=5,
                max_concurrent=10,
                max_retries=0,
                poll_interval=30,
                max_runtime=None,
                dry_run=True,
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=1, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None,
                job_name_prefix="sc", extra_sbatch=[], preamble=None,
                preamble_file=None,
                no_shuffle=False,
                after_run=None,
                self_resubmit=False,
                serial_chunks=False,
                config=None,
            )
            with self.assertRaises(SystemExit):
                cmd_submit(args)

    @patch("slurmgrid.cli.get_max_array_size", return_value=1001)
    def test_submit_validation_error_exits(self, mock_max_array):
        import argparse

        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                subcommand="submit",
                manifest=manifest,
                command="echo {nonexistent_column}",
                state_dir=tmpdir,
                delimiter=None,
                chunk_size=5,
                max_concurrent=10,
                max_retries=0,
                poll_interval=30,
                max_runtime=None,
                dry_run=True,
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=1, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None,
                job_name_prefix="sc", extra_sbatch=[], preamble=None,
                preamble_file=None,
                no_shuffle=False,
                after_run=None,
                self_resubmit=False,
                serial_chunks=False,
                config=None,
            )
            with self.assertRaises(SystemExit):
                cmd_submit(args)

    @patch("slurmgrid.cli.run_monitor")
    @patch("slurmgrid.cli.get_max_array_size", return_value=1001)
    def test_submit_auto_chunk_size(self, mock_max_array, mock_monitor):
        """When chunk_size is None, auto-detect from MaxArraySize."""
        import argparse

        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        mock_monitor.return_value = new_state(20, 100, 100, 0)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                subcommand="submit",
                manifest=manifest,
                command="echo {alpha}",
                state_dir=tmpdir,
                delimiter=None,
                chunk_size=None,  # Auto-detect
                max_concurrent=100,
                max_retries=0,
                poll_interval=30,
                max_runtime=None,
                dry_run=False,
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=1, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None,
                job_name_prefix="sc", extra_sbatch=[], preamble=None,
                preamble_file=None,
                no_shuffle=False,
                after_run=None,
                self_resubmit=False,
                serial_chunks=False,
                config=None,
            )
            cmd_submit(args)
            mock_max_array.assert_called_once()
            mock_monitor.assert_called_once()


class TestCmdStatus(unittest.TestCase):
    def test_status_no_state(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(state_dir=tmpdir)
            with self.assertRaises(SystemExit):
                cmd_status(args)

    def test_status_ok(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            state = new_state(10, 5, 10, 1)
            state.add_chunk("chunk_000", 5, {"0": 0})
            state.mark_submitted("chunk_000", "123")
            state.mark_completed("chunk_000")
            save_state(state, tmpdir)

            args = argparse.Namespace(state_dir=tmpdir)
            # Should not raise
            cmd_status(args)


class TestCmdCancel(unittest.TestCase):
    def setUp(self):
        logging.getLogger("slurmgrid").handlers = []

    def test_cancel_no_state(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(state_dir=tmpdir)
            with self.assertRaises(SystemExit):
                cmd_cancel(args)

    @patch("slurmgrid.slurm.scancel")
    def test_cancel_no_active(self, mock_scancel):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            state = new_state(5, 5, 10, 1)
            state.add_chunk("chunk_000", 5, {"0": 0})
            state.mark_submitted("chunk_000", "123")
            state.mark_completed("chunk_000")
            save_state(state, tmpdir)

            args = argparse.Namespace(state_dir=tmpdir)
            cmd_cancel(args)
            mock_scancel.assert_not_called()

    @patch("slurmgrid.cli.slurm_mod", create=True)
    def test_cancel_active_jobs(self, *args):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            state = new_state(5, 5, 10, 1)
            state.add_chunk("chunk_000", 5, {"0": 0})
            state.mark_submitted("chunk_000", "456")
            save_state(state, tmpdir)

            with patch("slurmgrid.slurm.scancel") as mock_scancel:
                args = argparse.Namespace(state_dir=tmpdir)
                cmd_cancel(args)
                mock_scancel.assert_called_once_with(["456"])

            # Verify chunks are marked as cancelled, not completed
            reloaded = load_state(tmpdir)
            self.assertEqual(reloaded.chunks["chunk_000"].status, "cancelled")
            self.assertFalse(reloaded.is_done())


class TestCmdResume(unittest.TestCase):
    def setUp(self):
        logging.getLogger("slurmgrid").handlers = []

    @patch("slurmgrid.cli.run_monitor")
    def test_resume(self, mock_monitor):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                manifest="/dev/null", command="echo hi", state_dir=tmpdir,
                slurm=SlurmConfig(),
            )
            from slurmgrid.config import freeze_config
            freeze_config(config, tmpdir)

            state = new_state(5, 5, 10, 1)
            state.add_chunk("chunk_000", 5, {"0": 0})
            save_state(state, tmpdir)

            mock_monitor.return_value = state

            args = argparse.Namespace(
                state_dir=tmpdir, poll_interval=5, max_runtime=60,
                self_resubmit=False,
            )
            cmd_resume(args)
            mock_monitor.assert_called_once()
            # Verify overrides were applied
            call_config = mock_monitor.call_args[0][1]
            self.assertEqual(call_config.poll_interval, 5)
            self.assertEqual(call_config.max_runtime, 60)

    def test_resume_no_state(self):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            args = argparse.Namespace(
                state_dir=tmpdir, poll_interval=None, max_runtime=None,
            )
            with self.assertRaises(SystemExit):
                cmd_resume(args)


class TestPrintSummary(unittest.TestCase):
    def test_print_summary(self):
        state = new_state(10, 5, 10, 1)
        state.add_chunk("chunk_000", 5, {"0": 0})
        state.add_chunk("chunk_001", 5, {"5": 0})
        state.mark_submitted("chunk_000", "123")
        state.mark_completed("chunk_000")
        # Should not raise
        _print_summary(state)

    def test_print_summary_zero_jobs(self):
        state = new_state(0, 5, 10, 1)
        # Should not raise (tests the total_jobs == 0 branch)
        _print_summary(state)


class TestPrintFinalReport(unittest.TestCase):
    def test_no_failures(self):
        state = new_state(5, 5, 10, 1)
        state.add_chunk("chunk_000", 5, {"0": 0})
        state.mark_submitted("chunk_000", "123")
        state.mark_completed("chunk_000")
        _print_final_report(state)

    def test_with_failures(self):
        state = new_state(5, 5, 10, 0)
        state.add_chunk("chunk_000", 5, {"0": 0, "1": 1})
        state.record_failure(0, "chunk_000", 0, 1)
        state.record_failure(1, "chunk_000", 1, 1)
        _print_final_report(state)

    def test_with_many_failures(self):
        state = new_state(30, 30, 30, 0)
        state.add_chunk("chunk_000", 30,
                        {str(i): i for i in range(30)})
        for i in range(25):
            state.record_failure(i, "chunk_000", i, 1)
        _print_final_report(state)


if __name__ == "__main__":
    unittest.main()
