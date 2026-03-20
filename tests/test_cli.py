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
    cmd_failures,
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
                self_resubmit=False, reset_failures=False,
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=None, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None,
                job_name_prefix=None, extra_sbatch=[], preamble=None,
                preamble_file=None,
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
                self_resubmit=False, reset_failures=False,
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=None, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None,
                job_name_prefix=None, extra_sbatch=[], preamble=None,
                preamble_file=None,
            )
            with self.assertRaises(SystemExit):
                cmd_resume(args)


class TestCmdResumeResetFailures(unittest.TestCase):
    def setUp(self):
        logging.getLogger("slurmgrid").handlers = []

    @patch("slurmgrid.cli.run_monitor")
    def test_reset_failures_clears_and_bumps(self, mock_monitor):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                manifest="/dev/null", command="echo hi", state_dir=tmpdir,
                max_retries=0, slurm=SlurmConfig(),
            )
            from slurmgrid.config import freeze_config
            freeze_config(config, tmpdir)

            state = new_state(5, 5, 10, 0)
            state.add_chunk("chunk_000", 5, {"0": 0, "1": 1})
            state.mark_submitted("chunk_000", "123")
            state.mark_partial_failure("chunk_000")
            state.record_failure(0, "chunk_000", 0, 1)
            state.record_failure(1, "chunk_000", 1, 1)
            save_state(state, tmpdir)

            mock_monitor.return_value = state

            args = argparse.Namespace(
                state_dir=tmpdir, poll_interval=None, max_runtime=None,
                self_resubmit=False, reset_failures=True,
                partition=None, time=None, mem=None, mem_per_cpu=None,
                cpus_per_task=None, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None,
                job_name_prefix=None, extra_sbatch=[], preamble=None,
                preamble_file=None,
            )
            cmd_resume(args)

            # Verify the config passed to monitor has bumped max_retries
            call_config = mock_monitor.call_args[0][1]
            self.assertGreaterEqual(call_config.max_retries, 1)

            # Verify the state passed to monitor has failures reset
            call_state = mock_monitor.call_args[0][0]
            for f in call_state.failures.values():
                self.assertFalse(f.permanently_failed)


class TestCmdResumeSlurmOverrides(unittest.TestCase):
    def setUp(self):
        logging.getLogger("slurmgrid").handlers = []

    @patch("slurmgrid.cli.run_monitor")
    def test_slurm_overrides_applied(self, mock_monitor):
        import argparse

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RunConfig(
                manifest="/dev/null", command="echo hi", state_dir=tmpdir,
                slurm=SlurmConfig(time="01:00:00", partition="cpu"),
            )
            from slurmgrid.config import freeze_config
            freeze_config(config, tmpdir)

            state = new_state(5, 5, 10, 1)
            state.add_chunk("chunk_000", 5, {"0": 0})
            save_state(state, tmpdir)

            mock_monitor.return_value = state

            args = argparse.Namespace(
                state_dir=tmpdir, poll_interval=None, max_runtime=None,
                self_resubmit=False, reset_failures=False,
                partition=None, time="04:00:00", mem=None, mem_per_cpu=None,
                cpus_per_task=None, gpus=None, gres=None, account=None,
                qos=None, constraint=None, exclude=None,
                job_name_prefix=None, extra_sbatch=[], preamble=None,
                preamble_file=None,
            )
            cmd_resume(args)

            call_config = mock_monitor.call_args[0][1]
            self.assertEqual(call_config.slurm.time, "04:00:00")
            self.assertEqual(call_config.slurm_overrides, {"time": "04:00:00"})
            # Partition was not overridden, so it should remain unchanged
            self.assertEqual(call_config.slurm.partition, "cpu")


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


class TestCmdFailures(unittest.TestCase):
    def setUp(self):
        logging.getLogger("slurmgrid").handlers = []

    def _make_args(self, state_dir, tail=20, paths_only=False,
                   permanently_failed_only=False):
        import argparse
        return argparse.Namespace(
            state_dir=state_dir,
            tail=tail,
            paths_only=paths_only,
            permanently_failed_only=permanently_failed_only,
        )

    def test_no_state_exits(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            args = self._make_args(tmpdir)
            with self.assertRaises(SystemExit):
                cmd_failures(args)

    def test_no_failures_prints_message(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = new_state(5, 5, 10, 0)
            state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
            state.mark_submitted("chunk_000", "123")
            state.mark_completed("chunk_000")
            save_state(state, tmpdir)

            args = self._make_args(tmpdir)
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                cmd_failures(args)
            self.assertIn("No failures", mock_out.getvalue())

    def test_shows_failures_with_manifest(self):
        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            from slurmgrid.config import RunConfig, SlurmConfig, freeze_config
            config = RunConfig(
                manifest=manifest, command="echo {alpha}",
                state_dir=tmpdir, max_retries=0, slurm=SlurmConfig(),
            )
            freeze_config(config, tmpdir)

            state = new_state(5, 5, 10, 0)
            state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
            state.mark_submitted("chunk_000", "456")
            state.mark_partial_failure("chunk_000")
            state.record_failure(0, "chunk_000", 0, 1)
            save_state(state, tmpdir)

            args = self._make_args(tmpdir, tail=0)
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                cmd_failures(args)
            output = mock_out.getvalue()
            self.assertIn("Row 0", output)
            self.assertIn("exit=1", output)

    def test_permanently_failed_only_filter(self):
        manifest = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            from slurmgrid.config import RunConfig, SlurmConfig, freeze_config
            config = RunConfig(
                manifest=manifest, command="echo {alpha}",
                state_dir=tmpdir, max_retries=2, slurm=SlurmConfig(),
            )
            freeze_config(config, tmpdir)

            state = new_state(5, 5, 10, 2)
            state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
            state.mark_submitted("chunk_000", "789")
            state.mark_partial_failure("chunk_000")
            state.record_failure(0, "chunk_000", 0, 1)  # not permanent
            state.record_failure(1, "chunk_000", 1, 1)  # not permanent
            # Make row 1 permanently failed
            state.failures["1"].permanently_failed = True
            save_state(state, tmpdir)

            args = self._make_args(tmpdir, tail=0, permanently_failed_only=True)
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                cmd_failures(args)
            output = mock_out.getvalue()
            self.assertIn("Row 1", output)
            self.assertNotIn("Row 0", output)

    def test_paths_only_skips_log_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = new_state(5, 5, 10, 0)
            state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
            state.mark_submitted("chunk_000", "999")
            state.mark_partial_failure("chunk_000")
            state.record_failure(0, "chunk_000", 0, 2)
            save_state(state, tmpdir)

            args = self._make_args(tmpdir, paths_only=True)
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                cmd_failures(args)
            output = mock_out.getvalue()
            self.assertIn("OUT:", output)
            self.assertIn("ERR:", output)
            self.assertNotIn("last", output)

    def test_tail_shows_err_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            state = new_state(5, 5, 10, 0)
            state.add_chunk("chunk_000", 5, {str(i): i for i in range(5)})
            state.mark_submitted("chunk_000", "111")
            state.mark_partial_failure("chunk_000")
            state.record_failure(0, "chunk_000", 0, 1)
            save_state(state, tmpdir)

            # Write a fake .err file
            log_dir = os.path.join(tmpdir, "logs", "chunk_000")
            os.makedirs(log_dir, exist_ok=True)
            err_path = os.path.join(log_dir, "slurm-111_0.err")
            with open(err_path, "w") as f:
                f.write("line1\nline2\nline3\n")

            args = self._make_args(tmpdir, tail=2)
            from io import StringIO
            with patch("sys.stdout", new_callable=StringIO) as mock_out:
                cmd_failures(args)
            output = mock_out.getvalue()
            self.assertIn("line2", output)
            self.assertIn("line3", output)
            self.assertNotIn("line1", output)


if __name__ == "__main__":
    unittest.main()
