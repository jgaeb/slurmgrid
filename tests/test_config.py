"""Tests for configuration dataclasses and validation."""

import json
import os
import tempfile
import unittest

from slurmgrid.config import (
    RunConfig,
    SlurmConfig,
    freeze_config,
    load_config,
)


class TestSlurmConfig(unittest.TestCase):
    def test_sbatch_directives_minimal(self):
        sc = SlurmConfig()
        lines = sc.sbatch_directives("test_job", "0-9", "out.log", "err.log")
        self.assertIn("#SBATCH --job-name=test_job", lines)
        self.assertIn("#SBATCH --array=0-9", lines)
        self.assertIn("#SBATCH --output=out.log", lines)
        self.assertIn("#SBATCH --error=err.log", lines)
        # No partition line since it's None
        self.assertFalse(any("partition" in l for l in lines))

    def test_sbatch_directives_full(self):
        sc = SlurmConfig(
            partition="gpu",
            time="02:00:00",
            mem="8G",
            cpus_per_task=4,
            account="myacct",
            extra_sbatch_flags=["--nodelist=node01"],
        )
        lines = sc.sbatch_directives("j", "0-99", "o", "e")
        self.assertIn("#SBATCH --partition=gpu", lines)
        self.assertIn("#SBATCH --time=02:00:00", lines)
        self.assertIn("#SBATCH --mem=8G", lines)
        self.assertIn("#SBATCH --cpus-per-task=4", lines)
        self.assertIn("#SBATCH --account=myacct", lines)
        self.assertIn("#SBATCH --nodelist=node01", lines)

    def test_sbatch_directives_gpus(self):
        sc = SlurmConfig(gpus="a100:2")
        lines = sc.sbatch_directives("j", "0-0", "o", "e")
        self.assertIn("#SBATCH --gpus=a100:2", lines)

    def test_sbatch_directives_gres(self):
        sc = SlurmConfig(gres="gpu:4")
        lines = sc.sbatch_directives("j", "0-0", "o", "e")
        self.assertIn("#SBATCH --gres=gpu:4", lines)

    def test_sbatch_directives_qos(self):
        sc = SlurmConfig(qos="high")
        lines = sc.sbatch_directives("j", "0-0", "o", "e")
        self.assertIn("#SBATCH --qos=high", lines)

    def test_sbatch_directives_constraint(self):
        sc = SlurmConfig(constraint="v100")
        lines = sc.sbatch_directives("j", "0-0", "o", "e")
        self.assertIn("#SBATCH --constraint=v100", lines)

    def test_sbatch_directives_exclude(self):
        sc = SlurmConfig(exclude="node[01-03]")
        lines = sc.sbatch_directives("j", "0-0", "o", "e")
        self.assertIn("#SBATCH --exclude=node[01-03]", lines)

    def test_sbatch_directives_mem_per_cpu(self):
        sc = SlurmConfig(mem_per_cpu="2G")
        lines = sc.sbatch_directives("j", "0-0", "o", "e")
        self.assertIn("#SBATCH --mem-per-cpu=2G", lines)


class TestRunConfig(unittest.TestCase):
    def test_placeholders(self):
        rc = RunConfig(
            manifest="x.csv",
            command="python train.py --alpha {alpha} --beta {beta}",
        )
        self.assertEqual(set(rc.placeholders()), {"alpha", "beta"})

    def test_resolved_delimiter_csv(self):
        rc = RunConfig(manifest="data.csv", command="echo {x}")
        self.assertEqual(rc.resolved_delimiter, ",")

    def test_resolved_delimiter_tsv(self):
        rc = RunConfig(manifest="data.tsv", command="echo {x}")
        self.assertEqual(rc.resolved_delimiter, "\t")

    def test_resolved_delimiter_explicit(self):
        rc = RunConfig(manifest="data.tsv", command="echo {x}", delimiter=",")
        self.assertEqual(rc.resolved_delimiter, ",")

    def test_validate_missing_columns(self):
        rc = RunConfig(manifest="/dev/null", command="echo {x} {y}")
        errors = rc.validate(["x"])  # missing y
        self.assertTrue(any("y" in e for e in errors))

    def test_validate_bad_column_names(self):
        rc = RunConfig(manifest="/dev/null", command="echo {x}")
        errors = rc.validate(["x", "bad column", "3starts_with_digit"])
        self.assertTrue(any("bad column" in e for e in errors))
        self.assertTrue(any("3starts_with_digit" in e for e in errors))

    def test_validate_ok(self):
        # Create a temp manifest so the file exists
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("x,y\n1,2\n")
            path = f.name
        try:
            rc = RunConfig(manifest=path, command="echo {x} {y}")
            errors = rc.validate(["x", "y"])
            self.assertEqual(errors, [])
        finally:
            os.unlink(path)

    def test_validate_max_concurrent_invalid(self):
        rc = RunConfig(manifest="/dev/null", command="echo hi",
                       max_concurrent=0)
        errors = rc.validate([])
        self.assertTrue(any("max_concurrent" in e for e in errors))

    def test_validate_max_retries_invalid(self):
        rc = RunConfig(manifest="/dev/null", command="echo hi",
                       max_retries=-1)
        errors = rc.validate([])
        self.assertTrue(any("max_retries" in e for e in errors))

    def test_validate_poll_interval_invalid(self):
        rc = RunConfig(manifest="/dev/null", command="echo hi",
                       poll_interval=0)
        errors = rc.validate([])
        self.assertTrue(any("poll_interval" in e for e in errors))

    def test_validate_chunk_size_invalid(self):
        rc = RunConfig(manifest="/dev/null", command="echo hi",
                       chunk_size=0)
        errors = rc.validate([])
        self.assertTrue(any("chunk_size" in e for e in errors))


class TestFreezeAndLoad(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            original = RunConfig(
                manifest="/some/path.csv",
                command="python run.py --a {a}",
                state_dir=tmpdir,
                max_concurrent=5000,
                max_retries=2,
                poll_interval=15,
                slurm=SlurmConfig(
                    partition="gpu",
                    time="01:00:00",
                    preamble="module load python",
                ),
            )
            freeze_config(original, tmpdir)
            loaded = load_config(tmpdir)
            self.assertEqual(loaded.manifest, original.manifest)
            self.assertEqual(loaded.command, original.command)
            self.assertEqual(loaded.max_concurrent, original.max_concurrent)
            self.assertEqual(loaded.slurm.partition, "gpu")
            self.assertEqual(loaded.slurm.preamble, "module load python")


if __name__ == "__main__":
    unittest.main()
