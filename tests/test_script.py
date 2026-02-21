"""Tests for sbatch script generation."""

import os
import tempfile
import unittest

from slurmgrid.config import RunConfig, SlurmConfig
from slurmgrid.manifest import CHUNK_DELIMITER, CHUNK_DELIMITER_BASH, ChunkInfo
from slurmgrid.script import (
    _resolve_command,
    generate_sbatch_script,
    write_sbatch_script,
)


class TestResolveCommand(unittest.TestCase):
    def test_basic(self):
        result = _resolve_command(
            "python run.py --alpha {alpha} --beta {beta}",
            ["alpha", "beta"],
        )
        self.assertEqual(
            result,
            'python run.py --alpha "$SC_alpha" --beta "$SC_beta"',
        )

    def test_unknown_placeholder_preserved(self):
        result = _resolve_command("echo {known} {unknown}", ["known"])
        self.assertIn('"$SC_known"', result)
        self.assertIn("{unknown}", result)

    def test_no_placeholders(self):
        result = _resolve_command("echo hello", ["alpha"])
        self.assertEqual(result, "echo hello")


class TestGenerateScript(unittest.TestCase):
    def test_basic_script(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write a chunk file
            chunk_path = os.path.join(tmpdir, "chunk_000.chunk")
            header = CHUNK_DELIMITER.join(["alpha", "beta", "seed"])
            row = CHUNK_DELIMITER.join(["0.1", "1", "42"])
            with open(chunk_path, "w") as f:
                f.write(header + "\n")
                f.write(row + "\n")

            chunk = ChunkInfo(chunk_id="chunk_000", path=chunk_path, size=1)
            config = RunConfig(
                manifest="/ignored.csv",
                command="python train.py --alpha {alpha} --beta {beta}",
                state_dir=tmpdir,
                slurm=SlurmConfig(
                    partition="gpu",
                    time="01:00:00",
                    mem="4G",
                    preamble="module load python/3.10",
                ),
            )

            script = generate_sbatch_script(chunk, config, tmpdir)

            # Check key components
            self.assertIn("#!/bin/bash", script)
            self.assertIn("#SBATCH --partition=gpu", script)
            self.assertIn("#SBATCH --time=01:00:00", script)
            self.assertIn("#SBATCH --mem=4G", script)
            self.assertIn("#SBATCH --array=0-0", script)
            self.assertIn("module load python/3.10", script)
            self.assertIn(CHUNK_DELIMITER_BASH, script)
            self.assertIn('export "SC_alpha=${VALS[0]}"', script)
            self.assertIn('export "SC_beta=${VALS[1]}"', script)
            self.assertIn('export "SC_seed=${VALS[2]}"', script)
            self.assertIn('"$SC_alpha"', script)
            self.assertIn('"$SC_beta"', script)

    def test_write_script_executable(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "scripts", "test.sh")
            write_sbatch_script("#!/bin/bash\necho hi\n", path)
            self.assertTrue(os.path.isfile(path))
            self.assertTrue(os.access(path, os.X_OK))


if __name__ == "__main__":
    unittest.main()
