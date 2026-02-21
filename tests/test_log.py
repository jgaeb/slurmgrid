"""Tests for logging configuration."""

import logging
import os
import tempfile
import unittest

from slurmgrid.log import setup_logging


class TestSetupLogging(unittest.TestCase):
    def setUp(self):
        # Clear handlers from any previous test
        root = logging.getLogger("slurmgrid")
        root.handlers = []

    def test_creates_directory_and_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_dir = os.path.join(tmpdir, "sub", "dir")
            setup_logging(log_dir)
            self.assertTrue(os.path.isdir(log_dir))
            self.assertTrue(os.path.isfile(
                os.path.join(log_dir, "slurmgrid.log"),
            ))

    def test_creates_two_handlers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(tmpdir)
            root = logging.getLogger("slurmgrid")
            self.assertEqual(len(root.handlers), 2)
            handler_types = {type(h) for h in root.handlers}
            self.assertIn(logging.FileHandler, handler_types)
            self.assertIn(logging.StreamHandler, handler_types)

    def test_no_duplicate_handlers(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(tmpdir)
            setup_logging(tmpdir)  # Second call
            root = logging.getLogger("slurmgrid")
            self.assertEqual(len(root.handlers), 2)

    def test_verbose_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(tmpdir, verbose=True)
            root = logging.getLogger("slurmgrid")
            stream_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            self.assertEqual(len(stream_handlers), 1)
            self.assertEqual(stream_handlers[0].level, logging.DEBUG)

    def test_non_verbose_mode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(tmpdir, verbose=False)
            root = logging.getLogger("slurmgrid")
            stream_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.StreamHandler)
                and not isinstance(h, logging.FileHandler)
            ]
            self.assertEqual(len(stream_handlers), 1)
            self.assertEqual(stream_handlers[0].level, logging.INFO)

    def test_file_handler_level(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            setup_logging(tmpdir)
            root = logging.getLogger("slurmgrid")
            file_handlers = [
                h for h in root.handlers
                if isinstance(h, logging.FileHandler)
            ]
            self.assertEqual(len(file_handlers), 1)
            self.assertEqual(file_handlers[0].level, logging.DEBUG)


if __name__ == "__main__":
    unittest.main()
