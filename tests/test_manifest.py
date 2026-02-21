"""Tests for manifest loading, validation, and chunking."""

import os
import tempfile
import unittest

from slurmgrid.manifest import (
    CHUNK_DELIMITER,
    ChunkInfo,
    chunk_manifest,
    count_rows,
    load_headers,
    validate_manifest,
)

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")


class TestLoadHeaders(unittest.TestCase):
    def test_csv(self):
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        headers = load_headers(path, ",")
        self.assertEqual(headers, ["alpha", "beta", "seed"])

    def test_tsv(self):
        path = os.path.join(FIXTURES, "sample_manifest.tsv")
        headers = load_headers(path, "\t")
        self.assertEqual(headers, ["alpha", "beta", "seed"])


class TestCountRows(unittest.TestCase):
    def test_csv(self):
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        self.assertEqual(count_rows(path), 20)

    def test_tsv(self):
        path = os.path.join(FIXTURES, "sample_manifest.tsv")
        self.assertEqual(count_rows(path), 3)


class TestValidateManifest(unittest.TestCase):
    def test_valid(self):
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        errors = validate_manifest(path, ",", ["alpha", "beta"])
        self.assertEqual(errors, [])

    def test_duplicate_columns(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("a,b,a\n1,2,3\n")
            f.flush()
            errors = validate_manifest(f.name, ",", ["a"])
        os.unlink(f.name)
        self.assertTrue(any("Duplicate" in e for e in errors))

    def test_empty_manifest(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("")
            f.flush()
            try:
                # Empty file will fail on csv.reader next() call
                with self.assertRaises(StopIteration):
                    validate_manifest(f.name, ",", [])
            finally:
                os.unlink(f.name)

    def test_unit_separator_in_values(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("a,b\nfoo,bar\x1fbaz\n")
            f.flush()
            try:
                errors = validate_manifest(f.name, ",", ["a"])
                self.assertTrue(any("unit separator" in e.lower()
                                    for e in errors))
            finally:
                os.unlink(f.name)

    def test_many_unit_separator_errors_truncated(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv",
                                         delete=False) as f:
            f.write("a\n")
            for _ in range(20):
                f.write("val\x1fbad\n")
            f.flush()
            try:
                errors = validate_manifest(f.name, ",", ["a"])
                # Should stop at 10 + the "too many" message
                self.assertTrue(any("too many" in e.lower() for e in errors))
                self.assertLessEqual(len(errors), 11)
            finally:
                os.unlink(f.name)


class TestChunkManifest(unittest.TestCase):
    def test_basic_chunking(self):
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 7, tmpdir, shuffle=False)
            # 20 rows, chunk_size=7 -> 3 chunks (7, 7, 6)
            self.assertEqual(len(chunks), 3)
            self.assertEqual(chunks[0].size, 7)
            self.assertEqual(chunks[1].size, 7)
            self.assertEqual(chunks[2].size, 6)

            # Each chunk file should have header + data rows
            for chunk in chunks:
                with open(chunk.path) as f:
                    lines = f.readlines()
                # header + chunk.size data lines
                self.assertEqual(len(lines), chunk.size + 1)
                # Should use unit separator
                self.assertIn(CHUNK_DELIMITER, lines[0])

    def test_single_chunk(self):
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 100, tmpdir)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].size, 20)

    def test_exact_fit(self):
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 10, tmpdir)
            self.assertEqual(len(chunks), 2)
            self.assertEqual(chunks[0].size, 10)
            self.assertEqual(chunks[1].size, 10)

    def test_tsv_input_converted(self):
        """TSV input should be converted to unit-separator chunks."""
        path = os.path.join(FIXTURES, "sample_manifest.tsv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, "\t", 10, tmpdir)
            self.assertEqual(len(chunks), 1)
            self.assertEqual(chunks[0].size, 3)
            with open(chunks[0].path) as f:
                header = f.readline()
            self.assertIn(CHUNK_DELIMITER, header)
            self.assertNotIn("\t", header)

    def test_chunk_file_extension(self):
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 100, tmpdir)
            self.assertTrue(chunks[0].path.endswith(".chunk"))


    def test_no_shuffle_preserves_order(self):
        """Without shuffle, original_indices should be sequential."""
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 7, tmpdir, shuffle=False)
            all_indices = []
            for chunk in chunks:
                all_indices.extend(chunk.original_indices)
            self.assertEqual(all_indices, list(range(20)))

    def test_shuffle_permutes_rows(self):
        """With shuffle, rows should be permuted (not in sequential order)."""
        import random
        random.seed(42)
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 20, tmpdir, shuffle=True)
            indices = chunks[0].original_indices
            # All 20 indices present
            self.assertEqual(sorted(indices), list(range(20)))
            # With seed=42, the shuffle is deterministic and not identity
            self.assertNotEqual(indices, list(range(20)))

    def test_shuffle_original_indices_map_correctly(self):
        """original_indices should correctly map chunk rows to manifest rows."""
        import csv

        path = os.path.join(FIXTURES, "sample_manifest.csv")

        # Read the original manifest rows
        with open(path, newline="") as f:
            reader = csv.reader(f, delimiter=",")
            next(reader)  # skip header
            original_rows = [row for row in reader]

        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 7, tmpdir, shuffle=True)

            for chunk in chunks:
                with open(chunk.path) as f:
                    lines = f.readlines()
                # Skip header line
                chunk_data_lines = [l.strip() for l in lines[1:]]

                for array_idx, orig_idx in enumerate(chunk.original_indices):
                    # The chunk row should match the original manifest row
                    expected = CHUNK_DELIMITER.join(original_rows[orig_idx])
                    self.assertEqual(chunk_data_lines[array_idx], expected)

    def test_shuffle_covers_all_rows(self):
        """All original manifest rows should appear exactly once across chunks."""
        path = os.path.join(FIXTURES, "sample_manifest.csv")
        with tempfile.TemporaryDirectory() as tmpdir:
            chunks = chunk_manifest(path, ",", 7, tmpdir, shuffle=True)
            all_indices = []
            for chunk in chunks:
                all_indices.extend(chunk.original_indices)
            self.assertEqual(sorted(all_indices), list(range(20)))


if __name__ == "__main__":
    unittest.main()
