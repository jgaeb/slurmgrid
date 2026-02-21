"""Manifest loading, validation, and chunking."""

from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

# ASCII Unit Separator (0x1F) — used as the internal delimiter for chunk files.
# This character essentially never appears in real data (numbers, paths, strings),
# making bash field splitting completely robust. In generated sbatch scripts we
# parse with IFS=$'\x1f'.
CHUNK_DELIMITER = "\x1f"

# Bash literal for the chunk delimiter, used in generated sbatch scripts.
CHUNK_DELIMITER_BASH = r"$'\x1f'"


@dataclass
class ChunkInfo:
    """Metadata about a single chunk."""

    chunk_id: str
    path: str
    size: int  # Number of data rows (excluding header)
    # Original 0-based manifest row indices for each row in this chunk.
    original_indices: List[int] = field(default_factory=list)


def load_headers(manifest_path: str, delimiter: str) -> List[str]:
    """Read just the header row from a manifest file."""
    with open(manifest_path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        headers = next(reader)
    return [h.strip() for h in headers]


def count_rows(manifest_path: str) -> int:
    """Count data rows in a manifest (excludes header)."""
    count = 0
    with open(manifest_path) as f:
        for _ in f:
            count += 1
    return max(0, count - 1)  # Subtract header


def validate_manifest(
    manifest_path: str,
    delimiter: str,
    command_placeholders: List[str],
) -> List[str]:
    """Validate manifest contents. Returns list of errors."""
    errors = []
    headers = load_headers(manifest_path, delimiter)

    if not headers:
        errors.append("Manifest has no columns")
        return errors

    # Check for duplicate column names
    seen = set()
    for h in headers:
        if h in seen:
            errors.append(f"Duplicate column name: {h}")
        seen.add(h)

    # Check that no values contain the chunk delimiter (unit separator).
    # This should essentially never happen with real data.
    with open(manifest_path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        next(reader)  # Skip header
        for row_num, row in enumerate(reader, start=2):
            for col_idx, val in enumerate(row):
                if CHUNK_DELIMITER in val:
                    errors.append(
                        f"Row {row_num}, column '{headers[col_idx]}' "
                        f"contains ASCII unit separator (0x1F)"
                    )
                    if len(errors) >= 10:
                        errors.append("(too many errors, stopping)")
                        return errors

    return errors


def chunk_manifest(
    manifest_path: str,
    delimiter: str,
    chunk_size: int,
    output_dir: str,
    chunk_id_prefix: str = "chunk",
    shuffle: bool = True,
) -> List[ChunkInfo]:
    """Split a manifest into sub-manifests.

    Each sub-manifest includes the header row and up to chunk_size data rows.
    Chunk files use ASCII Unit Separator (0x1F) as the field delimiter,
    regardless of the input format, since the generated sbatch scripts parse
    them with bash IFS and this delimiter never appears in real data.

    If shuffle is True (default), rows are randomly permuted before chunking
    so that each chunk gets a representative mix of the parameter space.

    Returns a list of ChunkInfo describing each chunk, including the original
    manifest row indices for each row.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Read all rows with their original indices
    indexed_rows: List[tuple] = []
    with open(manifest_path, newline="") as f:
        reader = csv.reader(f, delimiter=delimiter)
        header_row = next(reader)
        header_line = CHUNK_DELIMITER.join(header_row)
        for row_idx, row in enumerate(reader):
            indexed_rows.append((row_idx, CHUNK_DELIMITER.join(row)))

    if shuffle:
        random.shuffle(indexed_rows)

    # Split into chunks
    chunks: List[ChunkInfo] = []
    for batch_start in range(0, len(indexed_rows), chunk_size):
        batch = indexed_rows[batch_start:batch_start + chunk_size]
        original_indices = [idx for idx, _ in batch]
        rows = [row for _, row in batch]
        chunk = _write_chunk(
            len(chunks), chunk_id_prefix, header_line, rows, output_dir,
            original_indices=original_indices,
        )
        chunks.append(chunk)

    return chunks


def _write_chunk(
    chunk_idx: int,
    prefix: str,
    header_line: str,
    rows: List[str],
    output_dir: str,
    original_indices: Optional[List[int]] = None,
) -> ChunkInfo:
    """Write a single chunk file and return its ChunkInfo."""
    chunk_id = f"{prefix}_{chunk_idx:03d}"
    chunk_path = os.path.join(output_dir, f"{chunk_id}.chunk")
    with open(chunk_path, "w") as f:
        f.write(header_line + "\n")
        for row in rows:
            f.write(row + "\n")
    return ChunkInfo(
        chunk_id=chunk_id, path=chunk_path, size=len(rows),
        original_indices=original_indices or list(range(len(rows))),
    )
