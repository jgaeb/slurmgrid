"""sbatch script generation."""

from __future__ import annotations

import os
import re
import stat
from typing import List, Optional

from .config import RunConfig, SlurmConfig
from .manifest import CHUNK_DELIMITER_BASH, ChunkInfo


def generate_sbatch_script(
    chunk: ChunkInfo,
    config: RunConfig,
    state_dir: str,
    throttle: Optional[int] = None,
) -> str:
    """Generate a self-contained bash sbatch script for a chunk.

    The script:
    1. Uses awk to extract the manifest line for SLURM_ARRAY_TASK_ID
    2. Splits the line on the unit separator into positional fields
    3. Exports each field as SC_<column_name>
    4. Runs the user's command with placeholders resolved to $SC_<name>

    If throttle is provided and less than chunk.size, the array directive
    includes a %throttle suffix to limit how many tasks run simultaneously.
    """
    log_dir = os.path.join(state_dir, "logs", chunk.chunk_id)
    array_max = chunk.size - 1
    array_range = f"0-{array_max}"
    if throttle is not None and throttle < chunk.size:
        array_range = f"0-{array_max}%{throttle}"

    directives = config.slurm.sbatch_directives(
        job_name=f"{config.slurm.job_name_prefix}_{chunk.chunk_id}",
        array_range=array_range,
        output_path=os.path.join(log_dir, "slurm-%A_%a.out"),
        error_path=os.path.join(log_dir, "slurm-%A_%a.err"),
    )

    # Build the variable export block from the header columns
    headers = _read_chunk_headers(chunk.path)
    export_lines = []
    for i, col in enumerate(headers):
        # Fields array is 0-indexed; VALS[$i] gives the i-th column
        export_lines.append(f'  export "SC_{col}=${{VALS[{i}]}}"')

    # Replace {placeholder} in command with "$SC_placeholder"
    resolved_command = _resolve_command(config.command, headers)

    # Assemble the script
    parts = ["#!/bin/bash"]
    for d in directives:
        parts.append(d)
    parts.append("")

    # Preamble (module loads, conda activate, etc.)
    if config.slurm.preamble:
        parts.append("# Preamble")
        parts.append(config.slurm.preamble)
        parts.append("")

    parts.append("# Parse manifest line for this array task")
    parts.append(f'MANIFEST="{chunk.path}"')
    parts.append(f'DELIM={CHUNK_DELIMITER_BASH}')
    parts.append(
        'LINE=$(awk -v idx="$SLURM_ARRAY_TASK_ID" '
        "'NR == idx + 2' \"$MANIFEST\")"
    )
    parts.append('IFS="$DELIM" read -ra VALS <<< "$LINE"')
    parts.append("")

    parts.append("# Export columns as SC_<name> variables")
    for line in export_lines:
        parts.append(line)
    parts.append("")

    parts.append("# Run user command")
    parts.append(resolved_command)
    parts.append("")

    return "\n".join(parts)


def write_sbatch_script(content: str, output_path: str) -> None:
    """Write an sbatch script to disk and make it executable."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)
    os.chmod(output_path, os.stat(output_path).st_mode | stat.S_IEXEC)


def script_path_for_chunk(chunk: ChunkInfo, state_dir: str) -> str:
    """Return the conventional script path for a chunk."""
    return os.path.join(state_dir, "scripts", f"{chunk.chunk_id}.sh")


def ensure_log_dir(chunk: ChunkInfo, state_dir: str) -> None:
    """Create the log directory for a chunk's Slurm output."""
    log_dir = os.path.join(state_dir, "logs", chunk.chunk_id)
    os.makedirs(log_dir, exist_ok=True)


def _read_chunk_headers(chunk_path: str) -> List[str]:
    """Read the header row from a chunk file (unit-separator-delimited)."""
    with open(chunk_path) as f:
        header_line = f.readline().rstrip("\n")
    return header_line.split("\x1f")


def _resolve_command(command_template: str, headers: List[str]) -> str:
    """Replace {placeholder} in the command template with "$SC_placeholder".

    Only replaces placeholders whose names match manifest columns.
    """
    def replacer(match: re.Match) -> str:
        name = match.group(1)
        if name in headers:
            return f'"$SC_{name}"'
        # Leave unrecognized placeholders as-is (will be caught by validation)
        return match.group(0)

    return re.sub(r"\{(\w+)\}", replacer, command_template)
