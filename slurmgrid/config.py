"""Configuration dataclasses and validation for slurmgrid."""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SlurmConfig:
    """Slurm sbatch flags."""

    partition: Optional[str] = None
    time: Optional[str] = None
    mem: Optional[str] = None
    mem_per_cpu: Optional[str] = None
    cpus_per_task: int = 1
    gpus: Optional[str] = None
    gres: Optional[str] = None
    account: Optional[str] = None
    qos: Optional[str] = None
    constraint: Optional[str] = None
    exclude: Optional[str] = None
    job_name_prefix: str = "sc"
    extra_sbatch_flags: List[str] = field(default_factory=list)
    preamble: Optional[str] = None

    def sbatch_directives(self, job_name: str, array_range: str,
                          output_path: str, error_path: str) -> List[str]:
        """Generate #SBATCH directive lines for an sbatch script."""
        lines = [
            f"#SBATCH --job-name={job_name}",
            f"#SBATCH --array={array_range}",
            f"#SBATCH --output={output_path}",
            f"#SBATCH --error={error_path}",
        ]
        if self.partition:
            lines.append(f"#SBATCH --partition={self.partition}")
        if self.time:
            lines.append(f"#SBATCH --time={self.time}")
        if self.mem:
            lines.append(f"#SBATCH --mem={self.mem}")
        if self.mem_per_cpu:
            lines.append(f"#SBATCH --mem-per-cpu={self.mem_per_cpu}")
        if self.cpus_per_task != 1:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus_per_task}")
        if self.gpus:
            lines.append(f"#SBATCH --gpus={self.gpus}")
        if self.gres:
            lines.append(f"#SBATCH --gres={self.gres}")
        if self.account:
            lines.append(f"#SBATCH --account={self.account}")
        if self.qos:
            lines.append(f"#SBATCH --qos={self.qos}")
        if self.constraint:
            lines.append(f"#SBATCH --constraint={self.constraint}")
        if self.exclude:
            lines.append(f"#SBATCH --exclude={self.exclude}")
        for flag in self.extra_sbatch_flags:
            lines.append(f"#SBATCH {flag}")
        return lines


@dataclass
class RunConfig:
    """Full configuration for a slurmgrid run."""

    manifest: str
    command: str
    state_dir: str = "./sc_state"
    delimiter: Optional[str] = None  # Auto-detect from extension
    chunk_size: Optional[int] = None  # Auto-detect from MaxArraySize
    max_concurrent: int = 10000
    max_retries: int = 3
    poll_interval: int = 30
    max_runtime: Optional[int] = None  # Seconds; None means no limit
    dry_run: bool = False
    shuffle: bool = True
    after_run: Optional[str] = None  # State dir of a run to wait for
    self_resubmit: bool = False  # Resubmit as a new Slurm job on max_runtime exit
    serial_chunks: bool = False  # Submit one chunk at a time (for rate-limited workloads)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    slurm_overrides: Dict[str, str] = field(default_factory=dict)  # Transient; not frozen

    @property
    def resolved_delimiter(self) -> str:
        """Return the delimiter, auto-detecting from file extension if needed."""
        if self.delimiter is not None:
            return self.delimiter
        ext = Path(self.manifest).suffix.lower()
        if ext == ".tsv":
            return "\t"
        return ","  # Default to comma for .csv and anything else

    def placeholders(self) -> List[str]:
        """Extract {placeholder} names from the command template."""
        return re.findall(r"\{(\w+)\}", self.command)

    def validate(self, manifest_headers: List[str]) -> List[str]:
        """Validate config against manifest headers. Returns list of errors."""
        errors = []
        if not os.path.isfile(self.manifest):
            errors.append(f"Manifest file not found: {self.manifest}")
        missing = set(self.placeholders()) - set(manifest_headers)
        if missing:
            errors.append(
                f"Command template references columns not in manifest: "
                f"{', '.join(sorted(missing))}"
            )
        # Validate column names are valid bash identifiers
        bad_cols = [
            h for h in manifest_headers if not re.match(r"^[A-Za-z_]\w*$", h)
        ]
        if bad_cols:
            errors.append(
                f"Column names must be valid bash identifiers "
                f"(alphanumeric + underscore, not starting with digit): "
                f"{', '.join(bad_cols)}"
            )
        if self.max_concurrent < 1:
            errors.append("max_concurrent must be >= 1")
        if self.max_retries < 0:
            errors.append("max_retries must be >= 0")
        if self.poll_interval < 1:
            errors.append("poll_interval must be >= 1")
        if self.chunk_size is not None and self.chunk_size < 1:
            errors.append("chunk_size must be >= 1")
        return errors


def freeze_config(config: RunConfig, state_dir: str) -> None:
    """Write config to state_dir/config.json."""
    path = os.path.join(state_dir, "config.json")
    data = asdict(config)
    data.pop("slurm_overrides", None)  # Transient; not persisted
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def load_config(state_dir: str) -> RunConfig:
    """Load config from state_dir/config.json."""
    path = os.path.join(state_dir, "config.json")
    with open(path) as f:
        data = json.load(f)
    slurm_data = data.pop("slurm", {})
    slurm = SlurmConfig(**slurm_data)
    return RunConfig(**data, slurm=slurm)
