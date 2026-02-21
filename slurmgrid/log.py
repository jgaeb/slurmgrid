"""Logging configuration for slurmgrid."""

from __future__ import annotations

import logging
import os
import sys


def setup_logging(state_dir: str, verbose: bool = False) -> None:
    """Configure logging to stderr and a file in state_dir.

    Called once at startup. The file handler logs everything (DEBUG);
    the stderr handler logs INFO and above (or DEBUG if verbose).
    """
    os.makedirs(state_dir, exist_ok=True)
    log_path = os.path.join(state_dir, "slurmgrid.log")

    root = logging.getLogger("slurmgrid")
    root.setLevel(logging.DEBUG)

    # Avoid adding duplicate handlers on repeated calls (e.g., in tests)
    if root.handlers:
        return

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # File handler — always DEBUG
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)

    # Stderr handler
    sh = logging.StreamHandler(sys.stderr)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(sh)
