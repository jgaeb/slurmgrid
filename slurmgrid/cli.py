"""Command-line interface for slurmgrid."""

from __future__ import annotations

import configargparse
import logging
import os
import sys
from typing import List, Optional

from . import __version__
from .config import RunConfig, SlurmConfig, freeze_config, load_config
from .log import setup_logging
from .manifest import (
    ChunkInfo,
    chunk_manifest,
    count_rows,
    load_headers,
    validate_manifest,
)
from .monitor import run as run_monitor
from .script import (
    ensure_log_dir,
    generate_sbatch_script,
    script_path_for_chunk,
    write_sbatch_script,
)
from .slurm import get_max_array_size
from .state import (
    load_state,
    new_state,
    save_state,
    state_exists,
)

log = logging.getLogger(__name__)


def main(argv: Optional[List[str]] = None) -> None:
    parser = configargparse.ArgumentParser(
        prog="slurmgrid",
        description="Manage large Slurm job arrays that exceed submission limits.",
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}",
    )
    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # --- submit ---
    p_submit = subparsers.add_parser(
        "submit", help="Submit a new manifest for processing",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
    )
    p_submit.add_argument(
        "--config", is_config_file=True, metavar="CONFIG_FILE",
        help="YAML config file; any CLI option can be set here as a key",
    )
    p_submit.add_argument(
        "--manifest", required=True, help="CSV/TSV manifest file",
    )
    p_submit.add_argument(
        "--command", required=True,
        help="Command template with {column} placeholders",
    )
    p_submit.add_argument(
        "--state-dir", default="./sc_state",
        help="Directory for state, chunks, scripts, logs (default: ./sc_state)",
    )
    p_submit.add_argument(
        "--delimiter", default=None,
        help="Manifest delimiter (default: auto-detect from extension)",
    )
    p_submit.add_argument(
        "--chunk-size", type=int, default=None,
        help="Jobs per array chunk (default: auto-detect from MaxArraySize)",
    )
    p_submit.add_argument(
        "--max-concurrent", type=int, default=10000,
        help="Max total jobs in Slurm at once (default: 10000)",
    )
    p_submit.add_argument(
        "--max-retries", type=int, default=3,
        help="Max retries per failed job (default: 3)",
    )
    p_submit.add_argument(
        "--poll-interval", type=int, default=30,
        help="Seconds between status checks (default: 30)",
    )
    p_submit.add_argument(
        "--max-runtime", type=int, default=None,
        help="Max seconds to run before saving state and exiting (default: unlimited)",
    )
    p_submit.add_argument("--dry-run", action="store_true",
                          help="Generate scripts but don't submit")
    p_submit.add_argument("--no-shuffle", action="store_true",
                          help="Don't shuffle manifest rows before chunking "
                               "(default: shuffle for balanced chunks)")
    p_submit.add_argument(
        "--after-run", default=None, metavar="STATE_DIR",
        help="Wait for a previous run to finish before submitting "
             "(path to its state directory)",
    )
    _add_slurm_args(p_submit)

    # --- resume ---
    p_resume = subparsers.add_parser(
        "resume", help="Resume a previously interrupted run",
    )
    p_resume.add_argument("--state-dir", required=True)
    p_resume.add_argument("--poll-interval", type=int, default=None,
                          help="Override poll interval")
    p_resume.add_argument("--max-runtime", type=int, default=None,
                          help="Max seconds to run before saving state and exiting")

    # --- status ---
    p_status = subparsers.add_parser(
        "status", help="Show current status",
    )
    p_status.add_argument("--state-dir", required=True)

    # --- cancel ---
    p_cancel = subparsers.add_parser(
        "cancel", help="Cancel all active Slurm jobs for this run",
    )
    p_cancel.add_argument("--state-dir", required=True)

    args = parser.parse_args(argv)

    if args.subcommand == "submit":
        cmd_submit(args)
    elif args.subcommand == "resume":
        cmd_resume(args)
    elif args.subcommand == "status":
        cmd_status(args)
    elif args.subcommand == "cancel":
        cmd_cancel(args)


def _add_slurm_args(parser: argparse.ArgumentParser) -> None:
    """Add Slurm-specific arguments to a subparser."""
    g = parser.add_argument_group("Slurm options")
    g.add_argument("--partition", default=None)
    g.add_argument("--time", default=None, help="Wall time limit (e.g., 01:00:00)")
    g.add_argument("--mem", default=None, help="Memory per node (e.g., 4G)")
    g.add_argument("--mem-per-cpu", default=None, help="Memory per CPU")
    g.add_argument("--cpus-per-task", type=int, default=1)
    g.add_argument("--gpus", default=None)
    g.add_argument("--gres", default=None)
    g.add_argument("--account", default=None)
    g.add_argument("--qos", default=None)
    g.add_argument("--constraint", default=None)
    g.add_argument("--exclude", default=None)
    g.add_argument("--job-name-prefix", default="sc",
                   help="Prefix for Slurm job names (default: sc)")
    g.add_argument("--preamble", default=None,
                   help="Shell commands before the main command "
                        "(e.g., 'module load python/3.10')")
    g.add_argument("--preamble-file", default=None,
                   help="File containing preamble commands")
    g.add_argument("--extra-sbatch", action="append", default=[],
                   help="Extra #SBATCH flags (repeatable)")


def _build_slurm_config(args: argparse.Namespace) -> SlurmConfig:
    """Build a SlurmConfig from parsed CLI args."""
    preamble = args.preamble
    if args.preamble_file:
        with open(args.preamble_file) as f:
            file_preamble = f.read().strip()
        preamble = f"{preamble}\n{file_preamble}" if preamble else file_preamble

    return SlurmConfig(
        partition=args.partition,
        time=args.time,
        mem=args.mem,
        mem_per_cpu=args.mem_per_cpu,
        cpus_per_task=args.cpus_per_task,
        gpus=args.gpus,
        gres=args.gres,
        account=args.account,
        qos=args.qos,
        constraint=args.constraint,
        exclude=args.exclude,
        job_name_prefix=args.job_name_prefix,
        extra_sbatch_flags=args.extra_sbatch,
        preamble=preamble,
    )


def cmd_submit(args: argparse.Namespace) -> None:
    """Handle the 'submit' subcommand."""
    state_dir = os.path.abspath(args.state_dir)
    setup_logging(state_dir)

    if state_exists(state_dir):
        log.error("State directory already contains a run: %s", state_dir)
        log.error("Use 'resume' to continue, or choose a different --state-dir")
        sys.exit(1)

    after_run = os.path.abspath(args.after_run) if args.after_run else None
    if after_run and not state_exists(after_run):
        log.error("--after-run state directory not found: %s", after_run)
        sys.exit(1)

    # Resolve manifest path to absolute (it's referenced from sbatch scripts)
    manifest_path = os.path.abspath(args.manifest)

    slurm_config = _build_slurm_config(args)
    config = RunConfig(
        manifest=manifest_path,
        command=args.command,
        state_dir=state_dir,
        delimiter=args.delimiter,
        chunk_size=args.chunk_size,
        max_concurrent=args.max_concurrent,
        max_retries=args.max_retries,
        poll_interval=args.poll_interval,
        max_runtime=args.max_runtime,
        dry_run=args.dry_run,
        shuffle=not args.no_shuffle,
        after_run=after_run,
        slurm=slurm_config,
    )

    # Load and validate manifest
    delimiter = config.resolved_delimiter
    headers = load_headers(manifest_path, delimiter)
    errors = config.validate(headers)
    manifest_errors = validate_manifest(manifest_path, delimiter, config.placeholders())
    errors.extend(manifest_errors)
    if errors:
        for e in errors:
            log.error("Validation error: %s", e)
        sys.exit(1)

    total_rows = count_rows(manifest_path)
    log.info("Manifest: %s (%d jobs, %d columns)", manifest_path, total_rows,
             len(headers))

    # Determine chunk size
    chunk_size = config.chunk_size
    if chunk_size is None:
        max_array = get_max_array_size()
        # Default to 1/3 of MaxArraySize so chunks cycle quickly
        # (avoiding straggler jobs blocking all other work), while still
        # being large enough to keep submission overhead low.
        chunk_size = max(1, max_array // 3)
        log.info("Auto-detected chunk size: %d (MaxArraySize=%d)",
                 chunk_size, max_array)
    config.chunk_size = chunk_size

    # Create state directory structure
    os.makedirs(state_dir, exist_ok=True)
    chunks_dir = os.path.join(state_dir, "chunks")
    scripts_dir = os.path.join(state_dir, "scripts")

    # Freeze config
    freeze_config(config, state_dir)

    # Chunk the manifest
    log.info("Chunking manifest into groups of %d...%s", chunk_size,
             " (shuffled)" if config.shuffle else "")
    chunks = chunk_manifest(
        manifest_path, delimiter, chunk_size, chunks_dir,
        shuffle=config.shuffle,
    )
    log.info("Created %d chunks", len(chunks))

    # Initialize state
    state = new_state(total_rows, chunk_size, config.max_concurrent,
                      config.max_retries)

    # Register chunks in state and generate scripts
    throttle = config.max_concurrent
    for chunk in chunks:
        row_mapping = {
            str(orig_idx): array_idx
            for array_idx, orig_idx in enumerate(chunk.original_indices)
        }
        state.add_chunk(chunk.chunk_id, chunk.size, row_mapping)

        # Generate sbatch script
        script_content = generate_sbatch_script(
            chunk, config, state_dir, throttle=throttle,
        )
        spath = script_path_for_chunk(chunk, state_dir)
        write_sbatch_script(script_content, spath)
        ensure_log_dir(chunk, state_dir)

    save_state(state, state_dir)
    log.info("State initialized at %s", state_dir)

    # Print summary and enter monitor loop
    _print_summary(state)
    if config.dry_run:
        log.info("[DRY RUN] Scripts generated but not submitted. Inspect:")
        log.info("  Chunks:  %s", chunks_dir)
        log.info("  Scripts: %s", scripts_dir)
        return

    final_state = run_monitor(state, config)
    _print_final_report(final_state)


def cmd_resume(args: argparse.Namespace) -> None:
    """Handle the 'resume' subcommand."""
    state_dir = os.path.abspath(args.state_dir)
    setup_logging(state_dir)

    if not state_exists(state_dir):
        log.error("No state found at %s", state_dir)
        sys.exit(1)

    config = load_config(state_dir)
    if args.poll_interval is not None:
        config.poll_interval = args.poll_interval
    if args.max_runtime is not None:
        config.max_runtime = args.max_runtime

    state = load_state(state_dir)
    log.info("Resumed run from %s", state_dir)
    _print_summary(state)

    final_state = run_monitor(state, config)
    _print_final_report(final_state)


def cmd_status(args: argparse.Namespace) -> None:
    """Handle the 'status' subcommand."""
    state_dir = os.path.abspath(args.state_dir)

    if not state_exists(state_dir):
        print(f"No state found at {state_dir}")
        sys.exit(1)

    state = load_state(state_dir)
    _print_summary(state)


def cmd_cancel(args: argparse.Namespace) -> None:
    """Handle the 'cancel' subcommand."""
    from . import slurm as slurm_mod

    state_dir = os.path.abspath(args.state_dir)
    setup_logging(state_dir)

    if not state_exists(state_dir):
        log.error("No state found at %s", state_dir)
        sys.exit(1)

    state = load_state(state_dir)
    active = state.active_chunks()
    if not active:
        print("No active jobs to cancel.")
        return

    job_ids = [c.slurm_job_id for c in active if c.slurm_job_id]
    if not job_ids:
        print("No Slurm job IDs found for active chunks.")
        return

    print(f"Cancelling {len(job_ids)} Slurm jobs: {', '.join(job_ids)}")
    slurm_mod.scancel(job_ids)

    # Update state
    for chunk in active:
        state.mark_completed(chunk.chunk_id)
    save_state(state, state_dir)
    print("Done. Jobs cancelled and state updated.")


def _print_summary(state: State) -> None:
    """Print a human-readable status summary."""
    s = state.summary()
    print("=" * 50)
    print(f"  Total jobs:        {s['total_jobs']:>8}")
    print(f"  Completed:         {s['completed_tasks']:>8}", end="")
    if s["total_jobs"] > 0:
        pct = 100 * s["completed_tasks"] / s["total_jobs"]
        print(f"  ({pct:.1f}%)")
    else:
        print()
    print(f"  Active:            {s['active_tasks']:>8}")
    print(f"  Pending:           {s['pending_tasks']:>8}")
    print(f"  Failed (retrying): {s['failed_retry']:>8}")
    print(f"  Failed (final):    {s['failed_final']:>8}")
    print(f"  Chunks: {s['completed_chunks']}/{s['total_chunks']} completed, "
          f"{s['active_chunks']} active, {s['pending_chunks']} pending")
    print("=" * 50)


def _print_final_report(state: State) -> None:
    """Print the final report when the run is done."""
    s = state.summary()
    print()
    print("Run complete.")
    _print_summary(state)

    if s["failed_final"] > 0:
        print(f"\n{s['failed_final']} jobs failed permanently. "
              "Check state.json for details.")
        # List the failed global indices
        failed_indices = sorted(
            int(k) for k, f in state.failures.items() if f.permanently_failed
        )
        if len(failed_indices) <= 20:
            print(f"Failed manifest rows (0-indexed): {failed_indices}")
        else:
            print(f"First 20 failed manifest rows: {failed_indices[:20]}...")
