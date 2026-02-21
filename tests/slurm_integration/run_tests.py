#!/usr/bin/env python3
"""Integration tests that run against a real Slurm cluster.

This script is designed to run inside the slurmctld container of
giovtorres/slurm-docker-cluster. It exercises the full slurmgrid
workflow: submit -> monitor -> complete, and submit -> fail -> retry.

IMPORTANT: Test directories must be under /data (which is a shared Docker
volume mounted on slurmctld, c1, and c2). /tmp and /home are container-local,
so compute nodes wouldn't be able to read files stored there.

Usage (from inside the container):
    python3 run_tests.py /path/to/slurmgrid

Exit code 0 on success, 1 on failure.
"""

import csv
import json
import os
import shutil
import subprocess
import sys
import time

# Shared directory visible to all containers in the Docker Slurm cluster.
# The slurm-docker-cluster mounts the "slurm_jobdir" volume at /data on
# slurmctld, c1, and c2. This is the ONLY shared filesystem.
SHARED_BASE = "/data/slurm_test"


def log(msg):
    print(f"[TEST] {msg}", flush=True)


def make_shared_dir(name):
    """Create a test directory under the shared filesystem."""
    path = os.path.join(SHARED_BASE, name)
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return path


def cleanup_shared():
    """Remove the shared test directory."""
    if os.path.exists(SHARED_BASE):
        shutil.rmtree(SHARED_BASE)


def check_slurm_ready():
    """Verify Slurm is functional before running tests."""
    log("Checking Slurm is ready...")
    result = subprocess.run(
        ["sinfo", "--noheader"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"sinfo failed: {result.stderr}")
        return False
    if not result.stdout.strip():
        log("No nodes registered in Slurm")
        return False
    log(f"Slurm nodes: {result.stdout.strip()}")

    # Verify sacct works
    result = subprocess.run(
        ["sacct", "--noheader"], capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"sacct failed: {result.stderr}")
        return False
    log("sacct is functional")

    # Quick smoke test: submit a single job and verify sacct reports it
    log("Smoke test: submitting a single job...")
    result = subprocess.run(
        ["sbatch", "--wrap=echo hello", "--time=00:01:00", "--parsable"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        log(f"sbatch smoke test failed: {result.stderr}")
        return False
    job_id = result.stdout.strip().split(";")[0]
    log(f"Smoke test job: {job_id}")

    # Wait for it to complete and show up in sacct
    for _ in range(30):
        time.sleep(2)
        result = subprocess.run(
            ["sacct", f"--jobs={job_id}", "--format=State", "--parsable2",
             "--noheader"],
            capture_output=True, text=True,
        )
        if "COMPLETED" in result.stdout:
            log("Smoke test: sacct reports COMPLETED")
            return True
    log(f"Smoke test: sacct never reported COMPLETED. Output: {result.stdout}")
    return False


def create_manifest(path, num_rows):
    """Create a simple test manifest."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "value"])
        for i in range(num_rows):
            writer.writerow([i, f"val_{i}"])


def create_failing_manifest(path, num_rows, fail_indices):
    """Create a manifest where certain rows will cause the command to fail."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "should_fail"])
        for i in range(num_rows):
            writer.writerow([i, "1" if i in fail_indices else "0"])


def run_slurmgrid(project_dir, args, timeout=300):
    """Run slurmgrid as a subprocess."""
    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir
    cmd = [sys.executable, "-m", "slurmgrid"] + args
    log(f"Running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout, env=env,
    )
    log(f"Exit code: {result.returncode}")
    if result.stdout:
        for line in result.stdout.strip().splitlines():
            log(f"  stdout: {line}")
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            log(f"  stderr: {line}")
    return result


def load_state(state_dir):
    """Load state.json from a state directory."""
    with open(os.path.join(state_dir, "state.json")) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Debug helpers
# ---------------------------------------------------------------------------

def _dump_debug_info(state_dir):
    """Dump generated scripts and Slurm logs for debugging CI failures."""
    # Dump generated sbatch scripts
    scripts_dir = os.path.join(state_dir, "scripts")
    if os.path.isdir(scripts_dir):
        for fname in sorted(os.listdir(scripts_dir)):
            fpath = os.path.join(scripts_dir, fname)
            log(f"--- Generated script: {fname} ---")
            with open(fpath) as f:
                for line in f:
                    log(f"  {line.rstrip()}")

    # Dump chunk files
    chunks_dir = os.path.join(state_dir, "chunks")
    if os.path.isdir(chunks_dir):
        for fname in sorted(os.listdir(chunks_dir)):
            fpath = os.path.join(chunks_dir, fname)
            log(f"--- Chunk file: {fname} ---")
            with open(fpath) as f:
                for line in f:
                    log(f"  {line.rstrip()}")

    # Dump Slurm job output logs (stderr)
    logs_dir = os.path.join(state_dir, "logs")
    if os.path.isdir(logs_dir):
        for chunk_dir in sorted(os.listdir(logs_dir)):
            chunk_log_dir = os.path.join(logs_dir, chunk_dir)
            if not os.path.isdir(chunk_log_dir):
                continue
            for fname in sorted(os.listdir(chunk_log_dir))[:10]:
                fpath = os.path.join(chunk_log_dir, fname)
                log(f"--- Slurm log: {chunk_dir}/{fname} ---")
                try:
                    with open(fpath) as f:
                        content = f.read().strip()
                        if content:
                            for line in content.splitlines():
                                log(f"  {line}")
                        else:
                            log("  (empty)")
                except Exception as e:
                    log(f"  (error reading: {e})")

    # Dump state.json
    state_path = os.path.join(state_dir, "state.json")
    if os.path.isfile(state_path):
        log("--- state.json ---")
        with open(state_path) as f:
            for line in f:
                log(f"  {line.rstrip()}")


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

def test_basic_submit_and_complete(project_dir):
    """Test: submit 12 jobs in chunks of 5, all succeed."""
    log("=" * 60)
    log("TEST: basic_submit_and_complete")
    log("=" * 60)

    tmpdir = make_shared_dir("test_basic")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    create_manifest(manifest, 12)

    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", "echo idx={idx} value={value}",
        "--state-dir", state_dir,
        "--chunk-size", "5",
        "--max-concurrent", "10",
        "--max-retries", "0",
        "--max-runtime", "240",
        "--poll-interval", "5",
        "--time", "00:05:00",
    ])

    if result.returncode != 0:
        log("FAIL: slurmgrid exited with non-zero")
        return False

    state = load_state(state_dir)
    total_chunks = len(state["chunks"])
    completed = sum(
        1 for c in state["chunks"].values() if c["status"] == "completed"
    )
    failed = len(state.get("failures", {}))

    log(f"Chunks: {completed}/{total_chunks} completed")
    log(f"Failures: {failed}")

    if completed != total_chunks:
        log(f"FAIL: expected {total_chunks} completed chunks, got {completed}")
        return False
    if failed != 0:
        log(f"FAIL: expected 0 failures, got {failed}")
        return False

    log("PASS")
    return True


def test_failure_and_retry(project_dir):
    """Test: some jobs fail, get retried, and fail permanently."""
    log("=" * 60)
    log("TEST: failure_and_retry")
    log("=" * 60)

    tmpdir = make_shared_dir("test_failure")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    # 10 jobs, indices 2 and 7 will fail
    create_failing_manifest(manifest, 10, fail_indices={2, 7})

    # Command: exit with the value of should_fail (1=fail, 0=succeed).
    # This is simpler and avoids shell quoting issues with bash -c.
    command = "exit {should_fail}"

    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", command,
        "--state-dir", state_dir,
        "--chunk-size", "5",
        "--max-concurrent", "10",
        "--max-retries", "1",
        "--max-runtime", "240",
        "--poll-interval", "5",
        "--time", "00:05:00",
    ])

    if result.returncode != 0:
        log("FAIL: slurmgrid exited with non-zero")
        _dump_debug_info(state_dir)
        return False

    state = load_state(state_dir)
    failures = state.get("failures", {})
    perm_failed = sum(
        1 for f in failures.values() if f["permanently_failed"]
    )

    log(f"Total failures tracked: {len(failures)}")
    log(f"Permanently failed: {perm_failed}")

    if perm_failed != 2:
        log(f"FAIL: expected 2 permanently failed, got {perm_failed}")
        _dump_debug_info(state_dir)
        return False

    log("PASS")
    return True


def test_dry_run(project_dir):
    """Test: dry-run generates scripts without submitting."""
    log("=" * 60)
    log("TEST: dry_run")
    log("=" * 60)

    tmpdir = make_shared_dir("test_dry_run")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    create_manifest(manifest, 8)

    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", "echo {idx}",
        "--state-dir", state_dir,
        "--chunk-size", "5",
        "--max-concurrent", "10",
        "--dry-run",
        "--time", "00:05:00",
    ])

    if result.returncode != 0:
        log("FAIL: dry-run exited with non-zero")
        return False

    # Check scripts were generated
    scripts_dir = os.path.join(state_dir, "scripts")
    scripts = os.listdir(scripts_dir) if os.path.isdir(scripts_dir) else []
    log(f"Generated scripts: {scripts}")

    if len(scripts) != 2:  # 8 jobs / chunk_size 5 = 2 chunks
        log(f"FAIL: expected 2 scripts, got {len(scripts)}")
        return False

    # Check no jobs were submitted
    state = load_state(state_dir)
    submitted = sum(
        1 for c in state["chunks"].values()
        if c["slurm_job_id"] is not None
    )
    if submitted != 0:
        log(f"FAIL: dry-run submitted {submitted} jobs")
        return False

    log("PASS")
    return True


def test_status_command(project_dir):
    """Test: status command reads state without modifying it."""
    log("=" * 60)
    log("TEST: status_command")
    log("=" * 60)

    tmpdir = make_shared_dir("test_status")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    create_manifest(manifest, 5)

    # First do a dry-run to create state
    run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", "echo {idx}",
        "--state-dir", state_dir,
        "--chunk-size", "5",
        "--dry-run",
        "--time", "00:05:00",
    ])

    # Now run status
    result = run_slurmgrid(project_dir, [
        "status", "--state-dir", state_dir,
    ])

    if result.returncode != 0:
        log("FAIL: status command failed")
        return False

    if "Total jobs" not in result.stdout:
        log("FAIL: status output missing expected content")
        return False

    log("PASS")
    return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} /path/to/slurmgrid_project")
        sys.exit(1)

    project_dir = os.path.abspath(sys.argv[1])
    if not os.path.isdir(os.path.join(project_dir, "slurmgrid")):
        print(f"Error: {project_dir}/slurmgrid not found")
        sys.exit(1)

    if not check_slurm_ready():
        log("Slurm is not ready, aborting")
        sys.exit(1)

    tests = [
        test_dry_run,
        test_status_command,
        test_basic_submit_and_complete,
        test_failure_and_retry,
    ]

    results = {}
    for test_fn in tests:
        try:
            results[test_fn.__name__] = test_fn(project_dir)
        except Exception as e:
            log(f"EXCEPTION in {test_fn.__name__}: {e}")
            results[test_fn.__name__] = False

    # Clean up shared test directories
    cleanup_shared()

    log("")
    log("=" * 60)
    log("SUMMARY")
    log("=" * 60)
    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        log(f"  {status}: {name}")
        if not passed:
            all_pass = False

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
