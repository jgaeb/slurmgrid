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


def test_reset_failures_with_overrides(project_dir):
    """Test: submit jobs that fail permanently (max_retries=0), fix the
    manifest, then resume --reset-failures with a Slurm override and verify
    they complete. Also checks that slurm_overrides are recorded on retry chunks."""
    log("=" * 60)
    log("TEST: reset_failures_with_overrides")
    log("=" * 60)

    tmpdir = make_shared_dir("test_reset_failures")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")

    # 6 jobs; indices 1 and 4 will fail (exit 1)
    create_failing_manifest(manifest, 6, fail_indices={1, 4})

    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", "exit {should_fail}",
        "--state-dir", state_dir,
        "--chunk-size", "6",
        "--max-concurrent", "10",
        "--max-retries", "0",
        "--max-runtime", "120",
        "--poll-interval", "5",
        "--time", "00:05:00",
    ])

    if result.returncode != 0:
        log("FAIL: initial submit exited with non-zero")
        _dump_debug_info(state_dir)
        return False

    state = load_state(state_dir)
    perm_failed = sum(
        1 for f in state.get("failures", {}).values()
        if f["permanently_failed"]
    )
    log(f"Permanently failed after initial run: {perm_failed}")

    if perm_failed != 2:
        log(f"FAIL: expected 2 permanently failed, got {perm_failed}")
        _dump_debug_info(state_dir)
        return False

    # "Fix" the manifest so previously-failing rows now succeed.
    # _create_retry_batch re-reads the manifest to build retry chunks,
    # so the updated values will be picked up.
    create_failing_manifest(manifest, 6, fail_indices=set())

    # Resume with --reset-failures and a Slurm override (--mem) to verify
    # overrides are recorded on the retry chunks.
    log("Resuming with --reset-failures --mem 100M ...")
    result = run_slurmgrid(project_dir, [
        "resume",
        "--state-dir", state_dir,
        "--reset-failures",
        "--mem", "100M",
        "--max-runtime", "120",
        "--poll-interval", "5",
    ])

    if result.returncode != 0:
        log("FAIL: resume exited with non-zero")
        _dump_debug_info(state_dir)
        return False

    state = load_state(state_dir)
    perm_failed_after = sum(
        1 for f in state.get("failures", {}).values()
        if f["permanently_failed"]
    )
    total_failures = len(state.get("failures", {}))
    log(f"After resume: {total_failures} failure records, "
        f"{perm_failed_after} permanently failed")

    if perm_failed_after > 0:
        log(f"FAIL: {perm_failed_after} tasks still permanently failed")
        _dump_debug_info(state_dir)
        return False

    if total_failures > 0:
        log(f"FAIL: expected 0 failure records (succeeded on retry), "
            f"got {total_failures}")
        _dump_debug_info(state_dir)
        return False

    # Check that retry chunks have slurm_overrides recorded
    retry_chunks = {cid: c for cid, c in state["chunks"].items()
                    if cid.startswith("retry_")}
    if not retry_chunks:
        log("FAIL: no retry chunks found")
        _dump_debug_info(state_dir)
        return False

    for cid, c in retry_chunks.items():
        overrides = c.get("slurm_overrides", {})
        log(f"  Retry chunk {cid}: slurm_overrides={overrides}")
        if overrides.get("mem") != "100M":
            log(f"FAIL: retry chunk {cid} missing mem override")
            return False

    log("PASS")
    return True


def test_after_run(project_dir):
    """Test: stage 2 waits for a still-running stage 1 before submitting."""
    log("=" * 60)
    log("TEST: after_run")
    log("=" * 60)

    tmpdir = make_shared_dir("test_after_run")
    manifest1 = os.path.join(tmpdir, "manifest1.csv")
    manifest2 = os.path.join(tmpdir, "manifest2.csv")
    state_dir1 = os.path.join(tmpdir, "state1")
    state_dir2 = os.path.join(tmpdir, "state2")

    # Stage 1: 4 jobs that each sleep for a few seconds so stage 1 is still
    # running when stage 2 starts.
    create_manifest(manifest1, 4)
    create_manifest(manifest2, 4)

    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir
    stage1_cmd = [
        sys.executable, "-m", "slurmgrid", "submit",
        "--manifest", manifest1,
        "--command", "sleep 8 && echo done idx={idx}",
        "--state-dir", state_dir1,
        "--chunk-size", "5",
        "--max-concurrent", "10",
        "--max-retries", "0",
        "--max-runtime", "180",
        "--poll-interval", "5",
        "--time", "00:05:00",
    ]
    log(f"Starting stage 1 in background: {' '.join(stage1_cmd)}")
    stage1_proc = subprocess.Popen(
        stage1_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=env,
    )

    # Wait for stage 1 to create its state file (meaning it has been submitted)
    log("Waiting for stage 1 state file to appear...")
    state1_json = os.path.join(state_dir1, "state.json")
    for _ in range(60):
        if os.path.isfile(state1_json):
            break
        time.sleep(1)
    else:
        log("FAIL: stage 1 state file never appeared")
        stage1_proc.kill()
        return False

    # Confirm stage 1 is not yet done
    state1_snapshot = load_state(state_dir1)
    if all(c["status"] in ("completed", "partial_failure")
           for c in state1_snapshot["chunks"].values()):
        log("WARN: stage 1 finished before stage 2 could start — "
            "try increasing sleep duration")
        # This isn't a hard failure of the feature, just a race; carry on.

    log("Stage 1 is running; starting stage 2 with --after-run")

    # Stage 2 blocks until stage 1 is done, then runs its own jobs
    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest2,
        "--command", "echo stage2 idx={idx}",
        "--state-dir", state_dir2,
        "--chunk-size", "5",
        "--max-concurrent", "10",
        "--max-retries", "0",
        "--max-runtime", "300",
        "--poll-interval", "5",
        "--time", "00:05:00",
        "--after-run", state_dir1,
    ])

    # Also wait for the stage 1 background process to finish
    try:
        stdout1, stderr1 = stage1_proc.communicate(timeout=60)
        for line in (stdout1 + stderr1).splitlines():
            log(f"  stage1: {line}")
    except subprocess.TimeoutExpired:
        stage1_proc.kill()
        log("FAIL: stage 1 background process timed out")
        return False

    if result.returncode != 0:
        log("FAIL: stage 2 exited with non-zero")
        _dump_debug_info(state_dir2)
        return False
    if stage1_proc.returncode != 0:
        log(f"FAIL: stage 1 exited with non-zero ({stage1_proc.returncode})")
        _dump_debug_info(state_dir1)
        return False

    state1 = load_state(state_dir1)
    state2 = load_state(state_dir2)

    completed1 = sum(1 for c in state1["chunks"].values()
                     if c["status"] == "completed")
    completed2 = sum(1 for c in state2["chunks"].values()
                     if c["status"] == "completed")

    log(f"Stage 1: {completed1}/{len(state1['chunks'])} chunks completed")
    log(f"Stage 2: {completed2}/{len(state2['chunks'])} chunks completed")

    if completed1 != len(state1["chunks"]):
        log("FAIL: stage 1 did not fully complete")
        return False
    if completed2 != len(state2["chunks"]):
        log("FAIL: stage 2 did not fully complete")
        return False
    if state2.get("failures"):
        log("FAIL: stage 2 had unexpected failures")
        return False

    # Verify ordering: stage 2 must not have submitted until stage 1 was done.
    # Compare sacct End times for stage 1 jobs against stage 2's submitted_at.
    stage1_job_ids = [
        c["slurm_job_id"] for c in state1["chunks"].values()
        if c.get("slurm_job_id")
    ]
    stage2_submitted_at = min(
        c["submitted_at"] for c in state2["chunks"].values()
        if c.get("submitted_at")
    )
    log(f"Stage 2 first submitted_at: {stage2_submitted_at}")

    if stage1_job_ids:
        sacct_result = subprocess.run(
            ["sacct", "--jobs=" + ",".join(stage1_job_ids),
             "--format=JobID,End", "--parsable2", "--noheader"],
            capture_output=True, text=True,
        )
        ordering_ok = True
        for line in sacct_result.stdout.splitlines():
            parts = line.strip().rstrip("|").split("|")
            if len(parts) < 2:
                continue
            job_id, end_time = parts[0], parts[1]
            # Skip batch/extern steps; only check array tasks (JobID contains _)
            if "_" not in job_id:
                continue
            if not end_time or end_time in ("Unknown", "None"):
                continue
            # Both timestamps are UTC ISO; strip timezone suffix for comparison
            end_norm = end_time.replace("T", " ")
            sub_norm = stage2_submitted_at[:19].replace("T", " ")
            log(f"  stage1 task {job_id} ended {end_norm}, stage2 submitted {sub_norm}")
            if end_norm > sub_norm:
                log(f"FAIL: stage 1 task {job_id} ended AFTER stage 2 submitted")
                ordering_ok = False
        if not ordering_ok:
            return False
        log("Ordering verified: all stage 1 jobs ended before stage 2 submitted")

    log("PASS")
    return True


def test_self_resubmit(project_dir):
    """Test: --self-resubmit causes a new monitor job to be submitted on max_runtime."""
    log("=" * 60)
    log("TEST: self_resubmit")
    log("=" * 60)

    tmpdir = make_shared_dir("test_self_resubmit")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    # Jobs sleep long enough that the first monitor hits max_runtime before they finish
    create_manifest(manifest, 6)

    # Submit in a background process with a short max_runtime and --self-resubmit.
    # The monitor will hit max_runtime, submit a resume job, and exit.
    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir
    cmd = [sys.executable, "-m", "slurmgrid", "submit",
           "--manifest", manifest,
           "--command", "sleep 15 && echo done idx={idx}",
           "--state-dir", state_dir,
           "--chunk-size", "3",
           "--max-concurrent", "6",
           "--max-retries", "0",
           "--max-runtime", "10",
           "--poll-interval", "3",
           "--time", "00:05:00",
           "--self-resubmit"]
    log(f"Submitting with --self-resubmit: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    # Wait for the initial monitor to exit (it should within ~15s)
    try:
        stdout, stderr = proc.communicate(timeout=60)
        for line in (stdout + stderr).splitlines():
            log(f"  monitor: {line}")
    except subprocess.TimeoutExpired:
        proc.kill()
        log("FAIL: initial monitor did not exit within 60s")
        return False

    if proc.returncode != 0:
        log(f"FAIL: initial monitor exited with non-zero ({proc.returncode})")
        return False

    # The initial monitor should have submitted a resume job — find it in sacct
    log("Waiting for resume job to appear in sacct...")
    resume_job_id = None
    for _ in range(30):
        time.sleep(2)
        result = subprocess.run(
            ["squeue", "--format=%i|%j", "--noheader"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2 and "_monitor" in parts[1]:
                resume_job_id = parts[0]
                log(f"Found resume job {resume_job_id} ({parts[1]})")
                break
        if resume_job_id:
            break

    if not resume_job_id:
        # Maybe it already completed — check sacct
        result = subprocess.run(
            ["sacct", "--format=JobID,JobName,State", "--parsable2", "--noheader",
             "--starttime=now-1hour"],
            capture_output=True, text=True,
        )
        for line in result.stdout.splitlines():
            parts = line.strip().split("|")
            if len(parts) >= 2 and "_monitor" in parts[1]:
                resume_job_id = parts[0]
                log(f"Resume job already completed: {resume_job_id} ({parts[1]})")
                break

    if not resume_job_id:
        log("FAIL: no resume job found in squeue or sacct")
        return False

    # Wait for the run to complete (the resume job monitors and finishes it)
    log("Waiting for run to complete...")
    for _ in range(60):
        time.sleep(5)
        if not os.path.exists(os.path.join(state_dir, "state.json")):
            continue
        state = load_state(state_dir)
        completed = sum(1 for c in state["chunks"].values()
                        if c["status"] == "completed")
        total = len(state["chunks"])
        log(f"  {completed}/{total} chunks completed")
        if completed == total:
            log("PASS")
            return True

    log("FAIL: run did not complete within timeout")
    _dump_debug_info(state_dir)
    return False


def test_cancel_and_resume(project_dir):
    """Test: cancel active jobs, then resume to resubmit and complete them."""
    log("=" * 60)
    log("TEST: cancel_and_resume")
    log("=" * 60)

    tmpdir = make_shared_dir("test_cancel_resume")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    create_manifest(manifest, 6)

    # Submit jobs that sleep long enough for us to cancel them.
    env = os.environ.copy()
    env["PYTHONPATH"] = project_dir
    cmd = [sys.executable, "-m", "slurmgrid", "submit",
           "--manifest", manifest,
           "--command", "sleep 30 && echo done idx={idx}",
           "--state-dir", state_dir,
           "--chunk-size", "3",
           "--max-concurrent", "6",
           "--max-retries", "1",
           "--max-runtime", "20",
           "--poll-interval", "3",
           "--time", "00:05:00"]
    log(f"Submitting long-running jobs: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)

    # Wait for state file to appear (jobs have been submitted)
    state_json = os.path.join(state_dir, "state.json")
    for _ in range(60):
        if os.path.isfile(state_json):
            state = load_state(state_dir)
            submitted = sum(1 for c in state["chunks"].values()
                            if c.get("slurm_job_id"))
            if submitted > 0:
                break
        time.sleep(1)
    else:
        proc.kill()
        log("FAIL: jobs never submitted")
        return False

    log(f"Jobs submitted ({submitted} chunks), waiting for monitor to exit...")

    # Wait for the monitor to exit (via max_runtime)
    try:
        stdout, stderr = proc.communicate(timeout=60)
        for line in (stdout + stderr).splitlines():
            log(f"  monitor: {line}")
    except subprocess.TimeoutExpired:
        proc.kill()
        log("FAIL: monitor did not exit within 60s")
        return False

    # Now cancel all active jobs
    log("Running slurmgrid cancel...")
    result = run_slurmgrid(project_dir, [
        "cancel", "--state-dir", state_dir,
    ])
    if result.returncode != 0:
        log("FAIL: cancel command failed")
        return False

    # Verify state: chunks should be "cancelled", not "completed"
    state = load_state(state_dir)
    cancelled = sum(1 for c in state["chunks"].values()
                    if c["status"] == "cancelled")
    log(f"Cancelled chunks: {cancelled}")
    if cancelled == 0:
        log("FAIL: no chunks marked as cancelled")
        _dump_debug_info(state_dir)
        return False

    # Now resume with a fast command (the original sleep 30 command is baked
    # into the sbatch scripts, so we can't change it). Instead, we verify
    # that resume picks up the cancelled chunks and resubmits them.
    # Use a short max_runtime so we don't wait forever for the sleep 30 jobs.
    log("Running slurmgrid resume...")
    result = run_slurmgrid(project_dir, [
        "resume", "--state-dir", state_dir,
        "--max-runtime", "120",
        "--poll-interval", "5",
    ], timeout=180)

    if result.returncode != 0:
        log("FAIL: resume command failed")
        _dump_debug_info(state_dir)
        return False

    # After resume, all chunks should be completed
    state = load_state(state_dir)
    completed = sum(1 for c in state["chunks"].values()
                    if c["status"] == "completed")
    total = len(state["chunks"])
    log(f"After resume: {completed}/{total} chunks completed")

    # The original chunks get resubmitted (sleep 30 + echo), plus retry
    # chunks may be created for tasks that were CANCELLED. Check that
    # everything eventually finishes.
    still_cancelled = sum(1 for c in state["chunks"].values()
                          if c["status"] == "cancelled")
    if still_cancelled > 0:
        log(f"FAIL: {still_cancelled} chunks still cancelled after resume")
        _dump_debug_info(state_dir)
        return False

    # All original + retry chunks should be in a terminal state
    non_terminal = sum(1 for c in state["chunks"].values()
                       if c["status"] not in ("completed", "partial_failure"))
    if non_terminal > 0:
        log(f"FAIL: {non_terminal} chunks not in terminal state")
        _dump_debug_info(state_dir)
        return False

    log("PASS")
    return True


def test_restart_submit(project_dir):
    """Test: --restart on submit backs up the old state dir and starts fresh."""
    log("=" * 60)
    log("TEST: restart_submit")
    log("=" * 60)

    tmpdir = make_shared_dir("test_restart")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    create_manifest(manifest, 4)

    # First submit: run 4 jobs to completion
    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", "echo first idx={idx}",
        "--state-dir", state_dir,
        "--chunk-size", "4",
        "--max-concurrent", "10",
        "--max-retries", "0",
        "--max-runtime", "120",
        "--poll-interval", "5",
        "--time", "00:05:00",
    ])

    if result.returncode != 0:
        log("FAIL: first submit exited with non-zero")
        _dump_debug_info(state_dir)
        return False

    # Confirm state dir exists and is complete
    state = load_state(state_dir)
    completed = sum(1 for c in state["chunks"].values() if c["status"] == "completed")
    if completed != len(state["chunks"]):
        log("FAIL: first run did not fully complete")
        return False

    # Note the job IDs from the first run to distinguish from the second run
    first_run_job_ids = {
        c["slurm_job_id"] for c in state["chunks"].values() if c.get("slurm_job_id")
    }
    log(f"First run job IDs: {first_run_job_ids}")

    # Second submit with --restart: should back up old state and start fresh
    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", "echo second idx={idx}",
        "--state-dir", state_dir,
        "--chunk-size", "4",
        "--max-concurrent", "10",
        "--max-retries", "0",
        "--max-runtime", "120",
        "--poll-interval", "5",
        "--time", "00:05:00",
        "--restart",
    ])

    if result.returncode != 0:
        log("FAIL: restart submit exited with non-zero")
        _dump_debug_info(state_dir)
        return False

    # A backup directory should have been created next to state_dir
    parent = os.path.dirname(state_dir)
    backups = [
        d for d in os.listdir(parent)
        if d.startswith("state.bak.")
    ]
    if not backups:
        log("FAIL: no backup directory (state.bak.*) found after --restart")
        return False
    log(f"Backup directory created: {backups[0]}")

    # New state should be a fresh run (different job IDs)
    new_state = load_state(state_dir)
    new_job_ids = {
        c["slurm_job_id"] for c in new_state["chunks"].values()
        if c.get("slurm_job_id")
    }
    log(f"Second run job IDs: {new_job_ids}")
    if new_job_ids & first_run_job_ids:
        log("FAIL: second run reused job IDs from first run (state was not reset)")
        return False

    # Second run should have completed successfully
    completed2 = sum(
        1 for c in new_state["chunks"].values() if c["status"] == "completed"
    )
    if completed2 != len(new_state["chunks"]):
        log(f"FAIL: second run did not fully complete ({completed2}/{len(new_state['chunks'])})")
        _dump_debug_info(state_dir)
        return False

    log("PASS")
    return True


def test_failures_subcommand(project_dir):
    """Test: 'failures' subcommand lists permanently-failed jobs with details."""
    log("=" * 60)
    log("TEST: failures_subcommand")
    log("=" * 60)

    tmpdir = make_shared_dir("test_failures_cmd")
    manifest = os.path.join(tmpdir, "manifest.csv")
    state_dir = os.path.join(tmpdir, "state")
    # 6 jobs; indices 1 and 4 always fail (max_retries=0 → permanently failed)
    create_failing_manifest(manifest, 6, fail_indices={1, 4})

    result = run_slurmgrid(project_dir, [
        "submit",
        "--manifest", manifest,
        "--command", "exit {should_fail}",
        "--state-dir", state_dir,
        "--chunk-size", "6",
        "--max-concurrent", "10",
        "--max-retries", "0",
        "--max-runtime", "120",
        "--poll-interval", "5",
        "--time", "00:05:00",
    ])

    if result.returncode != 0:
        log("FAIL: initial submit exited with non-zero")
        _dump_debug_info(state_dir)
        return False

    state = load_state(state_dir)
    perm_failed = sum(
        1 for f in state.get("failures", {}).values() if f["permanently_failed"]
    )
    if perm_failed != 2:
        log(f"FAIL: expected 2 permanently failed before testing, got {perm_failed}")
        _dump_debug_info(state_dir)
        return False

    # Run the failures subcommand
    result = run_slurmgrid(project_dir, [
        "failures", "--state-dir", state_dir, "--paths-only",
    ])

    if result.returncode != 0:
        log("FAIL: failures subcommand exited with non-zero")
        return False

    output = result.stdout
    log(f"failures output:\n{output}")

    # Should show 2 failure entries
    separator_count = output.count("=" * 60)
    if separator_count < 2:
        log(f"FAIL: expected at least 2 failure entries, got {separator_count} separators")
        return False

    # Should display row indices 1 and 4 (our failing rows)
    if "Row 1" not in output:
        log("FAIL: Row 1 not found in failures output")
        return False
    if "Row 4" not in output:
        log("FAIL: Row 4 not found in failures output")
        return False

    # Should show exit code and permanent=True
    if "exit=1" not in output:
        log("FAIL: exit=1 not found in failures output")
        return False
    if "permanent=True" not in output:
        log("FAIL: permanent=True not found in failures output")
        return False

    # Should show log file paths
    if "OUT:" not in output or "ERR:" not in output:
        log("FAIL: log file paths (OUT:/ERR:) not found in failures output")
        return False

    # With --paths-only, should NOT show the "--- last N lines ---" section
    if "--- last" in output:
        log("FAIL: --paths-only should suppress err log content but found '--- last'")
        return False

    # Test --permanently-failed-only filter (should show same 2 failures)
    result2 = run_slurmgrid(project_dir, [
        "failures", "--state-dir", state_dir, "--paths-only",
        "--permanently-failed-only",
    ])
    if result2.returncode != 0:
        log("FAIL: failures --permanently-failed-only exited with non-zero")
        return False
    if result2.stdout.count("=" * 60) != separator_count:
        log("FAIL: --permanently-failed-only showed different count than expected")
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
        test_failures_subcommand,
        test_cancel_and_resume,
        test_reset_failures_with_overrides,
        test_restart_submit,
        test_after_run,
        test_self_resubmit,
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
