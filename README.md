# slurmgrid

[![CI](https://github.com/jgaeb/slurmgrid/actions/workflows/ci.yml/badge.svg)](https://github.com/jgaeb/slurmgrid/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/jgaeb/slurmgrid/graph/badge.svg)](https://codecov.io/gh/jgaeb/slurmgrid)
[![PyPI](https://img.shields.io/pypi/v/slurmgrid)](https://pypi.org/project/slurmgrid/)

Manage large Slurm job arrays that exceed your cluster's submission limit.

If you need to run 50,000 small jobs but your cluster caps `MaxArraySize` at
10,000 (or limits total queued jobs), `slurmgrid` handles the tedious cycle of
"submit a batch, wait, submit the next batch" automatically. It chunks your
parameter manifest, submits array jobs via `sbatch`, monitors completion via
`sacct`, retries failures, and persists state so you can resume if interrupted.

## Installation

```bash
pip install slurmgrid
```

Or just clone the repo and run directly (no dependencies beyond Python 3.8+):

```bash
git clone https://github.com/jgaeb/slurmgrid.git
cd slurmgrid
python -m slurmgrid --help
```

## Quick start

1. Create a manifest file (CSV or TSV) with one row per job:

```csv
alpha,beta,seed
0.1,1,42
0.1,2,42
0.5,1,42
0.5,2,42
...
```

2. Run `slurmgrid submit` with your command template:

```bash
python -m slurmgrid submit \
  --manifest params.csv \
  --command "python train.py --alpha {alpha} --beta {beta} --seed {seed}" \
  --partition gpu \
  --time 01:00:00 \
  --mem 4G \
  --max-concurrent 5000
```

That's it. `slurmgrid` will:
- Shuffle and split the manifest into chunks (default: 1/3 of `MaxArraySize`)
- Submit each chunk as a fast array job via `sbatch`, using Slurm's `%throttle`
  to limit concurrency to `--max-concurrent`
- Poll `sacct` every 30 seconds to track completion
- Submit the next chunk when the current one finishes
- Batch failed jobs into retry chunks (up to `--max-retries`, default 3)
- Save state to disk after every poll so you can resume if interrupted

## Usage

### Submit a new run

```bash
python -m slurmgrid submit \
  --manifest params.csv \
  --command "python train.py --alpha {alpha} --beta {beta}" \
  --state-dir ./my_run \
  --partition gpu \
  --time 02:00:00 \
  --mem 8G \
  --cpus-per-task 4 \
  --max-concurrent 5000 \
  --max-retries 3 \
  --poll-interval 30 \
  --preamble "module load python/3.10 && conda activate myenv"
```

The `--command` template uses `{column_name}` placeholders that are resolved
from the manifest columns. Any column in the manifest can be referenced.

### Resume an interrupted run

If you lose your SSH session or Ctrl-C out, running Slurm jobs continue
independently. Resume monitoring with:

```bash
python -m slurmgrid resume --state-dir ./my_run
```

### Check status

```bash
python -m slurmgrid status --state-dir ./my_run
```

```
==================================================
  Total jobs:            50000
  Completed:             35420  (70.8%)
  Active:                 4580
  Pending:               10000
  Failed (retrying):         0
  Failed (final):            0
  Chunks: 35/50 completed, 5 active, 10 pending
==================================================
```

### Cancel all jobs

```bash
python -m slurmgrid cancel --state-dir ./my_run
```

### Dry run

Generate all chunk files and sbatch scripts without actually submitting:

```bash
python -m slurmgrid submit --manifest params.csv --command "echo {x}" --dry-run
```

Inspect the generated scripts in `./sc_state/scripts/` to verify correctness.

## How it works

1. **Chunking**: The manifest is split into sub-manifests. Each chunk gets its
   own sbatch script that uses `SLURM_ARRAY_TASK_ID` to index into the
   sub-manifest and extract the parameters for that task.

2. **Shuffling**: Manifest rows are shuffled before chunking (disable with
   `--no-shuffle`) so each chunk gets a representative mix of the parameter
   space and chunks take roughly the same wall time.

3. **Batch submission**: Each chunk is submitted as a single `sbatch --array`
   call with a `%throttle` suffix to limit concurrency, which is orders of
   magnitude faster than submitting jobs individually.

4. **Monitoring**: The tool polls `sacct` to track job status. One chunk runs
   at a time; when it finishes, the next is submitted.

5. **Retries**: When all regular chunks are done, failed tasks are batched
   into a single retry chunk and resubmitted, up to `--max-retries` per task.

6. **State persistence**: All state is saved as JSON after every poll.
   Atomic writes (via temp file + rename) prevent corruption. You can
   resume at any time.

## State directory layout

```
sc_state/
  config.json          # Frozen copy of the submission configuration
  state.json           # Chunk-level status and failure tracking
  slurmgrid.log         # Tool's own log file
  chunks/
    chunk_000.chunk    # Sub-manifests (internal format)
    chunk_001.chunk
  scripts/
    chunk_000.sh       # Generated sbatch scripts
    chunk_001.sh
  logs/
    chunk_000/         # Slurm stdout/stderr per chunk
      slurm-12345_0.out
      slurm-12345_0.err
```

## All options

| Flag | Default | Description |
|------|---------|-------------|
| `--manifest` | (required) | CSV/TSV manifest file |
| `--command` | (required) | Command template with `{column}` placeholders |
| `--state-dir` | `./sc_state` | Directory for state, chunks, scripts, logs |
| `--delimiter` | auto-detect | Manifest delimiter (`,` for .csv, `\t` for .tsv) |
| `--chunk-size` | auto-detect | Jobs per array chunk (default: `MaxArraySize / 3`) |
| `--max-concurrent` | 10000 | Max simultaneously running tasks (Slurm `%throttle`) |
| `--max-retries` | 3 | Max retries per failed job |
| `--poll-interval` | 30 | Seconds between status checks |
| `--max-runtime` | unlimited | Max seconds to run before saving state and exiting |
| `--dry-run` | false | Generate scripts without submitting |
| `--no-shuffle` | false | Don't shuffle manifest rows before chunking |
| `--partition` | | Slurm partition |
| `--time` | | Wall time limit (e.g., `01:00:00`) |
| `--mem` | | Memory per node (e.g., `4G`) |
| `--mem-per-cpu` | | Memory per CPU |
| `--cpus-per-task` | 1 | CPUs per task |
| `--gpus` | | GPU specification |
| `--gres` | | Generic resource specification |
| `--account` | | Slurm account |
| `--qos` | | Quality of service |
| `--constraint` | | Node constraint |
| `--exclude` | | Nodes to exclude |
| `--job-name-prefix` | `sc` | Prefix for Slurm job names |
| `--preamble` | | Shell commands before the main command |
| `--preamble-file` | | File containing preamble commands |
| `--extra-sbatch` | | Extra `#SBATCH` flags (repeatable) |

## Requirements

- Python 3.8+ (stdlib only, no external dependencies)
- Slurm with `sbatch`, `sacct`, `squeue`, `scancel`, `scontrol` available
- Slurm accounting enabled (`sacct` must work)

## License

MIT
