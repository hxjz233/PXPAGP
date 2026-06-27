#!/bin/bash
set -euo pipefail

script_dir="$(dirname "$0")"

if ! sbatch_out=$(env -u SHELLOPTS sbatch --hold "$script_dir/job.slurm" 2>&1); then
    printf '%s\n' "$sbatch_out" >&2
    exit 1
fi

# Print original sbatch/job_submit output
printf '%s\n' "$sbatch_out"

jobid=$(printf '%s\n' "$sbatch_out" | sed -n 's/^Submitted batch job \([0-9][0-9]*\).*/\1/p')

if [[ -z "${jobid:-}" ]]; then
    echo "Could not determine job id from sbatch output" >&2
    exit 1
fi

mkdir -p "$script_dir/../log/pxp-hxz-$jobid"
scontrol release "$jobid"