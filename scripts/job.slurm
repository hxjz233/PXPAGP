#!/bin/bash
#SBATCH --job-name=pxp-hxz
#SBATCH --output=../log/%x-%A/%a.out
#SBATCH --error=../log/%x-%A/%a.err
#SBATCH --time=47:59:59
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=34000M
#SBATCH --ntasks=1
#SBATCH --array=0
#SBATCH --partition=bigmem

source ~/.bashrc
conda activate quspin

cd "$SLURM_SUBMIT_DIR"

count=1

python -u ../src/pxp_agp_scaling.py \
	--mode hxz \
	--inv-sector 0 \
	--l-values 22 \
	--hxz-min 0.0 \
	--hxz-max 0.25 \
	--hxz-count $count \
	--hxz-shard-count $count \
	--hxz-shard-index "$SLURM_ARRAY_TASK_ID" \
	--force

# After the array completes, run the same script once more with:
# python -u ../src/pxp_agp_scaling.py --mode hxz --l-values 10 12 14 16 --hxz-min 0.0 --hxz-max 0.25 --hxz-count $count --hxz-shard-count $count --collect-hxz-shards --output fig/pxp_agp_scaling.png
