#!/bin/bash
# Online pick loop, one seed per array task: 20 iterations, s_1 band 4-6 cm,
# 20 fallback rounds (default). Submit from the code folder:
#   cd /scratch/st-sdutta10-1/smallick/recovery_self_imitation && sbatch /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc/slurm/run_array.sh
# Single seeds / a subset:  add --array=3,7 after sbatch
# Results: $RES/arc_band4_6_20iters/seed<N>/ (iter01..iter20, iterations.csv, D.npz, fallback.log)
#SBATCH --job-name=arc_band4_6_20it
#SBATCH --account=st-sdutta10-1-gpu
#SBATCH --array=1-20
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --gpus-per-node=1
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/st-sdutta10-1/smallick/recovery_self_imitation/logs/%x_seed%a_%A.out
#SBATCH --error=/scratch/st-sdutta10-1/smallick/recovery_self_imitation/logs/%x_seed%a_%A.out

set -eo pipefail      # no -u: conda activate scripts read unset variables
source /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc/slurm/env.sh
SEED=${SLURM_ARRAY_TASK_ID}
OUT="$RES/arc_band4_6_20iters/seed$SEED"
echo "seed $SEED | host $(hostname) | GPU ${CUDA_VISIBLE_DEVICES:-none} | start $(date) | out $OUT"

python "$CODE/fallback_lift.py" --seed "$SEED" --n-iters 20 --max-rounds 20 \
    --s1-min-dist 0.04 --s1-max-dist 0.06 --outdir "$OUT"
echo "seed $SEED done $(date)"
