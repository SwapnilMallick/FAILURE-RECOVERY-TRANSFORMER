#!/bin/bash
# arc_offrl_dense: IQL reward = pbrs (robosuite Lift dense terms; see robosuite_reward_shaping.md) + round-1 unstuck fix (lowest-U candidate when all rejected; hand-off at s_1 if all collided)
# Online pick loop, one seed per array task: 50 iterations, s_1 band 4-6 cm,
# 20 fallback rounds (default). Submit from the code folder:
#   cd /scratch/st-sdutta10-1/smallick/recovery_self_imitation && sbatch /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc_offrl_dense_unstuck/slurm/run_pbrs.sh
# Single seeds / a subset:  add --array=3,7 after sbatch
# Variants: optional arguments after the script path = RUN_NAME, then extra fallback_lift.py flags, e.g.
#   sbatch --job-name=arc_iqlpbrs_us_b46_8x .../run_pbrs.sh steps8x --bc-max-steps-mult 8
#   -> results in $RES/arc_iqlpbrs_unstuck_band4_6_50iters_steps8x/seed<N>/ (no arguments: $RES/arc_iqlpbrs_unstuck_band4_6_50iters/)
# Results: $RES/arc_iqlpbrs_unstuck_band4_6_50iters/seed<N>/ (iter01..iter50, iterations.csv, D.npz, fallback.log)
#SBATCH --job-name=arc_iqlpbrs_us_b46
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
source /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc_offrl_dense_unstuck/slurm/env.sh
SEED=${SLURM_ARRAY_TASK_ID}
RUN_NAME=${1:-}
[ $# -gt 0 ] && shift
OUT="$RES/arc_iqlpbrs_unstuck_band4_6_50iters${RUN_NAME:+_$RUN_NAME}/seed$SEED"
echo "seed $SEED | host $(hostname) | GPU ${CUDA_VISIBLE_DEVICES:-none} | start $(date) | out $OUT | extra flags: $*"

python "$CODE/fallback_lift.py" --seed "$SEED" --n-iters 50 --max-rounds 20 \
    --s1-min-dist 0.04 --s1-max-dist 0.06 --iql-reward pbrs --outdir "$OUT" "$@"
echo "seed $SEED done $(date)"
