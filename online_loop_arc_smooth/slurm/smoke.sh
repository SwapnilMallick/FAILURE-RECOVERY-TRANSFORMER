#!/bin/bash
# Smoke test on ONE GPU node: CUDA, EGL rendering, offline DINOv2, then a tiny
# 2-iteration run of the pick loop. Submit from the code folder:
#   cd /scratch/st-sdutta10-1/smallick/recovery_self_imitation && sbatch /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc_smooth/slurm/smoke.sh
#SBATCH --job-name=arc_smoke
#SBATCH --account=st-sdutta10-1-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/st-sdutta10-1/smallick/recovery_self_imitation/logs/%x_%j.out
#SBATCH --error=/scratch/st-sdutta10-1/smallick/recovery_self_imitation/logs/%x_%j.out

set -eo pipefail      # no -u: conda activate scripts read unset variables
source /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc_smooth/slurm/env.sh
echo "host $(hostname) | GPU(s) ${CUDA_VISIBLE_DEVICES:-none} | $(date)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true

python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.randn(512, 512, device="cuda"); print("GPU matmul ok:", float((x @ x).sum()) == float((x @ x).sum()),
                                                   "|", torch.cuda.get_device_name(0))
import mujoco, robosuite; print("mujoco", mujoco.__version__, "| robosuite", robosuite.__version__)
PY

python - <<'PY'
import os, sys; sys.path.insert(0, os.environ["CODE"])
from lift_env import LiftEnv
env = LiftEnv(seed=1, cube_offset=(0.10, 0.0), enable_render=True)
img = env.render("agentview", 64, 64); print("EGL render ok:", img.shape, img.dtype, "mean", float(img.mean()))
from image_encoder import ImageEncoder, EncoderConfig
enc = ImageEncoder(EncoderConfig()); print("DINOv2 loaded offline on", enc.device)
PY

python "$CODE/fallback_lift.py" --smoke --seed 1 --n-iters 2 --s1-min-dist 0.04 --s1-max-dist 0.10 \
    --outdir "$RES/smoke/seed1"
echo "SMOKE OK $(date)"
