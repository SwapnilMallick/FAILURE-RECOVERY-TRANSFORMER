# Sourced by the SLURM job scripts. Edit the paths / module names here only.
# Paths (from the user, 2026-10-24)
export PROJ=/arc/project/st-sdutta10-1/smallick/recovery_self_imitation      # code: $PROJ/online_loop_arc
export RES=/scratch/st-sdutta10-1/smallick/recovery_self_imitation           # results + job logs
export CONDA_ENV=/arc/project/st-sdutta10-1/smallick/envs/robosuite          # conda env (created by SETUP.md step 2)
export TORCH_HOME=/arc/project/st-sdutta10-1/smallick/torch_cache            # holds hub/checkpoints/dinov2_vits14_pretrain.pth
export DINOV2_REPO=$TORCH_HOME/hub/facebookresearch_dinov2_main             # local DINOv2 repo copy (no internet on nodes)

# Modules (Sockeye, checked 2026-10-24 with `module spider conda`): miniconda3 needs a gcc module first
module load gcc/9.4.0 miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"

# Headless GPU rendering (MuJoCo + robosuite) through EGL on the job's GPU
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
_gpu=${CUDA_VISIBLE_DEVICES:-0}
export MUJOCO_EGL_DEVICE_ID=${_gpu%%,*}

# CPU threads = cores given to the job
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
export PYTHONUNBUFFERED=1

# /arc/project is not writable from the compute nodes (numba could not cache
# robosuite's jitted functions next to site-packages): put caches on scratch
export NUMBA_CACHE_DIR=$RES/cache/numba
export MPLCONFIGDIR=$RES/cache/matplotlib
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$NUMBA_CACHE_DIR" "$MPLCONFIGDIR"
# Jobs run from scratch (Sockeye: no job submission / working directory in /arc/project);
# the code is read from $PROJ by absolute path (/arc/project is read-only on the nodes)
export CODE=$PROJ/online_loop_arc_offrl_twobuffer
cd "$RES"
