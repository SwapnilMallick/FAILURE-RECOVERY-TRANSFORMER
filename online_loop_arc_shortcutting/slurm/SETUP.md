# Running `online_loop_arc` on ARC Sockeye

The pick-task online loop runs 20 iterations for each of seeds 1–20, with the s₁ band at 4–6 cm and 20 fallback rounds. There is one SLURM array task per seed, each on 1 GPU.

| What | Where |
|---|---|
| Allocation | `st-sdutta10-1`; GPU jobs use the account `st-sdutta10-1-gpu` |
| Code | `/arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc` |
| Conda env | `/arc/project/st-sdutta10-1/smallick/envs/robosuite` |
| DINOv2 cache | `/arc/project/st-sdutta10-1/smallick/torch_cache/hub/` |
| Results and logs | `/scratch/st-sdutta10-1/smallick/recovery_self_imitation/` |

Compute nodes are assumed to have **no internet**. Sockeye only accepts job submissions from `/scratch`, and `/arc/project` is read-only on the compute nodes: the jobs read the code from `/arc/project` by absolute path, and write all caches and results to `/scratch` (see `env.sh`). Everything is installed and copied from the login node or from your Mac. All paths are set in `slurm/env.sh`; edit them only there.

---

## 1. Copy the code and the DINOv2 cache (on your Mac)

```bash
cd /Users/swapnilmallick/Desktop/failure_recovery
ssh smallick@sockeye.arc.ubc.ca "mkdir -p /arc/project/st-sdutta10-1/smallick/recovery_self_imitation /arc/project/st-sdutta10-1/smallick/torch_cache/hub/checkpoints /scratch/st-sdutta10-1/smallick/recovery_self_imitation/logs"
rsync -av --exclude __pycache__ online_loop_arc smallick@sockeye.arc.ubc.ca:/arc/project/st-sdutta10-1/smallick/recovery_self_imitation/
rsync -av ~/.cache/torch/hub/facebookresearch_dinov2_main smallick@sockeye.arc.ubc.ca:/arc/project/st-sdutta10-1/smallick/torch_cache/hub/
rsync -av ~/.cache/torch/hub/checkpoints/dinov2_vits14_pretrain.pth smallick@sockeye.arc.ubc.ca:/arc/project/st-sdutta10-1/smallick/torch_cache/hub/checkpoints/
```

After any later code change, rerun only the `online_loop_arc` rsync line.

## 2. Create the conda environment (on the Sockeye login node, which has internet)

```bash
ssh smallick@sockeye.arc.ubc.ca
module load gcc/9.4.0 miniconda3/4.9.2
eval "$(conda shell.bash hook)"
conda create -y -p /arc/project/st-sdutta10-1/smallick/envs/robosuite python=3.10
conda activate /arc/project/st-sdutta10-1/smallick/envs/robosuite
pip install torch==2.12.1 torchvision==0.27.1
pip install -r /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc/slurm/requirements.txt
python -c "import torch, mujoco, robosuite; print(torch.__version__, mujoco.__version__, robosuite.__version__)"
```

These are the same versions as the local `robosuite` env. **Check GPU support in step 3.** Sockeye's GPUs are V100s. If the smoke log shows `CUDA available: False`, or a "no kernel image" error, reinstall torch from a CUDA build that still supports V100 (sm_70). For example:

```bash
pip install --force-reinstall torch==2.12.1 torchvision==0.27.1 --index-url https://download.pytorch.org/whl/cu126
```

The CUDA tag must exist for that torch version. See https://pytorch.org/get-started/previous-versions/.

## 3. Smoke test: one GPU node, about 10 minutes

```bash
cd /scratch/st-sdutta10-1/smallick/recovery_self_imitation      # Sockeye: submit jobs from /scratch, never /arc/project
sbatch /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc/slurm/smoke.sh
squeue -u $USER
tail -f /scratch/st-sdutta10-1/smallick/recovery_self_imitation/logs/arc_smoke_*.out
```

It must print all of these:
- `CUDA available: True`
- `GPU matmul ok: True`
- `EGL render ok: (64, 64, 3) …`
- `DINOv2 loaded offline on cuda`
- a 2-iteration smoke run ending in `SMOKE OK`

If rendering fails, check that `MUJOCO_GL=egl` is set (it is, in `env.sh`). If it still fails, ask ARC support whether the GPU nodes provide `libEGL`.

## 4. Submit the 20-seed batch

```bash
cd /scratch/st-sdutta10-1/smallick/recovery_self_imitation
sbatch /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc/slurm/run_array.sh                 # seeds 1-20
# sbatch --array=5,7 /arc/project/st-sdutta10-1/smallick/recovery_self_imitation/online_loop_arc/slurm/run_array.sh   # only some seeds (e.g. a rerun)
```

Monitoring:

```bash
squeue -u $USER                                    # running / pending tasks
sacct -j <JOBID> --format=JobID,State,Elapsed,MaxRSS
tail -n 30 /scratch/st-sdutta10-1/smallick/recovery_self_imitation/logs/arc_band4_6_20it_seed1_*.out
grep -h "iteration\|stop=" /scratch/st-sdutta10-1/smallick/recovery_self_imitation/arc_band4_6_20iters/seed*/fallback.log | tail
```

Each task requests 6 CPUs, 24 GB and 1 GPU, with a 36 h time limit. That limit is a guess: there are no cluster timings yet. Later iterations take longer, because BC builds corrective examples for every new τ_k. After the first few iterations, compare the `Elapsed` time and adjust `--time` if needed.

**A task that is killed starts again from iteration 1.** The loop has no checkpoint or resume.

## 5. Copy the results back (on your Mac)

```bash
cd /Users/swapnilmallick/Desktop/failure_recovery
rsync -av smallick@sockeye.arc.ubc.ca:/scratch/st-sdutta10-1/smallick/recovery_self_imitation/arc_band4_6_20iters pick_results/online/
```

Scratch is not backed up and may be purged, so copy the results back once the batch finishes.

## What `online_loop_arc` changes relative to `online_loop_v6`

- **τ_k phase labels:** a successful trajectory used to be labelled "approach" throughout. Now the fallback part is labelled approach, and the BC part comes from `bc_action_phases`, using its gripper commands and heights:
  - open near or above the cube = descend;
  - the final close = grasp, then lift once the gripper rises;
  - earlier closes that were reopened = grasp.
  
  Only approach and descend states get corrective examples, so from iteration 2 on, BC no longer gets open-gripper corrective examples around τ_k's grasp and lift.
- **Offline DINOv2:** if `DINOV2_REPO` is set (as in `env.sh`), the encoder loads from that local repo. The weights come from `$TORCH_HOME/hub/checkpoints`.
- **`slurm/`:** this guide, `env.sh`, `smoke.sh`, `run_array.sh` and `requirements.txt`.
