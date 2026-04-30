#!/bin/bash
#SBATCH --job-name=panoptic_eval  # Job name
#SBATCH --partition=gpu4            # Partition: 'gpu4' (A100) or 'gpu' (V100)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=32         # CPU cores per task (adjust based on availability)
#SBATCH --mem=185G                 # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs per node (1-4 for gpu4, 1-4 for gpu)
#SBATCH --time=9:30:00            # Time limit hrs:min:sec
#SBATCH --output=slurm_%j.out      # Standard output log
#SBATCH --error=slurm_%j.err       # Standard error log

# =============================================================================
# PER-SCENE EVALUATION SCRIPT FOR MULTI-VIEW MASK2FORMER ON SCANNET++
#
# Discovers all scenes in panoptic_val directory (48 val scenes).
# For each scene, picks a random reference view (overlap-aware selection
# from m2f_inference.py), runs single-view inference, computes per-scene
# PQ/SQ/RQ, then macro-averages across all scenes.
#
# Usage:
#   sbatch eval_job.sh
#
# To override the checkpoint path:
#   sbatch --export=ALL,CHECKPOINT=/path/to/model_final.pth eval_job.sh
#
# To set a reproducible seed for reference view selection:
#   sbatch --export=ALL,SEED=42 eval_job.sh
# =============================================================================

# 1. Load necessary modules (adjust versions if needed)
#module purge
#module load cuda/11.8

export PATH="/home/sramam2s/.conda/envs/robot_lab/bin:$PATH"

# 2. Verify it worked
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Load CUDA module
module load cuda/11.8

# Print environment info for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi
# =============================================================================
# DEBUGGING & STABILITY FLAGS
# =============================================================================
# Fixes "device busy/unavailable" errors on A100/complex topologies
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=INFO

# Better error reporting
export CUDA_LAUNCH_BLOCKING=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL

# =============================================================================
# EVALUATION LAUNCH
# =============================================================================

# Automatically detect number of GPUs assigned by Slurm
# Count the number of commas in CUDA_VISIBLE_DEVICES and add 1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
fi

echo "Detected $NUM_GPUS GPUs. Launching evaluation..."

# Set paths for the new cluster
export MASK2FORMER_ROOT=/work/sramam2s/Mask2Former
export SCANNETPP_ROOT=${MASK2FORMER_ROOT}/datasets/scannet/scannetpp
export PANOPTIC_ROOT=${SCANNETPP_ROOT}/panoptic
export PANOPTIC_VAL_ROOT=${SCANNETPP_ROOT}/panoptic_val

cd ${MASK2FORMER_ROOT}

# ===========================
# CONFIGURE THESE PATHS
# ===========================
# Path to the config file used during training
CONFIG_FILE="configs/scannetpp/panoptic-segmentation/ma40.yaml"

# Path to the trained checkpoint
# Override via environment variable CHECKPOINT or edit here
CHECKPOINT=${CHECKPOINT:-"output_multiview_8a40_warped_attention/model_final.pth"}

# Output directory for evaluation results
OUTPUT_DIR=${OUTPUT_DIR:-"eval_results/$(date +%Y%m%d_%H%M%S)"}

# Random seed for reference view selection (None = random each run)
# Override via environment variable SEED or edit here
SEED=${SEED:-"42"}

echo ""
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Panoptic val root: ${PANOPTIC_VAL_ROOT}"
echo "Seed: ${SEED:-None (random)}"
echo ""

# Build optional seed argument
SEED_ARG=""
if [ -n "$SEED" ]; then
    SEED_ARG="--seed ${SEED}"
fi

# Run per-scene evaluation
python m2f_evaluate.py \
    --config-file ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT} \
    --output-dir ${OUTPUT_DIR} \
    --scannetpp-root ${SCANNETPP_ROOT} \
    --panoptic-val-root ${PANOPTIC_VAL_ROOT} \
    --target-short-edge 480 \
    --max-size 640 \
    --num-views 3 \
    --overlap-threshold 0.6 \
    --object-mask-threshold 0.6 \
    --save-predictions \
    ${SEED_ARG} \
    MODEL.WEIGHTS ${CHECKPOINT}

echo ""
echo "========================================="
echo "Evaluation completed at $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================="
