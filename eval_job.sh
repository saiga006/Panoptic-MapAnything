#!/bin/bash
#SBATCH --job-name=panoptic_eval  # Job name
#SBATCH --partition=gpu4            # Partition: 'gpu4' (A100) or 'gpu' (V100)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --cpus-per-task=32         # CPU cores per task (adjust based on availability)
#SBATCH --mem=185G                 # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs per node (1-4 for gpu4, 1-4 for gpu)
#SBATCH --time=0:30:00            # Time limit hrs:min:sec
#SBATCH --output=slurm_%j.out      # Standard output log
#SBATCH --error=slurm_%j.err       # Standard error log

# =============================================================================
# EVALUATION SCRIPT FOR MULTI-VIEW MASK2FORMER ON SCANNET++ VALIDATION SET
#
# Computes PQ, SQ, RQ metrics on panoptic_val annotations.
#
# Usage:
#   sbatch eval_job.sh
#
# To override the checkpoint path:
#   sbatch --export=ALL,CHECKPOINT=/path/to/model_final.pth eval_job.sh
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
# TRAINING LAUNCH
# =============================================================================

# Automatically detect number of GPUs assigned by Slurm
# Count the number of commas in CUDA_VISIBLE_DEVICES and add 1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
fi

echo "Detected $NUM_GPUS GPUs. Launching training..."

# Run the training script
# We use detectron2's launch utility to handle multi-GPU distributed training

# Set paths for the new cluster
export MASK2FORMER_ROOT=/work/sramam2s/Mask2Former
export SCANNETPP_ROOT=${MASK2FORMER_ROOT}/datasets/scannet/scannetpp
export PANOPTIC_ROOT=${SCANNETPP_ROOT}/panoptic
export PANOPTIC_VAL_ROOT=${SCANNETPP_ROOT}/panoptic_val
export SPLIT_DIR=${SCANNETPP_ROOT}/splits
export VAL_SPLIT=${SPLIT_DIR}/nvs_sem_val_clean.txt

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

echo ""
echo "Config: ${CONFIG_FILE}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Val split: ${VAL_SPLIT}"
echo "Panoptic val root: ${PANOPTIC_VAL_ROOT}"
echo ""

# Run evaluation
python m2f_evaluate.py \
    --config-file ${CONFIG_FILE} \
    --checkpoint ${CHECKPOINT} \
    --output-dir ${OUTPUT_DIR} \
    --scannetpp-root ${SCANNETPP_ROOT} \
    --panoptic-val-root ${PANOPTIC_VAL_ROOT} \
    --val-split ${VAL_SPLIT} \
    --target-short-edge 480 \
    --max-size 640 \
    --overlap-threshold 0.8 \
    --object-mask-threshold 0.8 \
    --save-predictions \
    MODEL.WEIGHTS ${CHECKPOINT}

echo ""
echo "========================================="
echo "Evaluation completed at $(date)"
echo "Results saved to: ${OUTPUT_DIR}"
echo "========================================="
