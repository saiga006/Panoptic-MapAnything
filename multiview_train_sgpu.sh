#!/bin/bash
#SBATCH --job-name=m2f_sgpu
#SBATCH --partition=sgpu_short
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=2:00:00
#SBATCH --output=logs/train_8a40_%j.out
#SBATCH --error=logs/train_8a40_%j.err

# =============================================================================
# MULTI-VIEW MASK2FORMER TRAINING ON SCANNET++
# =============================================================================

mkdir -p logs
source ~/.bashrc
conda activate robot_lab  # Use training environment (not scanpp_raster)

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $SLURM_GPUS_ON_NODE"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "========================================="

# Load CUDA
module load CUDA/11.8.0

# Print GPU info
nvidia-smi

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=1  # Disable InfiniBand (not needed for single node)


# Set paths
export MASK2FORMER_ROOT=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former
export SCANNETPP_ROOT=${MASK2FORMER_ROOT}/datasets/scannet/scannetpp
export PANOPTIC_ROOT=${SCANNETPP_ROOT}/panoptic  # Converted panoptic annotations
export PANOPTIC_VAL_ROOT=${SCANNETPP_ROOT}/panoptic_val
export SPLIT_DIR=${SCANNETPP_ROOT}/splits

cd ${MASK2FORMER_ROOT}

echo ""
echo "Starting multi-view training at $(date)"
echo "Training data: ${PANOPTIC_ROOT}"
echo "Validation data: ${PANOPTIC_VAL_ROOT}"
echo ""


# Step 2: Run multi-view training
# The config expects these dataset names:
#   DATASETS.TRAIN: ("scannetpp_panoptic_train",)
#   DATASETS.TEST: ("scannetpp_panoptic_val",)
# which were just registered above

python m2f_train_multiview.py \
    --num-gpus 4 \
    --config-file configs/scannetpp/panoptic-segmentation/ma40.yaml \
    OUTPUT_DIR output_multiview_8gpu40

echo ""
echo "Training completed at $(date)"
echo ""
