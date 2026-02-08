#!/bin/bash
#SBATCH --job-name=m2f_scannet_16gpu
#SBATCH --partition=mlgpu_medium
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=400G
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_16a40_%j.out
#SBATCH --error=logs/train_16a40_%j.err

# =============================================================================
# MULTI-VIEW MASK2FORMER TRAINING ON SCANNET++ (2 NODES x 8 GPUs)
# ==================m2f_train_multiview===========================================================

mkdir -p logs
source ~/.bashrc
conda activate robot_lab

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: 8"
echo "Total GPUs: 16"
echo "========================================="

# Load CUDA
module load CUDA/11.8.0

# Master Address setup for Multi-Node
# Get the first node name from the node list
HEAD_NODE=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
# Create the distributed URL
EXT_PORT=29500
export DIST_URL="tcp://$HEAD_NODE:$EXT_PORT"

echo "Head Node: $HEAD_NODE"
echo "Distributed URL: $DIST_URL"

# OMP threads (Adjust as needed, typically total_cores / gpus_per_node if running directly, 
# but here python spawns processes. Safe to set lower to avoid overload)
export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # Commented out for multi-node. Enable InfiniBand for better performance.

# Set paths
export MASK2FORMER_ROOT=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former
export SCANNETPP_ROOT=${MASK2FORMER_ROOT}/datasets/scannet/scannetpp
export PANOPTIC_ROOT=${SCANNETPP_ROOT}/panoptic
export PANOPTIC_VAL_ROOT=${SCANNETPP_ROOT}/panoptic_val
export SPLIT_DIR=${SCANNETPP_ROOT}/splits

cd ${MASK2FORMER_ROOT}

echo ""
echo "Starting multi-node training at $(date)"
echo "Config: configs/scannetpp/panoptic-segmentation/ma40_16gpu.yaml"
echo ""

# Run training on each node using srun
# We use 'bash -c' to ensure python arguments are parsed correctly and variables expanded appropriately
# --machine-rank is set to $SLURM_NODEID (0 for first node, 1 for second, etc.)
# dist-url is passed from the head node calculation above

srun bash -c "python m2f_train_multiview.py \
    --num-gpus 8 \
    --num-machines 2 \
    --machine-rank \$SLURM_NODEID \
    --dist-url $DIST_URL \
    --config-file configs/scannetpp/panoptic-segmentation/ma40_16gpu.yaml \
    OUTPUT_DIR output_multiview_16a40"

echo ""
echo "Training completed at $(date)"
echo ""
