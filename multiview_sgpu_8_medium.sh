#!/bin/bash
#SBATCH --job-name=m2f_scannet_8gpu_long
#SBATCH --partition=sgpu_long
#SBATCH --nodes=2
#SBATCH --gpus-per-node=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=16:00:00
#SBATCH --qos=normal  # Priority 10 (higher than medium=6 or long=4). If validation fails, remove this line.
#SBATCH --output=logs/train_8gpu_med_%j.out
#SBATCH --error=logs/train_8gpu_med_%j.err

# =============================================================================
# MULTI-VIEW MASK2FORMER TRAINING ON SCANNET++ (2 NODES x 4 GPUs = 8 GPUs)
# =============================================================================

mkdir -p logs
source ~/.bashrc
conda activate robot_lab

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "GPUs per node: 4"
echo "Total GPUs: 8"
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

# OMP threads
export OMP_NUM_THREADS=16
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1  # Enable/Disable InfiniBand as needed

# Set paths
export MASK2FORMER_ROOT=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former
export SCANNETPP_ROOT=${MASK2FORMER_ROOT}/datasets/scannet/scannetpp
export PANOPTIC_ROOT=${SCANNETPP_ROOT}/panoptic
export PANOPTIC_VAL_ROOT=${SCANNETPP_ROOT}/panoptic_val
export SPLIT_DIR=${SCANNETPP_ROOT}/splits

cd ${MASK2FORMER_ROOT}

echo ""
echo "Starting multi-node training at $(date)"
# We use ma40.yaml because it is configured for 8 images global batch size (IMS_PER_BATCH: 8)
# With 8 GPUs total (2 nodes * 4 GPUs), we get 1 image per GPU.
echo "Config: configs/scannetpp/panoptic-segmentation/ma40.yaml"
echo ""

# Run training on each node using srun
# --num-gpus 4 (per machine)
# --num-machines 2
srun bash -c "python m2f_train_multiview.py \
    --num-gpus 4 \
    --num-machines 2 \
    --machine-rank \$SLURM_NODEID \
    --dist-url $DIST_URL \
    --config-file configs/scannetpp/panoptic-segmentation/ma40.yaml \
    OUTPUT_DIR output_multiview_8gpu_medium"

echo ""
echo "Training completed at $(date)"
echo ""
