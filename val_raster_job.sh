#!/bin/bash
#SBATCH --job-name=scanpp_val_raster
#SBATCH --partition=sgpu_medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=logs/val_raster_%j.out
#SBATCH --error=logs/val_raster_%j.err

mkdir -p logs
source ~/.bashrc
conda activate scanpp_raster  # Use separate environment for rasterization

echo "========================================="
echo "Validation Rasterization Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "========================================="

# Load CUDA FIRST
module load CUDA/11.8.0

# FORCE GPU visibility
export CUDA_VISIBLE_DEVICES=0
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Print GPU info
nvidia-smi

# Verify PyTorch can see GPU
python -c "import torch; print(f'PyTorch CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU ONLY!\"}')"

export MASK2FORMER_ROOT=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former
cd ${MASK2FORMER_ROOT}/configs/scannetpp/scannetpp

echo "Starting validation rasterization at $(date)"
python -m semantic.prep.rasterize --config-name=rasterize_val
echo "Completed at $(date)"
