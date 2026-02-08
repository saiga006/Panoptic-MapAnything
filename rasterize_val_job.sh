#!/bin/bash
#SBATCH --job-name=raster_val
#SBATCH --output=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/logs/rasterize_val_%j.out
#SBATCH --error=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/logs/rasterize_val_%j.err
#SBATCH --partition=sgpu_medium
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100_80g:1
#SBATCH --mem=64G
#SBATCH --time=04:00:00

# ScanNet++ Validation Rasterization Job
# Uses the dedicated scanpp_raster environment with PyTorch 2.1.0
# Estimated time: ~2-3 hours for validation scenes

echo "=========================================="
echo "ScanNet++ VALIDATION Rasterization Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"
echo ""

# Set working directory
cd /lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former || exit 1

# Activate the dedicated rasterization environment
echo "Activating scanpp_raster conda environment..."
source /home/saiga/.bashrc
conda activate scanpp_raster

# Verify environment
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "PyTorch3D version: $(python -c 'import pytorch3d; print(pytorch3d.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo ""

# Run validation rasterization
echo "Starting validation rasterization..."
cd configs/scannetpp/scannetpp/semantic || exit 1

python -u rasterize.py \
    --config-name=rasterize_val \
    hydra.run.dir=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/outputs/rasterize_val_${SLURM_JOB_ID}

echo ""
echo "Validation rasterization completed!"
echo "End time: $(date)"
