#!/bin/bash
#SBATCH --job-name=render_depth
#SBATCH --output=logs/render_depth_%j.out
#SBATCH --error=logs/render_depth_%j.err
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate renderpy

# Navigate to scannetpp toolkit directory
cd /lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/configs/scannetpp/scannetpp

# Run depth rendering
echo "Starting depth rendering at $(date)"
python -m common.render common/configs/render.yml
echo "Depth rendering completed at $(date)"
