#!/bin/bash
#SBATCH --job-name=scanpp_render      # Job name
#SBATCH --partition=sgpu_medium       # Partition: 'gpu4' (A100) or 'gpu' (V100)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --gres=gpu:1                  # Only need 1 GPU
#SBATCH --cpus-per-task=16            # CPU cores
#SBATCH --mem=64G                     # Memory
#SBATCH --time=17:59:00                # Just under 8hr limit - will auto-resubmit if needed
#SBATCH --output=logs/labels_%j.out   # Standard output log
#SBATCH --error=logs/labels_%j.err    # Standard error log

# =============================================================================
# SCANNET++ LABEL GENERATION JOB - RESUMABLE
# Generates 2D panoptic labels from rasterization cache
# 
# RESUME CAPABILITY:
#   - skip_existing_semantic_gt_2d: true (already in config)
#   - Job will skip completed images automatically
#   - If interrupted, just resubmit - it will continue from where it left off
#
# Estimated time: 8+ hours for 150 scenes (may need multiple runs)
# =============================================================================

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate conda environment
source ~/.bashrc
conda activate renderpy


echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Python: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"
echo "========================================="

# Load CUDA
module load CUDA/11.8.0

# Print GPU info
nvidia-smi



# Navigate to scannetpp toolkit directory
cd /lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/configs/scannetpp/scannetpp

# Run depth rendering
echo "Starting depth rendering at $(date)"
python -m common.render common/configs/render.yml
echo "Depth rendering completed at $(date)"
