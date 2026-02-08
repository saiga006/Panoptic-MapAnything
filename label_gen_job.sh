#!/bin/bash
#SBATCH --job-name=scanpp_labels      # Job name
#SBATCH --partition=sgpu_medium       # Partition: 'gpu4' (A100) or 'gpu' (V100)
#SBATCH --nodes=1                     # Number of nodes
#SBATCH --gres=gpu:1                  # Only need 1 GPU
#SBATCH --cpus-per-task=16            # CPU cores
#SBATCH --mem=64G                     # Memory
#SBATCH --time=7:59:00                # Just under 8hr limit - will auto-resubmit if needed
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

# Load environment
source ~/.bashrc
conda activate scanpp_raster  # Use separate environment for label generation

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

# Set environment variable for Mask2Former root
export MASK2FORMER_ROOT=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former

# Change to scannetpp directory
cd ${MASK2FORMER_ROOT}/configs/scannetpp/scannetpp

echo ""
echo "Starting label generation at $(date)"
echo "Config: semantics_2d_quick (250 scenes)"
echo ""

# Generate semantic palette if it doesn't exist
if [ ! -f "${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/metadata/scannetpp_semantic_palette.txt" ]; then
    echo "Generating semantic color palette..."
    cd ${MASK2FORMER_ROOT}
    python scripts/generate_semantic_palette.py
    cd ${MASK2FORMER_ROOT}/configs/scannetpp/scannetpp
fi

# Run label generation
python -m semantic.prep.semantics_2d --config-name=semantics_2d_quick

echo ""
echo "Label generation completed at $(date)"
echo ""

# =============================================================================
# CONVERT TO DETECTRON2 PANOPTIC FORMAT
# =============================================================================
echo "Converting to Detectron2 panoptic format..."

cd ${MASK2FORMER_ROOT}
python scripts/convert_to_panoptic_format.py \
    --input_dir ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/panoptic_annotations \
    --output_dir ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/panoptic \
    --scene_list ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/splits/nvs_sem_train_subset150.txt \
    --semantic_classes ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/metadata/semantic_classes.txt \
    --instance_classes ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/metadata/instance_classes.txt

echo ""
echo "Conversion to Detectron2 format completed at $(date)"
echo ""
