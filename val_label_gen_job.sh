#!/bin/bash
#SBATCH --job-name=scanpp_val_labels
#SBATCH --partition=sgpu_medium
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=7:59:00                # Just under 8hr limit - resumable
#SBATCH --output=logs/val_labels_%j.out
#SBATCH --error=logs/val_labels_%j.err

# =============================================================================
# SCANNET++ VALIDATION LABEL GENERATION JOB - RESUMABLE
# Generates 2D panoptic labels from validation rasterization cache
# Run this AFTER validation rasterization completes
# 
# RESUME: Just resubmit if interrupted - skips existing files automatically
# =============================================================================

mkdir -p logs
source ~/.bashrc
conda activate scanpp_raster

echo "========================================="
echo "Validation Label Generation Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "========================================="

module load CUDA/11.8.0
nvidia-smi

export MASK2FORMER_ROOT=/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former
cd ${MASK2FORMER_ROOT}/configs/scannetpp/scannetpp

echo ""
echo "Starting validation label generation at $(date)"

# Generate semantic palette if it doesn't exist
if [ ! -f "${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/metadata/scannetpp_semantic_palette.txt" ]; then
    echo "Generating semantic color palette..."
    cd ${MASK2FORMER_ROOT}
    python scripts/generate_semantic_palette.py
    cd ${MASK2FORMER_ROOT}/configs/scannetpp/scannetpp
fi

python -m semantic.prep.semantics_2d --config-name=semantics_2d_val
echo "Label generation completed at $(date)"

# =============================================================================
# CONVERT TO DETECTRON2 PANOPTIC FORMAT
# =============================================================================
echo "Converting to Detectron2 panoptic format..."

cd ${MASK2FORMER_ROOT}
python scripts/convert_to_panoptic_format.py \
    --input_dir ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/panoptic_annotations_val \
    --output_dir ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/panoptic_val \
    --scene_list ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/splits/nvs_sem_val_clean.txt \
    --semantic_classes ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/metadata/semantic_classes.txt \
    --instance_classes ${MASK2FORMER_ROOT}/datasets/scannet/scannetpp/metadata/instance_classes.txt

echo ""
echo "Validation conversion completed at $(date)"
echo ""
