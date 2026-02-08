#!/bin/bash
#SBATCH --job-name=rsync_scanpp
#SBATCH --partition=intelsr_medium
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=logs/rsync_%j.out
#SBATCH --error=logs/rsync_%j.err

mkdir -p logs

echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Started at: $(date)"
echo "========================================="

# Option: Use sshpass if SSH keys are not set up
# SECURITY WARNING: Storing passwords in scripts is not recommended
# Better approach: Set up SSH key-based authentication (see instructions)
# 
# If you must use password authentication:
# 1. Uncomment the lines below
# 2. Set SSH_PASSWORD environment variable before submitting job:
#    export SSH_PASSWORD="your_password"
#    sbatch rsync_job.sh
#
# if [ -n "$SSH_PASSWORD" ]; then
#     export RSYNC_RSH="sshpass -e ssh"
#     echo "Using sshpass for authentication"
# fi

# Source directory (adjust if needed)
SOURCE_DIR="/lustre/mlnvme/data/sairamm_hpc-plr2025/Mask2Former/datasets/scannet/scannetpp"
DEST_HOST="sramam2s@wr0.wr.inf.h-brs.de"
DEST_DIR="/work/sramam2s/Mask2Former/datasets/scannet/scannetpp/."

# Change to source directory
cd ${SOURCE_DIR}

echo "Source directory: $(pwd)"
echo "Destination: ${DEST_HOST}:${DEST_DIR}"
echo "========================================="
echo ""

# List of directories/files to sync
ITEMS_TO_SYNC=(
    "panoptic"
    "panoptic_annotations"
    "panoptic_annotations_val"
    "panoptic_val"
    "rasterization"
    "splits"
    "metadata"
    "data/valid_scenes"
    "data/valid_val_scenes"
)

echo "Items to sync:"
for item in "${ITEMS_TO_SYNC[@]}"; do
    echo "  - $item"
done
echo ""
echo "========================================="
echo "Starting rsync at $(date)"
echo "========================================="
echo ""

# Run rsync with progress and partial transfer support
rsync -avP --partial \
    "${ITEMS_TO_SYNC[@]}" \
    ${DEST_HOST}:${DEST_DIR}

RSYNC_EXIT_CODE=$?

echo ""
echo "========================================="
echo "Rsync completed with exit code: ${RSYNC_EXIT_CODE}"
echo "Finished at: $(date)"
echo "========================================="

if [ ${RSYNC_EXIT_CODE} -eq 0 ]; then
    echo "✓ Transfer completed successfully!"
else
    echo "✗ Transfer failed with errors. Check the log for details."
fi

exit ${RSYNC_EXIT_CODE}
