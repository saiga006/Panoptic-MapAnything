#!/bin/bash
#SBATCH --job-name=panoptic_train_sp  # Job name
#SBATCH --partition=gpu4            # Partition: 'gpu4' (A100) or 'gpu' (V100)
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --gres=gpu:4               # Number of GPUs per node (1-4 for gpu4, 1-4 for gpu)
#SBATCH --cpus-per-task=64         # CPU cores per task (adjust based on availability)
#SBATCH --mem=185G                 # Memory per node
#SBATCH --time=24:00:00            # Time limit hrs:min:sec
#SBATCH --output=slurm_%j.out      # Standard output log
#SBATCH --error=slurm_%j.err       # Standard error log

# =============================================================================
# ENVIRONMENT SETUP
# =============================================================================

# 1. Load necessary modules (adjust versions if needed)
#module purge
#module load cuda/11.8
source ~/.bashrc
# 2. Activate your Python environment (Conda / Virtualenv)
# Uncomment and adjust the line below to match your environment
eval "$(conda shell.bash hook)"
conda activate robot_lab
# OR
# source /home/saiga/rpl/mask2former/venv/bin/activate

echo "Python path: $(which python)"
echo "Conda env: $CONDA_DEFAULT_ENV"

# Load CUDA module
module load cuda/11.8

# Print environment info for debugging
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# =============================================================================
# TRAINING LAUNCH
# =============================================================================

# Automatically detect number of GPUs assigned by Slurm
# Count the number of commas in CUDA_VISIBLE_DEVICES and add 1
if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
    NUM_GPUS=1
else
    NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr -cd ',' | wc -c)
    NUM_GPUS=$((NUM_GPUS + 1))
fi

echo "Detected $NUM_GPUS GPUs. Launching training..."

# Run the training script
# We use detectron2's launch utility to handle multi-GPU distributed training
#cd /work/sramam2s/Mask2Former

# Set paths for the new cluster
export MASK2FORMER_ROOT=$(pwd)
# Ensure Mask2Former and current dir are in PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$MASK2FORMER_ROOT/Mask2Former:$MASK2FORMER_ROOT

export SCANNETPP_ROOT=${MASK2FORMER_ROOT}/Mask2Former/datasets/scannet/scannetpp
export PANOPTIC_ROOT=${SCANNETPP_ROOT}/panoptic
export PANOPTIC_VAL_ROOT=${SCANNETPP_ROOT}/panoptic_val
export SPLIT_DIR=${SCANNETPP_ROOT}/splits

python3 m2f_train_multiview.py \
    --num-gpus $NUM_GPUS \
    --config-file Mask2Former/configs/scannetpp/panoptic-segmentation/ma40.yaml \
    --num-classes 2878 \
    --scannetpp-root $SCANNETPP_ROOT \
    --panoptic-root $PANOPTIC_ROOT \
    --split-dir $SPLIT_DIR \
    SOLVER.IMS_PER_BATCH $NUM_GPUS \
    OUTPUT_DIR output_multiview_8a40

echo ""
echo "Training completed at $(date)"
echo ""