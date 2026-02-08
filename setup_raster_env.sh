#!/bin/bash
# Setup separate conda environment for ScanNet++ rasterization
# This avoids breaking your main training environment

set -e  # Exit on error

echo "========================================="
echo "Creating ScanNet++ Rasterization Environment"
echo "========================================="
echo ""

ENV_NAME="scanpp_raster"

# Check if environment already exists
if conda env list | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Remove and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Exiting without changes."
        exit 0
    fi
fi

echo "Creating conda environment: $ENV_NAME"
conda create -n $ENV_NAME python=3.10 -y

echo ""
echo "Activating environment..."
source ~/.bashrc
conda activate $ENV_NAME

echo ""
echo "Installing PyTorch 2.1.0 with CUDA 11.8..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Installing PyTorch3D (pre-built for PyTorch 2.1.0)..."
pip install pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

echo ""
echo "Installing dependencies..."
pip install "numpy<2.0"  # NumPy 1.x required for PyTorch 2.1.0
pip install hydra-core omegaconf
pip install tqdm opencv-python pillow
pip install open3d
pip install wandb
pip install codetiming

echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import pytorch3d; print(f'PyTorch3D version: {pytorch3d.__version__}')"

echo ""
echo "========================================="
echo "✓ Environment '$ENV_NAME' created successfully!"
echo "========================================="
echo ""
echo "To use this environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To test GPU availability:"
echo "  python -c \"import torch; print(torch.cuda.is_available())\""
echo ""
