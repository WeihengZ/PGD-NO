#!/bin/bash

# Setup script for PGD-NO project
# This script creates a conda environment and installs all required packages

set -e  # Exit on error

ENV_NAME="pgd-no"
PYTHON_VERSION="3.10"

echo "=========================================="
echo "Setting up conda environment: $ENV_NAME"
echo "=========================================="

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed or not in PATH"
    echo "Please install Miniconda or Anaconda first"
    exit 1
fi

# Remove existing environment if it exists
if conda env list | grep -q "^${ENV_NAME} "; then
    echo "Environment $ENV_NAME already exists. Removing it..."
    conda env remove -n $ENV_NAME -y
fi

# Create new conda environment
echo "Creating conda environment with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Activate the environment
echo "Activating environment..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Update conda and pip
echo "Updating conda and pip..."
conda update -n base conda -y
pip install --upgrade pip

# Install PyTorch (with CUDA support if available, otherwise CPU)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    echo "CUDA detected. Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo "No CUDA detected. Installing PyTorch CPU version..."
    pip install torch torchvision torchaudio
fi

# Install core scientific computing packages
echo "Installing core scientific computing packages..."
conda install -y numpy scipy pandas -c conda-forge

# Install VTK (Visualization Toolkit)
echo "Installing VTK..."
conda install -y vtk -c conda-forge

# Install networkx for graph operations
echo "Installing networkx..."
pip install networkx

# Install specialized packages
echo "Installing specialized packages..."
pip install timm einops tqdm pyvista matplotlib

# Verify installations
echo ""
echo "=========================================="
echo "Verifying installations..."
echo "=========================================="

python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import numpy; print(f'NumPy version: {numpy.__version__}')"
python -c "import scipy; print(f'SciPy version: {scipy.__version__}')"
python -c "import vtk; print(f'VTK version: {vtk.VTK_VERSION}')"
python -c "import networkx; print(f'NetworkX version: {networkx.__version__}')"
python -c "import timm; print(f'timm version: {timm.__version__}')"
python -c "import einops; print(f'einops version: {einops.__version__}')"
python -c "import pyvista; print(f'PyVista version: {pyvista.__version__}')"
python -c "import matplotlib; print(f'Matplotlib version: {matplotlib.__version__}')"

echo ""
echo "=========================================="
echo "✅ Environment setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To deactivate, run:"
echo "  conda deactivate"
echo ""
