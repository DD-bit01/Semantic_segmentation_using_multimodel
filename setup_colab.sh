#!/bin/bash
# ============================================================
# ASG-Net: Google Colab Setup Script
# Run this at the start of your Colab session
# ============================================================

echo "🚀 Setting up ASG-Net environment..."

# Install dependencies
pip install -q torch torchvision timm einops transformers
pip install -q tifffile rasterio opencv-python scikit-image
pip install -q pyyaml tensorboard matplotlib seaborn scikit-learn tqdm

# Check GPU
python -c "
import torch
if torch.cuda.is_available():
    gpu = torch.cuda.get_device_name(0)
    mem = torch.cuda.get_device_properties(0).total_mem / 1e9
    print(f'✅ GPU: {gpu} ({mem:.1f} GB)')
else:
    print('⚠️  No GPU detected! Go to Runtime > Change runtime type > GPU')
"

echo "✅ Setup complete!"
