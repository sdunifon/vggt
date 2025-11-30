#!/bin/bash
# start.sh - RunPod startup script for VGGT Gradio API

set -e

echo "========================================="
echo "VGGT - Visual Geometry Grounded Transformer"
echo "Starting Gradio API Server..."
echo "========================================="

# Print GPU info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || echo "nvidia-smi not available"

# Set environment variables for RunPod
export GRADIO_SERVER_NAME="0.0.0.0"
export GRADIO_SERVER_PORT="${GRADIO_PORT:-7860}"

# Create necessary directories
mkdir -p /app/input_images /app/cache

# Set cache directories
export HF_HOME=/app/cache
export TORCH_HOME=/app/cache

# Check if model should be pre-downloaded
if [ "${PRELOAD_MODEL:-true}" = "true" ]; then
    echo "Pre-downloading VGGT model weights..."
    python -c "
import torch
print('Downloading VGGT-1B model weights...')
torch.hub.load_state_dict_from_url(
    'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt',
    map_location='cpu'
)
print('Model weights downloaded successfully!')
" || echo "Model download will happen on first inference"
fi

echo "Starting Gradio server on port ${GRADIO_SERVER_PORT}..."

# Run the Gradio API server
exec python demo_gradio_api.py
