# Dockerfile for VGGT - Visual Geometry Grounded Transformer
# NVIDIA CUDA compatible for RunPod deployment with Gradio API

FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.10 /usr/bin/python \
    && ln -sf /usr/bin/python3.10 /usr/bin/python3

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /app

# Install PyTorch with CUDA 12.1 support
RUN pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

# Copy requirements files first for better caching
COPY requirements.txt requirements_demo.txt ./

# Install core dependencies
RUN pip install -r requirements.txt

# Install demo dependencies (excluding pycolmap/pyceres which can be problematic)
RUN pip install \
    gradio==5.17.1 \
    viser==0.2.23 \
    tqdm \
    hydra-core \
    omegaconf \
    opencv-python \
    scipy \
    onnxruntime \
    requests \
    trimesh \
    matplotlib \
    pydantic==2.10.6

# Install LightGlue from git
RUN pip install git+https://github.com/jytime/LightGlue.git#egg=lightglue

# Copy the entire project
COPY . .

# Create directories for temporary files and model cache
RUN mkdir -p /app/input_images /app/cache /root/.cache/torch/hub

# Set HuggingFace cache directory
ENV HF_HOME=/app/cache
ENV TORCH_HOME=/app/cache

# Pre-download the model weights during build (optional - can be removed to reduce image size)
# RUN python -c "import torch; torch.hub.load_state_dict_from_url('https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt', map_location='cpu')"

# Expose Gradio default port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:7860/ || exit 1

# Default command - run the Gradio demo with API enabled
CMD ["python", "demo_gradio_api.py"]
