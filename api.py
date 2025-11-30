# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
api.py - Clean Gradio API server for VGGT
Provides both UI and REST API for 3D reconstruction from images.
API endpoint: /vggt/create/images
"""

import os
import torch
import numpy as np
import gradio as gr
import sys
import shutil
from datetime import datetime
import glob
import gc
import time
from fastapi import Response

sys.path.append("vggt/")

from visual_util import predictions_to_glb
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

# -------------------------------------------------------------------------
# Model initialization
# -------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("WARNING: CUDA not available. Running on CPU will be very slow.")
else:
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

print("Initializing and loading VGGT model...")
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
model.eval()
model = model.to(device)
print("Model loaded successfully!")


# -------------------------------------------------------------------------
# run_inference - Core model inference function with timing
# -------------------------------------------------------------------------
def run_inference(image_paths: list) -> tuple[dict, dict]:
    """
    Run the VGGT model on a list of image paths and return predictions with timing stats.

    Args:
        image_paths: List of paths to input images

    Returns:
        tuple: (predictions dict, timing_stats dict)
    """
    timing_stats = {}

    # Preprocessing timing
    preprocess_start = time.time()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not torch.cuda.is_available():
        raise ValueError("CUDA is not available. Check your environment.")

    print(f"Processing {len(image_paths)} images")
    images = load_and_preprocess_images(image_paths).to(device)
    print(f"Preprocessed images shape: {images.shape}")

    timing_stats["preprocessing_seconds"] = time.time() - preprocess_start

    # Model inference timing
    inference_start = time.time()

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)

    timing_stats["model_inference_seconds"] = time.time() - inference_start

    # Post-processing timing
    postprocess_start = time.time()

    # Convert pose encoding to extrinsic and intrinsic matrices
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic

    # Convert tensors to numpy
    for key in predictions.keys():
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    predictions['pose_enc_list'] = None

    # Generate world points from depth map
    depth_map = predictions["depth"]
    world_points = unproject_depth_map_to_point_map(depth_map, predictions["extrinsic"], predictions["intrinsic"])
    predictions["world_points_from_depth"] = world_points

    timing_stats["postprocessing_seconds"] = time.time() - postprocess_start

    torch.cuda.empty_cache()
    return predictions, timing_stats


# -------------------------------------------------------------------------
# export_point_cloud_ply - Export predictions as PLY point cloud
# -------------------------------------------------------------------------
def export_point_cloud_ply(predictions: dict, output_path: str, conf_thres: float = 50.0) -> None:
    """
    Export predictions as a PLY point cloud file.

    Args:
        predictions: Model predictions dict
        output_path: Path to save the PLY file
        conf_thres: Confidence threshold percentage (0-100)
    """
    import trimesh

    # Get point cloud data
    pred_world_points = predictions["world_points_from_depth"]
    pred_world_points_conf = predictions.get("depth_conf", np.ones_like(pred_world_points[..., 0]))
    images = predictions["images"]

    # Flatten points
    vertices_3d = pred_world_points.reshape(-1, 3)

    # Handle image format
    if images.ndim == 4 and images.shape[1] == 3:  # NCHW format
        colors_rgb = np.transpose(images, (0, 2, 3, 1))
    else:
        colors_rgb = images
    colors_rgb = (colors_rgb.reshape(-1, 3) * 255).astype(np.uint8)

    # Apply confidence filtering
    conf = pred_world_points_conf.reshape(-1)
    if conf_thres > 0:
        conf_threshold = np.percentile(conf, conf_thres)
    else:
        conf_threshold = 0.0

    conf_mask = (conf >= conf_threshold) & (conf > 1e-5)
    vertices_3d = vertices_3d[conf_mask]
    colors_rgb = colors_rgb[conf_mask]

    # Create and export point cloud
    point_cloud = trimesh.PointCloud(vertices=vertices_3d, colors=colors_rgb)
    point_cloud.export(output_path)


# -------------------------------------------------------------------------
# process_images - Main processing function for API
# -------------------------------------------------------------------------
def process_images(
    images: list,
    output_format: str = "ply",
    conf_thres: float = 50.0
) -> tuple[str, dict]:
    """
    Process uploaded images and generate 3D reconstruction.

    Args:
        images: List of uploaded image file paths
        output_format: Output format - 'ply' (point cloud) or 'glb'
        conf_thres: Confidence threshold percentage (0-100)

    Returns:
        tuple: (output_file_path, analytics_dict)
    """
    total_start = time.time()
    analytics = {
        "num_images": 0,
        "output_format": output_format,
        "confidence_threshold": conf_thres,
        "timing": {}
    }

    # Validate inputs
    if images is None or len(images) < 2:
        raise gr.Error("Please upload at least 2 images")

    # Setup timing
    setup_start = time.time()

    gc.collect()
    torch.cuda.empty_cache()

    # Create temp directory for processing
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    target_dir = f"api_output_{timestamp}"
    target_dir_images = os.path.join(target_dir, "images")
    os.makedirs(target_dir_images, exist_ok=True)

    # Copy uploaded images to target directory
    image_paths = []
    for i, img_file in enumerate(images):
        if isinstance(img_file, dict) and "name" in img_file:
            file_path = img_file["name"]
        else:
            file_path = img_file

        dst_path = os.path.join(target_dir_images, f"{i:06}.png")
        shutil.copy(file_path, dst_path)
        image_paths.append(dst_path)

    image_paths = sorted(image_paths)
    analytics["num_images"] = len(image_paths)
    analytics["timing"]["setup_seconds"] = time.time() - setup_start

    # Run inference
    predictions, inference_timing = run_inference(image_paths)
    analytics["timing"].update(inference_timing)

    # Export to requested format
    export_start = time.time()

    if output_format.lower() == "glb":
        # Export as GLB
        output_file = os.path.join(target_dir, "output.glb")
        glb_scene = predictions_to_glb(
            predictions,
            conf_thres=conf_thres,
            filter_by_frames="All",
            mask_black_bg=False,
            mask_white_bg=False,
            show_cam=True,
            mask_sky=False,
            target_dir=target_dir,
            prediction_mode="Depthmap and Camera Branch",
        )
        glb_scene.export(file_obj=output_file)
    else:
        # Export as PLY (default)
        output_file = os.path.join(target_dir, "output.ply")
        export_point_cloud_ply(predictions, output_file, conf_thres)

    analytics["timing"]["export_seconds"] = time.time() - export_start
    analytics["timing"]["total_seconds"] = time.time() - total_start

    # Cleanup
    del predictions
    gc.collect()
    torch.cuda.empty_cache()

    return output_file, analytics


# -------------------------------------------------------------------------
# format_analytics - Format analytics dict as markdown string
# -------------------------------------------------------------------------
def format_analytics(analytics: dict) -> str:
    """
    Format analytics dictionary as a readable markdown string.
    """
    return f"""## Processing Analytics

**Input:**
- Images processed: {analytics['num_images']}
- Output format: {analytics['output_format'].upper()}
- Confidence threshold: {analytics['confidence_threshold']}%

**Timing:**
- Setup: {analytics['timing']['setup_seconds']:.3f}s
- Preprocessing: {analytics['timing']['preprocessing_seconds']:.3f}s
- Model inference: {analytics['timing']['model_inference_seconds']:.3f}s
- Post-processing: {analytics['timing']['postprocessing_seconds']:.3f}s
- Export: {analytics['timing']['export_seconds']:.3f}s
- **Total: {analytics['timing']['total_seconds']:.3f}s**
"""


# -------------------------------------------------------------------------
# gradio_process_images - Gradio UI/API handler
# -------------------------------------------------------------------------
def gradio_process_images(images, output_format, conf_thres):
    """
    Gradio handler for image processing. Works for both UI and API.
    Returns output file and formatted analytics string.
    """
    try:
        output_file, analytics = process_images(images, output_format, conf_thres)
        analytics_str = format_analytics(analytics)
        return output_file, analytics_str
    except Exception as e:
        return None, f"Error: {str(e)}"


# -------------------------------------------------------------------------
# ping - Health check endpoint for RunPod
# -------------------------------------------------------------------------
def ping():
    """
    Health check endpoint for RunPod.
    Returns server status and model readiness.
    """
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "device": device,
    }


# -------------------------------------------------------------------------
# Build Gradio UI with API
# -------------------------------------------------------------------------
theme = gr.themes.Ocean()

with gr.Blocks(
    theme=theme,
    title="VGGT API",
    css="""
    .analytics-box {
        font-family: monospace;
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 8px;
    }
    """,
) as demo:
    gr.HTML(
        """
        <h1>VGGT API - 3D Reconstruction</h1>
        <p>
        <a href="https://github.com/facebookresearch/vggt">GitHub Repository</a> |
        <strong>API Endpoint: <code>/api/vggt/create/images</code></strong>
        </p>

        <div style="font-size: 14px; line-height: 1.5; margin-bottom: 20px;">
        <p>Upload 2 or more images to generate a 3D point cloud or GLB model.</p>
        </div>
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            # Input section
            gr.Markdown("### Input")
            input_images = gr.File(
                file_count="multiple",
                label="Upload Images (minimum 2)",
                file_types=["image"],
            )

            with gr.Row():
                output_format = gr.Radio(
                    choices=["ply", "glb"],
                    value="ply",
                    label="Output Format",
                    info="PLY = point cloud, GLB = 3D scene"
                )
                conf_thres = gr.Slider(
                    minimum=0,
                    maximum=100,
                    value=50,
                    step=1,
                    label="Confidence Threshold (%)",
                    info="Filter out low-confidence points"
                )

            process_btn = gr.Button("Generate 3D", variant="primary", size="lg")

        with gr.Column(scale=1):
            # Output section
            gr.Markdown("### Output")
            output_file = gr.File(label="Download Result")
            analytics_output = gr.Markdown(
                value="Upload images and click 'Generate 3D' to begin.",
                elem_classes=["analytics-box"]
            )

    # Wire up the UI with API endpoint
    process_btn.click(
        fn=gradio_process_images,
        inputs=[input_images, output_format, conf_thres],
        outputs=[output_file, analytics_output],
        api_name="vggt/create/images",
    )

    # Ping endpoint for RunPod health checks (hidden from UI)
    ping_btn = gr.Button("Ping", visible=False)
    ping_output = gr.JSON(visible=False)
    ping_btn.click(
        fn=ping,
        inputs=[],
        outputs=[ping_output],
        api_name="ping",
    )

# -------------------------------------------------------------------------
# Launch configuration
# -------------------------------------------------------------------------
server_name = os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0")
server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))

# -------------------------------------------------------------------------
# Add /ping endpoint for RunPod health checks (direct FastAPI route)
# -------------------------------------------------------------------------
@demo.app.get("/ping")
def ping_endpoint():
    """
    Health check endpoint for RunPod load balancing.
    Returns 200 when model is loaded and ready, 503 otherwise.
    """
    if model is not None:
        return Response(content="OK", status_code=200)
    else:
        return Response(content="Model not ready", status_code=503)


if __name__ == "__main__":
    print(f"Starting VGGT API server on {server_name}:{server_port}")
    print(f"Web UI: http://{server_name}:{server_port}")
    print(f"API endpoint: http://{server_name}:{server_port}/api/vggt/create/images")
    print(f"Health check: http://{server_name}:{server_port}/ping")
    print(f"API docs: http://{server_name}:{server_port}/docs")

    demo.queue(max_size=20).launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        share=False,
    )
