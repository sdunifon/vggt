#!/usr/bin/env python3
"""
local_ui.py - Local Flask UI that connects to remote VGGT RunPod API
Simple web interface for the remote 3D reconstruction service.
"""

from flask import Flask, render_template_string, request, send_file, jsonify
from gradio_client import Client, handle_file
import tempfile
import shutil
import os
import httpx
import requests

app = Flask(__name__)

# Remote API configuration
ENDPOINT = "https://gqennwuwqjqkji.api.runpod.ai"
API_KEY = "rpa_O9K64UIIBYISKT9A6R5K8QACBW7YNENAA0HGMLKR1wwhb1"

# Global client
_client = None


def get_client():
    """Get or create the Gradio client connection."""
    global _client
    if _client is None:
        print("Connecting to remote API...")
        _client = Client(
            ENDPOINT,
            headers={"Authorization": API_KEY},
            httpx_kwargs={"timeout": httpx.Timeout(300.0, connect=60.0)}
        )
        print("Connected!")
    return _client


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>VGGT - 3D Reconstruction</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        h1 { color: #333; }
        .card {
            background: white;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .status {
            padding: 8px 16px;
            border-radius: 4px;
            display: inline-block;
            margin-bottom: 20px;
        }
        .status.online { background: #d4edda; color: #155724; }
        .status.offline { background: #f8d7da; color: #721c24; }
        label { display: block; margin: 10px 0 5px; font-weight: 500; }
        input[type="file"] {
            width: 100%;
            padding: 10px;
            border: 2px dashed #ccc;
            border-radius: 4px;
            cursor: pointer;
        }
        select, input[type="range"] { width: 100%; padding: 8px; }
        button {
            background: #007bff;
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 16px;
            width: 100%;
            margin-top: 20px;
        }
        button:hover { background: #0056b3; }
        button:disabled { background: #ccc; cursor: not-allowed; }
        .result {
            margin-top: 20px;
            padding: 15px;
            background: #e8f5e9;
            border-radius: 4px;
        }
        .error {
            background: #ffebee;
            color: #c62828;
        }
        #progress {
            display: none;
            margin-top: 20px;
            text-align: center;
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #007bff;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 0 auto 10px;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        pre { white-space: pre-wrap; word-wrap: break-word; }
        .row { display: flex; gap: 20px; }
        .col { flex: 1; }
    </style>
</head>
<body>
    <h1>🎯 VGGT - 3D Reconstruction</h1>
    <p>Local UI → Remote GPU Processing</p>

    <div class="status {{ 'online' if api_status else 'offline' }}">
        {{ '🟢 API Online' if api_status else '🔴 API Offline' }}
    </div>

    <form id="uploadForm" enctype="multipart/form-data">
        <div class="card">
            <h3>📸 Input Images</h3>
            <label>Upload Images (minimum 2)</label>
            <input type="file" name="images" id="images" multiple accept="image/*" required>
            <small id="fileCount">No files selected</small>
        </div>

        <div class="card">
            <h3>⚙️ Settings</h3>
            <div class="row">
                <div class="col">
                    <label>Output Format</label>
                    <select name="output_format" id="output_format">
                        <option value="glb">GLB (3D Scene)</option>
                        <option value="ply">PLY (Point Cloud)</option>
                    </select>
                </div>
                <div class="col">
                    <label>Confidence Threshold: <span id="confValue">50</span>%</label>
                    <input type="range" name="conf_thres" id="conf_thres" min="0" max="100" value="50">
                </div>
            </div>
        </div>

        <button type="submit" id="submitBtn">🚀 Generate 3D</button>
    </form>

    <div id="progress">
        <div class="spinner"></div>
        <p id="progressText">Processing on remote GPU...</p>
    </div>

    <div id="result" class="card" style="display:none;"></div>

    <script>
        // File count display
        document.getElementById('images').addEventListener('change', function(e) {
            document.getElementById('fileCount').textContent =
                this.files.length + ' file(s) selected';
        });

        // Slider value display
        document.getElementById('conf_thres').addEventListener('input', function(e) {
            document.getElementById('confValue').textContent = this.value;
        });

        // Form submission
        document.getElementById('uploadForm').addEventListener('submit', async function(e) {
            e.preventDefault();

            const formData = new FormData(this);
            const btn = document.getElementById('submitBtn');
            const progress = document.getElementById('progress');
            const result = document.getElementById('result');

            btn.disabled = true;
            progress.style.display = 'block';
            result.style.display = 'none';

            try {
                const response = await fetch('/process', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (data.success) {
                    result.className = 'card result';
                    result.innerHTML = `
                        <h3>✅ Success!</h3>
                        <p><a href="${data.download_url}" download>📥 Download ${data.filename}</a></p>
                        <pre>${data.analytics}</pre>
                    `;
                } else {
                    result.className = 'card result error';
                    result.innerHTML = `<h3>❌ Error</h3><p>${data.error}</p>`;
                }
            } catch (err) {
                result.className = 'card result error';
                result.innerHTML = `<h3>❌ Error</h3><p>${err.message}</p>`;
            }

            btn.disabled = false;
            progress.style.display = 'none';
            result.style.display = 'block';
        });
    </script>
</body>
</html>
"""


@app.route('/')
def index():
    """Render the main page."""
    try:
        response = requests.get(
            f"{ENDPOINT}/ping",
            headers={"Authorization": API_KEY},
            timeout=5
        )
        api_status = response.status_code == 200
    except:
        api_status = False

    return render_template_string(HTML_TEMPLATE, api_status=api_status)


@app.route('/process', methods=['POST'])
def process():
    """Process uploaded images."""
    try:
        files = request.files.getlist('images')
        output_format = request.form.get('output_format', 'glb')
        conf_thres = int(request.form.get('conf_thres', 50))

        if len(files) < 2:
            return jsonify({'success': False, 'error': 'Please upload at least 2 images'})

        # Save uploaded files temporarily
        temp_dir = tempfile.mkdtemp()
        image_paths = []

        for i, f in enumerate(files):
            path = os.path.join(temp_dir, f"{i:04d}_{f.filename}")
            f.save(path)
            image_paths.append(path)

        # Call the remote API
        client = get_client()
        file_handles = [handle_file(p) for p in image_paths]

        result = client.predict(
            images=file_handles,
            output_format=output_format,
            conf_thres=conf_thres,
            api_name="/vggt/create/images"
        )

        # Clean up temp files
        shutil.rmtree(temp_dir)

        remote_output = result[0]
        analytics = result[1] if len(result) > 1 else ""

        if remote_output is None:
            return jsonify({'success': False, 'error': analytics})

        # Copy to output directory
        ext = '.glb' if output_format == 'glb' else '.ply'
        output_filename = f"vggt_output{ext}"
        output_path = os.path.join(os.getcwd(), output_filename)
        shutil.copy(remote_output, output_path)

        return jsonify({
            'success': True,
            'download_url': f'/download/{output_filename}',
            'filename': output_filename,
            'analytics': analytics
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/download/<filename>')
def download(filename):
    """Download a result file."""
    path = os.path.join(os.getcwd(), filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404


if __name__ == '__main__':
    print("=" * 50)
    print("VGGT Local UI")
    print("=" * 50)
    print(f"Remote API: {ENDPOINT}")
    print()
    print("Starting server at http://127.0.0.1:7861")
    print("=" * 50)

    app.run(host='127.0.0.1', port=7861, debug=False)
