"""WebUI for ml-sharp 3D Gaussian Splat prediction.

A simple Flask-based web interface for uploading images and generating 3DGS PLY files.
"""

from __future__ import annotations

import logging
import os
import sys
import subprocess
import tempfile
import uuid
import shutil
import threading
import time
import gc
from pathlib import Path
import urllib.parse
import traceback

# --- Environment Auto-Fix Start ---
# Check for Windows + CPU-only PyTorch and attempt to fix
try:
    import torch
    if os.name == 'nt' and not torch.cuda.is_available():
        if '+cpu' in torch.__version__:
            print("!" * 80)
            print(f"WARNING: Detected CPU-only PyTorch ({torch.__version__}) on Windows.")
            print("Attempting to automatically reinstall CUDA-enabled PyTorch...")
            print("!" * 80)
            
            try:
                # Uninstall existing torch packages
                subprocess.check_call([
                    sys.executable, "-m", "pip", "uninstall", "-y", 
                    "torch", "torchvision", "torchaudio"
                ])
                
                # Install GPU versions (using stable CUDA 12.x index)
                # Note: Using --no-cache-dir to avoid picking up the cached CPU wheel again
                print("Installing CUDA-enabled PyTorch... (This may take a while)")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", 
                    "torch", "torchvision", "torchaudio",
                    "--index-url", "https://download.pytorch.org/whl/cu124",
                    "--no-cache-dir"
                ])
                
                print("\n" + "=" * 80)
                print("SUCCESS: PyTorch reinstalled with CUDA support.")
                print("Please RESTART this application now to use the GPU.")
                print("=" * 80 + "\n")
                sys.exit(0)
                
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Failed to auto-install PyTorch: {e}")
                print("Please manually run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
                # Continue anyway, falling back to CPU logic below
except ImportError:
    pass
# --- Environment Auto-Fix End ---

import numpy as np
import imageio.v2 as iio
import torch.nn.functional as F
from flask import Flask, jsonify, render_template, request, send_file

from sharp.models import PredictorParams, RGBGaussianPredictor, create_predictor
from sharp.utils import io
from sharp.utils.gaussians import Gaussians3D, save_ply, unproject_gaussians

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
LOGGER = logging.getLogger(__name__)

# SILENCE WERKZEUG (HTTP LOGS)
# This prevents the console from being flooded by /job_status polling
try:
    # Set to ERROR to only see actual problems, not every GET request
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
except Exception:
    pass

# Flask app - use absolute paths for static and template folders
_base_dir = Path(__file__).parent.absolute()
app = Flask(
    __name__,
    static_folder=str(_base_dir / "webui_static"),
    static_url_path="/static",
    template_folder=str(_base_dir / "webui_templates")
)

# Global model cache
_model_cache = {"predictor": None, "device": None}

# Global job store for async video processing
_active_jobs = {}

# Output directory for generated PLY files
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Model URL
DEFAULT_MODEL_URL = "https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt"


def get_device() -> torch.device:
    """Get the best available device."""
    # If the model is already loaded, return the device it is on
    if _model_cache["device"] is not None:
        return _model_cache["device"]

    # Independent check if model isn't loaded yet
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def get_predictor() -> tuple[RGBGaussianPredictor, torch.device]:
    """Get or create the Gaussian predictor model."""
    if _model_cache["predictor"] is None:
        target_device = torch.device("cpu")
        
        # 1. Aggressive Device Detection & Diagnostics
        if torch.cuda.is_available():
            target_device = torch.device("cuda")
            try:
                gpu_name = torch.cuda.get_device_name(0)
                LOGGER.info(f"CUDA GPU detected: {gpu_name}")
            except Exception:
                LOGGER.info("CUDA GPU detected (name unknown)")
        elif torch.mps.is_available():
            target_device = torch.device("mps")
            LOGGER.info("Apple MPS acceleration detected.")
        else:
            LOGGER.info("No active GPU detected. Using CPU.")

        LOGGER.info(f"Targeting device for inference: {target_device}")

        # 2. Download and Load Model (Always load to CPU first for safety)
        LOGGER.info(f"Downloading model from {DEFAULT_MODEL_URL}")
        try:
            state_dict = torch.hub.load_state_dict_from_url(
                DEFAULT_MODEL_URL, 
                progress=True, 
                map_location="cpu"
            )
        except Exception as e:
            LOGGER.error(f"Failed to download/load model checkpoint: {e}")
            raise

        LOGGER.info("Initializing predictor...")
        predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state_dict)
        predictor.eval()
        
        # 3. Move to Target Device (with Fallback)
        final_device = torch.device("cpu")
        if target_device.type != "cpu":
            try:
                LOGGER.info(f"Moving model to {target_device}...")
                predictor.to(target_device)
                
                # Verify functionality with a dummy tensor
                dummy = torch.zeros(1).to(target_device)
                del dummy
                
                final_device = target_device
            except RuntimeError as e:
                LOGGER.warning(f"Failed to initialize on {target_device}: {e}.")
                LOGGER.warning("Falling back to CPU mode.")
                predictor.to("cpu")
                final_device = torch.device("cpu")
        else:
            predictor.to("cpu")

        _model_cache["predictor"] = predictor
        _model_cache["device"] = final_device
        LOGGER.info(f"Model successfully loaded and running on: {final_device}")

    return _model_cache["predictor"], _model_cache["device"]


@torch.no_grad()
def predict_image(
    predictor: RGBGaussianPredictor,
    image: np.ndarray,
    f_px: float,
    device: torch.device,
    use_fp16: bool = False  # <--- Added parameter
) -> Gaussians3D:
    """Predict Gaussians from an image."""
    internal_shape = (1536, 1536)

    image_pt = torch.from_numpy(image.copy()).float().to(device).permute(2, 0, 1) / 255.0
    _, height, width = image_pt.shape
    disparity_factor = torch.tensor([f_px / width]).float().to(device)

    image_resized_pt = F.interpolate(
        image_pt[None],
        size=(internal_shape[1], internal_shape[0]),
        mode="bilinear",
        align_corners=True,
    )

    # Predict Gaussians in the NDC space.
    # <--- Added FP16 Logic Context
    if use_fp16 and device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.float16):
            gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    else:
        gaussians_ndc = predictor(image_resized_pt, disparity_factor)
    # <--- End FP16 Logic

    intrinsics = (
        torch.tensor(
            [
                [f_px, 0, width / 2, 0],
                [0, f_px, height / 2, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        .float()
        .to(device)
    )
    intrinsics_resized = intrinsics.clone()
    intrinsics_resized[0] *= internal_shape[0] / width
    intrinsics_resized[1] *= internal_shape[1] / height

    # Convert Gaussians to metrics space.
    gaussians = unproject_gaussians(
        gaussians_ndc, torch.eye(4).to(device), intrinsics_resized, internal_shape
    )

    return gaussians


@app.route("/")
def index():
    """Serve the main page."""
    return render_template("index.html")


@app.route("/test")
def test_viewer():
    """Serve the test viewer page."""
    return render_template("test-viewer.html")


@app.route("/generate", methods=["POST"])
def generate():
    """Generate a 3DGS PLY file from an uploaded image."""
    if "image" not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Get Quality/FP16 Setting
    quality = request.form.get('quality', 'balanced')
    # If user selected 'fast', we use half-precision (FP16)
    use_fp16 = (quality == 'fast')

    allowed_extensions = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".tiff", ".tif", ".webp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        # Save uploaded file temporarily
        unique_id = str(uuid.uuid4())[:8]
        original_stem = Path(file.filename).stem

        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        LOGGER.info(f"Processing uploaded file: {file.filename} | Quality: {quality} | FP16: {use_fp16}")

        # Load the image
        image, _, f_px = io.load_rgb(tmp_path)
        height, width = image.shape[:2]

        # Get the model
        predictor, device = get_predictor()

        # Run prediction
        gaussians = predict_image(predictor, image, f_px, device, use_fp16=use_fp16)

        # Save the PLY file
        output_filename = f"{original_stem}_{unique_id}.ply"
        output_path = OUTPUT_DIR / output_filename
        save_ply(gaussians, f_px, (height, width), output_path)

        LOGGER.info(f"Saved PLY to: {output_path}")

        # Clean up temp file
        tmp_path.unlink()

        return jsonify({
            "success": True,
            "filename": output_filename,
            "download_url": f"/download/{output_filename}",
            "view_url": f"/ply/{output_filename}",
        })

    except Exception as e:
        LOGGER.exception("Error during generation")
        return jsonify({"error": str(e)}), 500


def _process_video_job(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16):
    """Background worker to process video frames."""
    reader = None
    try:
        # Pass 1: Get Total Frames
        # We open a dedicated reader just for counting to avoid iterator consumption issues
        total_frames = 0
        try:
            tmp_reader = iio.get_reader(tmp_path)
            total_frames = tmp_reader.count_frames()
            tmp_reader.close()
            _active_jobs[job_id]['total_frames'] = total_frames
        except Exception:
            _active_jobs[job_id]['total_frames'] = 0

        # Pass 2: Main Processing
        # Re-open reader for the actual processing loop
        reader = iio.get_reader(tmp_path)
        
        for i, frame in enumerate(reader):
            # Check for stop signal
            if _active_jobs[job_id]['stop_signal']:
                LOGGER.info(f"Job {job_id} stopped by user.")
                _active_jobs[job_id]['status'] = 'stopped'
                break

            try:
                # Frame is numpy array (H, W, C)
                if frame.shape[2] > 3:
                    frame = frame[:, :, :3]
                
                h, w = frame.shape[:2]
                f_px = io.convert_focallength(w, h, 30.0)

                # Predict
                gaussians = predict_image(predictor, frame, f_px, device, use_fp16=use_fp16)

                # Save PLY
                frame_filename = f"{original_stem}_{unique_id}_f{i:04d}.ply"
                output_path = OUTPUT_DIR / frame_filename
                save_ply(gaussians, f_px, (h, w), output_path)
                
                # Cleanup tensors to prevent memory accumulation causing hangs
                del gaussians
                
                # MEMORY MANAGEMENT
                # Crucial for preventing hangs on Apple MPS and long videos on CUDA
                if i % 2 == 0:  # Check frequently
                    if device.type == 'cuda':
                        torch.cuda.empty_cache()
                    elif device.type == 'mps':
                        try:
                            # MPS requires sync to prevent command buffer from filling up
                            torch.mps.empty_cache()
                            torch.mps.synchronize() 
                        except Exception:
                            pass # Fallback for older torch versions
                    
                    # Force Python Garbage Collection
                    gc.collect()
                    
            except Exception as e:
                LOGGER.error(f"Error processing frame {i}: {e}")
                # We continue to the next frame instead of crashing the whole job
                continue 

            # Update job state
            _active_jobs[job_id]['files'].append(frame_filename)
            _active_jobs[job_id]['processed_frames'] = i + 1
            
            # Log progress (Server side log) - reduced frequency
            if i % 10 == 0 or i == total_frames - 1:
                LOGGER.info(f"Job {job_id}: Processed frame {i+1} / {total_frames}")

        if not _active_jobs[job_id]['stop_signal']:
            _active_jobs[job_id]['status'] = 'done'

    except Exception as e:
        LOGGER.exception(f"Job {job_id} failed")
        _active_jobs[job_id]['status'] = 'error'
        _active_jobs[job_id]['error_msg'] = str(e)
    finally:
        # Cleanup
        if reader:
            try:
                reader.close()
            except:
                pass
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except:
                pass


@app.route("/generate_video", methods=["POST"])
def generate_video():
    """Start async video generation."""
    if "video" not in request.files:
        return jsonify({"error": "No video file provided"}), 400

    file = request.files["video"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    # Get Quality/FP16 Setting for video
    quality = request.form.get('quality', 'balanced')
    use_fp16 = (quality == 'fast')
    LOGGER.info(f"Starting video generation | Quality: {quality} | FP16: {use_fp16}")

    allowed_extensions = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed_extensions:
        return jsonify({"error": f"Unsupported file type: {ext}"}), 400

    try:
        unique_id = str(uuid.uuid4())[:8]
        original_stem = Path(file.filename).stem
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            file.save(tmp.name)
            tmp_path = Path(tmp.name)

        # Get metadata
        try:
            reader = iio.get_reader(tmp_path)
            meta = reader.get_meta_data()
            fps = meta.get('fps', 30.0)
            reader.close() 
        except Exception:
            fps = 30.0
        
        # Prepare job
        job_id = str(uuid.uuid4())
        _active_jobs[job_id] = {
            "status": "running",
            "files": [],
            "total_frames": 0,
            "processed_frames": 0,
            "stop_signal": False,
            "error_msg": "",
            "fps": fps
        }

        # Get model (ensure loaded)
        predictor, device = get_predictor()

        # Start thread
        thread = threading.Thread(
            target=_process_video_job,
            args=(job_id, tmp_path, original_stem, unique_id, predictor, device, fps, use_fp16)
        )
        thread.start()

        return jsonify({
            "success": True,
            "job_id": job_id,
            "status": "running"
        })

    except Exception as e:
        LOGGER.exception("Error starting video generation")
        return jsonify({"error": str(e)}), 500


@app.route("/job_status/<job_id>")
def job_status(job_id):
    """Check status of a background job."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    return jsonify({
        "status": job["status"],
        "processed": job["processed_frames"],
        "total": job["total_frames"],
        "files": job["files"], # Returns full list so client can see what's new
        "fps": job["fps"],
        "error": job["error_msg"],
        "base_url": "/ply/"
    })


@app.route("/stop_job/<job_id>", methods=["POST"])
def stop_job(job_id):
    """Signal a job to stop."""
    job = _active_jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    
    job["stop_signal"] = True
    return jsonify({"success": True, "status": "stopping"})


@app.route("/scan_local", methods=["POST"])
def scan_local():
    """Scan a local path for PLY files."""
    try:
        data = request.json
        if not data:
            return jsonify({"error": "Invalid JSON body"}), 400

        path_str = data.get("path")
        mode = data.get("mode") # 'single' or 'video'
        
        if not path_str:
            return jsonify({"error": "No path provided"}), 400

        # Clean path string (remove quotes users might copy-paste)
        path_str = path_str.strip().strip('"').strip("'")
        path = Path(path_str)
        
        if not path.exists():
            return jsonify({"error": f"Path does not exist: {path_str}"}), 404
            
        if mode == "single":
            if not path.is_file() or path.suffix.lower() != ".ply":
                 return jsonify({"error": "Path is not a valid PLY file"}), 400
            
            # Quote path to safely handle spaces in URL
            encoded_path = urllib.parse.quote(str(path))
            return jsonify({
                "success": True,
                "type": "single",
                "view_url": f"/view_local?path={encoded_path}",
                "filename": path.name
            })
            
        elif mode == "video":
            if not path.is_dir():
                 return jsonify({"error": "Path is not a directory"}), 400
            
            ply_files = sorted([f for f in path.glob("*.ply")])
            if not ply_files:
                 return jsonify({"error": "No PLY files found in directory"}), 400
                 
            # Create full URLs
            file_urls = []
            for p in ply_files:
                 encoded_path = urllib.parse.quote(str(p))
                 file_urls.append(f"/view_local?path={encoded_path}")
            
            return jsonify({
                "success": True,
                "type": "video",
                "fps": 30, # Defaulting to 30 as we can't infer from just files easily
                "frame_count": len(file_urls),
                "ply_files": file_urls,
                "base_url": "" # Empty because urls are fully formed
            })
        
        return jsonify({"error": "Invalid mode"}), 400
        
    except Exception as e:
        LOGGER.exception("Error scanning local path")
        # Return JSON error instead of 500 HTML
        return jsonify({"error": f"Server Error: {str(e)}"}), 500


@app.route("/view_local")
def view_local():
    """Serve a file from a local absolute path."""
    try:
        path_str = request.args.get("path")
        if not path_str:
            return "No path provided", 400
        
        # Check if path exists before sending
        path = Path(path_str)
        if not path.exists() or not path.is_file():
            return "File not found", 404

        return send_file(path, mimetype="application/octet-stream")
    except Exception as e:
        LOGGER.exception("Error serving local file")
        return str(e), 500


@app.route("/download/<filename>")
def download(filename: str):
    """Download a generated PLY file."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        as_attachment=True,
        download_name=filename,
        mimetype="application/octet-stream",
    )


@app.route("/ply/<filename>")
def serve_ply(filename: str):
    """Serve a PLY file for the viewer."""
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        return jsonify({"error": "File not found"}), 404

    return send_file(
        file_path,
        mimetype="application/octet-stream",
    )


@app.route("/status")
def status():
    """Get server status."""
    device = get_device()
    model_loaded = _model_cache["predictor"] is not None
    return jsonify({
        "status": "ok",
        "device": str(device),
        "model_loaded": model_loaded,
        "cuda_available": torch.cuda.is_available(),
    })


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ml-sharp WebUI")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=7860, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--preload", action="store_true", help="Preload model on startup")

    args = parser.parse_args()

    if args.preload:
        LOGGER.info("Preloading model...")
        get_predictor()

    LOGGER.info(f"Starting WebUI at http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)