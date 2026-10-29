#!/usr/bin/env python3
"""
UANet Interactive Web Demonstration Server
Serves the web dashboard, precomputed remote sensing evaluation assets,
and handles optional live PyTorch inference on uploaded satellite/drone imagery.
"""

import os
import sys
import json
import base64
import io
import mimetypes
import webbrowser
from http.server import HTTPServer, BaseHTTPRequestHandler
import urllib.parse

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
WEB_DIR = os.path.join(REPO_ROOT, "web")
EVAL_DIR = os.path.join(REPO_ROOT, "evaluation_results")
CKPT_PATH = os.path.join(REPO_ROOT, "model_weights", "whubuilding", "UANet_VGG", "UANet_VGG.ckpt")

# Check model inference availability
INFERENCE_READY = False
MODEL = None
DEVICE = "cpu"

try:
    import torch
    import torch.nn.functional as F
    import numpy as np
    from PIL import Image
    import cv2
    from geoseg.models.UANet import UANet_VGG

    if os.path.exists(CKPT_PATH):
        print(f"[*] Found checkpoint at {CKPT_PATH}. Initializing UANet-VGG...")
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        net = UANet_VGG(channel=32, num_classes=2)
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
        state_dict = ckpt["state_dict"]
        # Strip 'net.' prefix from Lightning state dict
        cleaned_sd = {k.replace("net.", ""): v for k, v in state_dict.items() if k.startswith("net.")}
        net.load_state_dict(cleaned_sd, strict=False)
        net.to(DEVICE)
        net.eval()
        MODEL = net
        INFERENCE_READY = True
        print(f"[+] Live inference engine initialized successfully on {DEVICE}!")
    else:
        print("[!] Model checkpoint not found; serving precomputed benchmark dataset.")
except Exception as e:
    print(f"[*] Note: Running in lightweight preview mode (live inference requires: torch, torchvision, opencv-python). Reason: {e}")


def run_inference_on_pil_image(pil_img):
    """Run UANet model inference on a PIL image and return base64 results."""
    if not INFERENCE_READY or MODEL is None:
        return None

    w, h = pil_img.size
    img_512 = pil_img.resize((512, 512), Image.BILINEAR)
    img_np = np.array(img_512).astype(np.float32) / 255.0
    
    # Normalize RGB (Mean [0.485, 0.456, 0.406], Std [0.229, 0.224, 0.225])
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_norm = (img_np - mean) / std
    
    tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(tensor)
        # UANet returns [m5, m4, m3, m2, m1]
        m1_logits = outputs[4]
        probs = F.softmax(m1_logits, dim=1).squeeze(0).cpu().numpy()

        # Class 0: Building, Class 1: Background
        bldg_prob = probs[0]
        mask_binary = (bldg_prob > 0.5).astype(np.uint8) * 255

        # Uncertainty: 0.5 - |0.5 - p|
        raw_unc = 0.5 - np.abs(0.5 - bldg_prob)
        norm_unc = np.clip(raw_unc / 0.5, 0.0, 1.0)
        unc_colored = cv2.applyColorMap((norm_unc * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
        unc_rgb = cv2.cvtColor(unc_colored, cv2.COLOR_BGR2RGB)

    # Encode mask to PNG base64
    mask_pil = Image.fromarray(mask_binary)
    mask_buffer = io.BytesIO()
    mask_pil.save(mask_buffer, format="PNG")
    mask_b64 = "data:image/png;base64," + base64.b64encode(mask_buffer.getvalue()).decode()

    # Encode uncertainty to PNG base64
    unc_pil = Image.fromarray(unc_rgb)
    unc_buffer = io.BytesIO()
    unc_pil.save(unc_buffer, format="PNG")
    unc_b64 = "data:image/png;base64," + base64.b64encode(unc_buffer.getvalue()).decode()

    # Calculate statistics
    total_px = 512 * 512
    bldg_px = int(np.sum(mask_binary > 0))
    coverage = round((bldg_px / total_px) * 100, 2)

    # Simple connected components for building count
    num_labels, _ = cv2.connectedComponents((mask_binary > 0).astype(np.uint8))
    num_buildings = max(0, num_labels - 1)

    return {
        "prediction_mask_url": mask_b64,
        "uncertainty_map_url": unc_b64,
        "building_area_percent": coverage,
        "building_pixel_count": bldg_px,
        "num_buildings": num_buildings,
        "device": DEVICE
    }


class UANetDemoHandler(BaseHTTPRequestHandler):
    def end_headers(self):
        # Allow CORS
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path

        # Root route -> web/index.html
        if path in ["/", "/index.html"]:
            self.serve_file(os.path.join(WEB_DIR, "index.html"), "text/html")
            return

        # Serve static web directory
        if path.startswith("/web/"):
            rel_path = path[5:]
            self.serve_file(os.path.join(WEB_DIR, rel_path))
            return

        # Serve evaluation results
        if path.startswith("/evaluation_results/"):
            rel_path = path[20:]
            self.serve_file(os.path.join(EVAL_DIR, rel_path))
            return

        # Serve data
        if path.startswith("/data/"):
            rel_path = path[6:]
            self.serve_file(os.path.join(REPO_ROOT, "data", rel_path))
            return

        # API: Samples and metrics
        if path == "/api/samples":
            metrics_file = os.path.join(EVAL_DIR, "metrics_summary.json")
            if os.path.exists(metrics_file):
                with open(metrics_file, "r") as f:
                    data = json.load(f)
                self.send_json(data)
            else:
                self.send_json({"status": "no_metrics_file"})
            return

        # API: Health check
        if path == "/api/health":
            self.send_json({
                "status": "healthy",
                "inference_ready": INFERENCE_READY,
                "device": DEVICE
            })
            return

        # 404
        self.send_error(404, f"File not found: {path}")

    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/api/predict":
            content_type = self.headers.get("Content-Type", "")
            content_len = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_len)

            if not INFERENCE_READY:
                self.send_json({
                    "error": "Live inference requires torchvision and timm. To enable: pip install -r requirement.txt torchvision",
                    "status": "inference_unavailable"
                }, status=503)
                return

            try:
                # Handle raw binary image or multipart
                if "multipart/form-data" in content_type:
                    # Parse multipart boundary
                    boundary = content_type.split("boundary=")[1].encode()
                    parts = body.split(boundary)
                    img_bytes = None
                    for part in parts:
                        if b"filename=" in part:
                            # Header is separated by \r\n\r\n from content
                            header_end = part.find(b"\r\n\r\n")
                            if header_end != -1:
                                img_bytes = part[header_end + 4:].rstrip(b"\r\n--")
                                break
                    if not img_bytes:
                        self.send_json({"error": "No image payload found in multipart"}, status=400)
                        return
                    pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                else:
                    pil_img = Image.open(io.BytesIO(body)).convert("RGB")

                result = run_inference_on_pil_image(pil_img)
                if result:
                    self.send_json(result)
                else:
                    self.send_json({"error": "Inference returned empty result"}, status=500)
            except Exception as ex:
                self.send_json({"error": str(ex)}, status=500)
            return

        self.send_error(404, "Endpoint not found")

    def serve_file(self, file_path, forced_type=None):
        if not os.path.exists(file_path) or os.path.isdir(file_path):
            self.send_error(404, f"Not Found: {file_path}")
            return

        mime_type = forced_type or mimetypes.guess_type(file_path)[0] or "application/octet-stream"
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            self.send_response(200)
            self.send_header("Content-Type", mime_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Cache-Control", "public, max-age=3600")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self.send_error(500, f"Error reading file: {e}")

    def send_json(self, data, status=200):
        content = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main():
    port = 8501
    server_address = ("", port)
    
    # Try preferred port 8501 or fallback
    try:
        httpd = HTTPServer(server_address, UANetDemoHandler)
    except OSError:
        port = 8080
        server_address = ("", port)
        httpd = HTTPServer(server_address, UANetDemoHandler)

    url = f"http://localhost:{port}"
    print("\n" + "=" * 70)
    print("  [UANET] REMOTE SENSING BUILDING EXTRACTION WEB DASHBOARD")
    print("=" * 70)
    print(f" [+] Web Dashboard is LIVE at: {url}")
    print(f" [+] Open your browser at:    {url}")
    print("=" * 70 + "\n")

    # Try opening browser automatically
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[*] Shutting down web server.")
        httpd.server_close()


if __name__ == "__main__":
    main()
