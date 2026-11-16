"""本机手写数字画板。启动：python app.py，打开 http://127.0.0.1:7860。"""

import argparse
import base64
import binascii
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
import json
from pathlib import Path
from threading import Lock
from urllib.parse import urlsplit

from PIL import Image, UnidentifiedImageError
import torch

from model import ROOT, DEFAULT_MODEL, DEFAULT_LINEAR_MODEL, choose_device, load_model
from predict import predict_image


def make_handler(model_path, device):
    model = None
    checkpoint = {}
    if model_path.is_file():
        model, checkpoint = load_model(model_path, device)
    is_linear = checkpoint.get("architecture") == "Linear" or (model is None and model_path.name == DEFAULT_LINEAR_MODEL.name)
    metrics_path = model_path.with_suffix(".metrics.json") if is_linear else model_path.parent / "metrics.json"
    metrics = None
    if model and metrics_path.is_file():
        try:
            candidate = json.loads(metrics_path.read_text(encoding="utf-8"))
            if not is_linear or candidate.get("weights_sha256") == hashlib.sha256(model_path.read_bytes()).hexdigest():
                metrics = candidate
        except (OSError, ValueError, AttributeError):
            pass  # 缺少可用 metrics 不妨碍使用已加载的 model。
    model_name = "Linear classifier" if is_linear else "CNN"
    launch_command = ".venv/bin/python draw.py" if is_linear else ".venv/bin/python app.py"
    missing_message = ("请先保存 mnist_linear_weights.pt，再重启画板。" if is_linear else
                       "请先运行 .venv/bin/python train.py，完成后重启画板。")
    inference_lock = Lock()

    class Handler(BaseHTTPRequestHandler):
        def respond(self, status, body, content_type="application/json; charset=utf-8"):
            if not isinstance(body, bytes):
                body = json.dumps(body, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == "/":
                self.respond(200, (ROOT / "web" / "index.html").read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self.respond(200, {"ready": model is not None, "device": str(device),
                                   "metrics": metrics, "best_epoch": checkpoint.get("epoch"),
                                   "model_name": model_name, "model_file": model_path.name,
                                   "model_architecture": "Linear" if is_linear else "DigitCNN",
                                   "parameter_count": sum(p.numel() for p in model.parameters()) if model else None,
                                   "launch_command": launch_command, "missing_message": missing_message})
            elif path == "/favicon.ico":
                self.respond(204, b"", "image/x-icon")
            else:
                self.respond(404, {"error": "页面不存在。"})

        def do_POST(self):
            if urlsplit(self.path).path != "/api/predict":
                self.respond(404, {"error": "接口不存在。"})
                return
            if model is None:
                self.respond(503, {"error": missing_message})
                return
            if self.headers.get_content_type() != "application/json":
                self.respond(415, {"error": "请使用 JSON 格式提交图片。"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 3_000_000:
                    self.respond(413, {"error": "图片过大或为空，请选择小于 2 MB 的图片。"})
                    return
                body = json.loads(self.rfile.read(length))
                data_url = body.get("image") if isinstance(body, dict) else None
                if not isinstance(data_url, str) or not data_url.startswith("data:image/") or ";base64," not in data_url:
                    raise ValueError("请提交有效的图片。")
                raw = base64.b64decode(data_url.split(";base64,", 1)[1], validate=True)
                with Image.open(BytesIO(raw)) as image:
                    if image.width * image.height > 16_000_000:
                        raise ValueError("图片过大，请缩小至 1600 万像素以内。")
                    with inference_lock:
                        result = predict_image(model, image, device)
                self.respond(200, result)
            except (ValueError, UnicodeError, binascii.Error, UnidentifiedImageError,
                    Image.DecompressionBombError, OSError) as exc:
                self.respond(400, {"error": str(exc) or "无法读取这张图片，请重新选择。"})

    return Handler


def main(default_model=DEFAULT_MODEL, default_port=7860):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=default_port)
    parser.add_argument("--model", type=Path, default=default_model)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    device = choose_device(args.device)
    handler = make_handler(args.model, device)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), handler)
    print(f"画板已启动：http://127.0.0.1:{args.port}（按 Ctrl+C 停止）", flush=True)
    if not args.model.is_file():
        print(f"尚未找到 weights：{args.model}。保存后重启画板即可。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n画板已关闭。", flush=True)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
