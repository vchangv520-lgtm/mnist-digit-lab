"""画板 HTTP 接口的离线验证，使用临时随机权重和系统分配的空闲端口。"""

import base64
from contextlib import contextmanager
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from io import BytesIO
import hashlib
import json
from pathlib import Path
import tempfile
from threading import Thread
import unittest

from PIL import Image, ImageDraw
import torch

from app import make_handler
from model import DigitCNN


@contextmanager
def running_server(model_path):
    handler = make_handler(model_path, "cpu")
    handler.log_message = lambda *args: None
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.02}, daemon=True)
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def request(port, method, path, body=None, headers=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


class DrawingAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = tempfile.TemporaryDirectory()
        cls.path = Path(cls.folder.name) / "model.pt"
        torch.save({"model_state": DigitCNN().state_dict(), "epoch": 1}, cls.path)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_missing_model_is_reported_as_not_ready(self):
        with running_server(Path(self.folder.name) / "missing.pt") as port:
            status, body = request(port, "GET", "/api/status")
            self.assertEqual(status, 200)
            self.assertFalse(body["ready"])
            self.assertIsNone(body["metrics"])
            status, body = request(port, "POST", "/api/predict", "{}", {"Content-Type": "application/json"})
            self.assertEqual(status, 503)
            self.assertIn("error", body)

    def test_malformed_requests_return_json_errors(self):
        with running_server(self.path) as port:
            cases = [
                ("{", {"Content-Type": "application/json"}, 400),
                ("[]", {"Content-Type": "application/json"}, 400),
                ('{"image":"data:image/png;base64,not base64!"}', {"Content-Type": "application/json"}, 400),
                ('{"image":"data:image/png;base64,aGVsbG8="}', {"Content-Type": "application/json"}, 400),
                ("{}", {"Content-Type": "text/plain"}, 415),
                (b"", {"Content-Type": "application/json", "Content-Length": "3000001"}, 413),
            ]
            for payload, headers, expected_status in cases:
                with self.subTest(payload=payload):
                    status, body = request(port, "POST", "/api/predict", payload, headers)
                    self.assertEqual(status, expected_status)
                    self.assertIsInstance(body["error"], str)

    def test_canvas_png_round_trips_through_prediction_api(self):
        image = Image.new("RGB", (140, 140), "white")
        ImageDraw.Draw(image).line([(45, 30), (90, 30), (58, 112)], fill="black", width=9)
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        payload = json.dumps({"image": "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode()})
        with running_server(self.path) as port:
            status, body = request(port, "GET", "/api/status")
            self.assertEqual(status, 200)
            self.assertTrue(body["ready"])
            status, body = request(port, "POST", "/api/predict", payload, {"Content-Type": "application/json"})
            self.assertEqual(status, 200)
            self.assertIn(body["prediction"], range(10))
            self.assertEqual(len(body["probabilities"]), 10)
            self.assertAlmostEqual(sum(body["probabilities"]), 1, places=6)
            self.assertTrue(body["processed_image"].startswith("data:image/png;base64,"))

    def test_linear_status_does_not_borrow_neighboring_cnn_metrics(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "linear_weights.pt"
            manual_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
            torch.save(manual_model.state_dict(), path)
            (path.parent / "metrics.json").write_text(
                json.dumps({"test_accuracy": 0.999, "epochs": 3, "parameter_count": 206_922}),
                encoding="utf-8",
            )
            with running_server(path) as port:
                status, body = request(port, "GET", "/api/status")
            self.assertEqual(status, 200)
            self.assertTrue(body["ready"])
            self.assertEqual(body["parameter_count"], 7_850)
            self.assertEqual(body["model_file"], path.name)
            self.assertIn("linear", body["model_architecture"].lower())
            self.assertTrue(body["model_name"])
            self.assertIsNone(body["metrics"])
            self.assertIsNone(body["best_epoch"])

    def test_linear_metrics_must_match_the_loaded_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "linear_weights.pt"
            manual_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10))
            torch.save(manual_model.state_dict(), path)
            actual_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            for recorded_hash, accepted in (("stale-checkpoint-hash", False), (actual_hash, True)):
                with self.subTest(matching_weights=accepted):
                    metrics = {"weights_sha256": recorded_hash, "test_accuracy": 0.91}
                    path.with_suffix(".metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
                    with running_server(path) as port:
                        status, body = request(port, "GET", "/api/status")
                    self.assertEqual(status, 200)
                    self.assertEqual(body["metrics"], metrics if accepted else None)


if __name__ == "__main__":
    unittest.main()
