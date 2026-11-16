"""导出同一份 CNN 权重供 GitHub Pages 使用，并复核 ONNX 与 PyTorch 的预测。"""

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np
import onnx
import onnxruntime as ort
from PIL import Image
import torch
from torchvision.datasets import MNIST

from model import DEFAULT_MODEL, ROOT, MEAN, STD, load_model


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.inference_mode()
def validate_export(model, onnx_path, test_data, batch_size=256):
    """同一批标准化后的像素，比较两个运行时；不重新训练或调整模型。"""
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    total, agreement, torch_correct, onnx_correct = 0, 0, 0, 0
    max_absolute_logit_error = 0.0
    for start in range(0, len(test_data), batch_size):
        pixels = test_data.data[start:start + batch_size].to(torch.float32).unsqueeze(1) / 255.0
        images = ((pixels - MEAN) / STD).contiguous()
        labels = test_data.targets[start:start + batch_size].numpy()
        torch_logits = model(images).numpy()
        onnx_logits = session.run(["logits"], {"images": images.numpy()})[0]
        torch_predictions = torch_logits.argmax(axis=1)
        onnx_predictions = onnx_logits.argmax(axis=1)
        total += len(labels)
        agreement += int((torch_predictions == onnx_predictions).sum())
        torch_correct += int((torch_predictions == labels).sum())
        onnx_correct += int((onnx_predictions == labels).sum())
        max_absolute_logit_error = max(max_absolute_logit_error, float(np.max(np.abs(torch_logits - onnx_logits))))
        if not np.allclose(torch_logits, onnx_logits, atol=1e-4, rtol=1e-4):
            raise RuntimeError("ONNX 与 PyTorch 的 logits 不满足数值误差阈值，停止发布。")
    if agreement != total:
        raise RuntimeError(f"ONNX 与 PyTorch 预测存在差异：{total - agreement}/{total}。")
    # 确认交互页面用到的 batch=1 也符合导出的动态维度契约。
    sample = ((test_data.data[:1].to(torch.float32).unsqueeze(1) / 255.0 - MEAN) / STD).numpy()
    if session.run(["logits"], {"images": sample})[0].shape != (1, 10):
        raise RuntimeError("ONNX 单张图片输出形状不正确。")
    return {"test_count": total, "matching_predictions": agreement,
            "prediction_agreement": agreement / total,
            "pytorch_correct": torch_correct, "onnx_correct": onnx_correct,
            "pytorch_accuracy": torch_correct / total, "onnx_accuracy": onnx_correct / total,
            "max_absolute_logit_error": max_absolute_logit_error,
            "comparison_atol": 1e-4, "comparison_rtol": 1e-4,
            "method": "Same normalized official MNIST test tensors; PyTorch CPU vs ONNX Runtime CPU. This validates export equivalence, not browser handwriting preprocessing.",
            "torch_version": str(torch.__version__), "onnx_version": onnx.__version__,
            "onnxruntime_version": ort.__version__}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "docs")
    parser.add_argument("--report", type=Path, default=ROOT / "reports" / "onnx_validation.json")
    args = parser.parse_args()
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.model, "cpu")
    if checkpoint.get("architecture") != "DigitCNN":
        parser.error("网页版导出脚本仅支持 CNN，请使用 artifacts/mnist_cnn.pt。")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "mnist_cnn.onnx"
    torch.onnx.export(
        model, (torch.zeros(2, 1, 28, 28),), str(onnx_path),
        input_names=["images"], output_names=["logits"],
        dynamic_shapes={"images": {0: torch.export.Dim("batch", min=1, max=1024)}},
        opset_version=18, dynamo=True, external_data=False,
    )
    onnx.checker.check_model(onnx.load(str(onnx_path)))
    # 原项目已经下载数据；导出不会悄悄联网下载或改变训练结果。
    test_data = MNIST(args.data_dir, train=False, download=False)
    report = validate_export(model, onnx_path, test_data)
    report.update({"checkpoint_sha256": sha256(args.model), "onnx_sha256": sha256(onnx_path),
                   "onnx_bytes": onnx_path.stat().st_size, "onnx_opset": 18,
                   "browser_runtime_version": "1.31.1"})
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metrics = args.model.parent / "metrics.json"
    if metrics.is_file():
        shutil.copyfile(metrics, args.output_dir / "metrics.json")
    examples = args.output_dir / "examples"
    examples.mkdir(exist_ok=True)
    for digit in range(10):
        index = int((test_data.targets == digit).nonzero()[0].item())
        Image.fromarray(test_data.data[index].numpy()).save(examples / f"digit_{digit}.png")
    print(f"ONNX: {onnx_path}")
    print(f"预测一致：{report['matching_predictions']:,}/{report['test_count']:,}")
    print(f"ONNX 测试准确率：{report['onnx_accuracy']:.2%}")
    print(f"最大 logits 绝对差：{report['max_absolute_logit_error']:.8g}")


if __name__ == "__main__":
    main()
