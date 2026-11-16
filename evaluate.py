"""不训练，只复测已保存的 CNN：标准 MNIST 输入与 Python 画板预处理。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import ssl
import subprocess
import tempfile
import time

import certifi
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.datasets import MNIST

from model import DEFAULT_MODEL, ROOT, MEAN, STD, choose_device, load_model
from predict import image_to_tensor, preprocess_image


def wilson_interval(correct, total, z=1.959963984540054):
    """二项比例的 95% Wilson 区间；不等同于新数据分布上的表现保证。"""
    if total <= 0 or not 0 <= correct <= total:
        raise ValueError("需要 0 <= correct <= total 且 total > 0。")
    proportion = correct / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - half_width), min(1.0, center + half_width)]


def summarize_predictions(labels, predictions):
    """-1 表示预处理拒绝；拒绝样本保留在准确率和召回率的分母中。"""
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.ndim != 1 or predictions.shape != labels.shape or not len(labels):
        raise ValueError("标签与预测必须是同长度、非空的一维序列。")
    if np.any((labels < 0) | (labels > 9)) or np.any((predictions < -1) | (predictions > 9)):
        raise ValueError("标签必须为 0～9；预测只能为 -1 或 0～9。")
    matrix = np.zeros((10, 10), dtype=np.int64)
    accepted = predictions >= 0
    np.add.at(matrix, (labels[accepted], predictions[accepted]), 1)
    support = np.bincount(labels, minlength=10)
    rejected_by_class = np.bincount(labels[~accepted], minlength=10)
    per_class = []
    for digit in range(10):
        true_positive = int(matrix[digit, digit])
        predicted_count = int(matrix[:, digit].sum())
        precision = true_positive / predicted_count if predicted_count else 0.0
        recall = true_positive / int(support[digit]) if support[digit] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class.append({"digit": digit, "precision": precision, "recall": recall,
                          "f1": f1, "support": int(support[digit]), "correct": true_positive,
                          "rejected": int(rejected_by_class[digit])})
    pairs = [{"true_digit": truth, "predicted_digit": predicted, "count": int(matrix[truth, predicted])}
             for truth in range(10) for predicted in range(10)
             if truth != predicted and matrix[truth, predicted] > 0]
    pairs.sort(key=lambda item: (-item["count"], item["true_digit"], item["predicted_digit"]))
    correct = int((labels == predictions).sum())
    total = len(labels)
    return {"total": total, "correct": correct, "incorrect": total - correct,
            "accuracy": correct / total, "accuracy_wilson_95_ci": wilson_interval(correct, total),
            "accepted_count": int(accepted.sum()), "rejected_count": int((~accepted).sum()),
            "per_class": per_class, "macro_f1": sum(item["f1"] for item in per_class) / 10,
            "weighted_f1": sum(item["f1"] * item["support"] for item in per_class) / total,
            "confusion_matrix": matrix.tolist(), "confusion_matrix_rows": "true label 0..9",
            "confusion_matrix_columns": "predicted label 0..9; rejected inputs counted separately",
            "most_frequent_confusions": pairs[:10]}


@torch.inference_mode()
def evaluate_mode(model, dataset, device, batch_size, preprocessing, prepared_inputs=None, prepared_rejections=None):
    started = time.perf_counter()
    predictions = np.full(len(dataset), -1, dtype=np.int64)
    confidence = np.zeros(len(dataset), dtype=np.float64)
    rejected = prepared_rejections or []
    rejected_indices = {item["test_index"] for item in rejected}
    loss_sum, accepted_count = 0.0, 0
    for start in range(0, len(dataset), batch_size):
        stop = min(start + batch_size, len(dataset))
        indices = list(range(start, stop))
        if prepared_inputs is not None:
            indices = [index for index in indices if index not in rejected_indices]
            if not indices:
                continue
            inputs = torch.from_numpy(prepared_inputs[indices])
        elif preprocessing:
            prepared, accepted_indices = [], []
            for index in indices:
                try:
                    image = Image.fromarray(dataset.data[index].numpy())
                    prepared.append(image_to_tensor(preprocess_image(image)))
                    accepted_indices.append(index)
                except ValueError as exc:
                    rejected.append({"test_index": index, "reason": str(exc)})
            if not prepared:
                continue
            inputs = torch.cat(prepared)
            indices = accepted_indices
        else:
            inputs = (dataset.data[start:stop].to(torch.float32).unsqueeze(1) / 255.0 - MEAN) / STD
        logits = model(inputs.to(device))
        labels = dataset.targets[indices].to(device)
        loss_sum += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
        probabilities = logits.softmax(1)
        values, classes = probabilities.max(1)
        predictions[indices] = classes.cpu().numpy()
        confidence[indices] = values.cpu().numpy()
        accepted_count += len(indices)
    summary = summarize_predictions(dataset.targets.numpy(), predictions)
    mode = "browser_js_preprocessing_pytorch_cpu" if prepared_inputs is not None else "python_app_preprocessing" if preprocessing else "standard_mnist"
    summary.update({"mode": mode,
                    "cross_entropy_on_accepted_inputs": loss_sum / accepted_count if accepted_count else None,
                    "seconds": round(time.perf_counter() - started, 3), "rejections": rejected})
    wrong_indices = np.flatnonzero(predictions != dataset.targets.numpy())
    summary["errors"] = [{"test_index": int(index), "true_digit": int(dataset.targets[index]),
                           "predicted_digit": int(predictions[index]), "confidence": float(confidence[index])}
                          for index in wrong_indices]
    return summary


def prepare_browser_inputs(dataset):
    """通过同一份纯 JS 代码预处理，再回到 PyTorch 评估；不模拟浏览器解码或 WASM。"""
    source = ROOT / "docs" / "preprocess.js"
    script = r"""
import {readFileSync, writeFileSync} from 'node:fs';
const [input, output, rejectionOutput, moduleUrl] = process.argv.slice(1);
const {preprocessRGBA} = await import(moduleUrl);
const raw = readFileSync(input);
const processed = new Float32Array(raw.length);
const rejections = [];
for (let i = 0; i < raw.length / 784; i++) {
  const rgba = new Uint8Array(784 * 4);
  for (let p = 0; p < 784; p++) {
    const value = raw[i * 784 + p];
    rgba[p * 4] = rgba[p * 4 + 1] = rgba[p * 4 + 2] = value;
    rgba[p * 4 + 3] = 255;
  }
  try {
    processed.set(preprocessRGBA(rgba, 28, 28).tensor, i * 784);
  } catch (error) {
    rejections.push({test_index:i, reason:error.message});
  }
}
writeFileSync(output, Buffer.from(processed.buffer));
writeFileSync(rejectionOutput, JSON.stringify(rejections));
"""
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="mnist-browser-eval-") as folder:
        folder = Path(folder)
        raw, processed, rejections = folder / "raw.bin", folder / "processed.bin", folder / "rejections.json"
        dataset.data.numpy().tofile(raw)
        subprocess.run(["node", "--input-type=module", "--eval", script,
                        str(raw), str(processed), str(rejections), source.as_uri()],
                       check=True, cwd=ROOT, capture_output=True, text=True)
        inputs = np.fromfile(processed, dtype=np.float32).reshape(len(dataset), 1, 28, 28)
        rejected = json.loads(rejections.read_text(encoding="utf-8"))
    return inputs, rejected, round(time.perf_counter() - started, 3)


def draw_confusion_matrices(results, path):
    font = ImageFont.load_default(size=14)
    small = ImageFont.load_default(size=12)
    board = Image.new("RGB", (1110, 650), "#f8fafc")
    draw = ImageDraw.Draw(board)
    draw.text((24, 14), "MNIST test set: 10,000 images | rows = truth, columns = prediction", fill="#334155", font=font)
    for panel, (key, title) in enumerate((('standard_mnist', 'Standard input'), ('python_app_preprocessing', 'Python app preprocessing'))):
        result = results[key]
        matrix = np.asarray(result["confusion_matrix"])
        left, top, cell = 64 + panel * 550, 110, 43
        draw.text((left, 50), f"{title}: {result['accuracy']:.2%}", fill="#172554", font=font)
        for digit in range(10):
            draw.text((left + digit * cell + 17, top - 23), str(digit), fill="#334155", font=font)
            draw.text((left - 25, top + digit * cell + 13), str(digit), fill="#334155", font=font)
        row_totals = matrix.sum(1)
        for y in range(10):
            for x in range(10):
                ratio = matrix[y, x] / max(1, row_totals[y])
                # sqrt scale makes the off-diagonal errors visible; counts are always printed.
                intensity = math.sqrt(ratio)
                color = tuple(round(247 - intensity * (247 - component)) for component in (37, 99, 235))
                x0, y0 = left + x * cell, top + y * cell
                draw.rectangle((x0, y0, x0 + cell - 2, y0 + cell - 2), fill=color)
                label = str(int(matrix[y, x]))
                width = draw.textlength(label, font=small)
                draw.text((x0 + (cell - width) / 2, y0 + 14), label,
                          fill="white" if ratio > 0.4 else "#1e293b", font=small)
        draw.text((left, top + 10 * cell + 16), f"Correct {result['correct']:,} | Errors {result['incorrect']:,} | Rejected {result['rejected_count']}", fill="#334155", font=font)
    draw.text((24, 613), "Color: square-root of within-row proportion. Counts shown in every cell. No retraining or test-driven tuning.", fill="#64748b", font=small)
    board.save(path)


def draw_errors(dataset, result, path):
    font = ImageFont.load_default(size=13)
    small = ImageFont.load_default(size=11)
    board = Image.new("RGB", (960, 500), "#f8fafc")
    draw = ImageDraw.Draw(board)
    draw.text((16, 12), "First 24 misclassified test images (dataset order) | Standard MNIST input", fill="#334155", font=font)
    for slot, error in enumerate(result["errors"][:24]):
        x, y = (slot % 8) * 120 + 12, (slot // 8) * 145 + 43
        raw = Image.fromarray(dataset.data[error["test_index"]].numpy())
        board.paste(raw.resize((72, 72), Image.Resampling.NEAREST), (x + 12, y))
        draw.text((x, y + 78), f"true {error['true_digit']}  pred {error['predicted_digit']}", fill="#b91c1c", font=font)
        draw.text((x, y + 98), f"score {error['confidence']:.1%}", fill="#64748b", font=small)
        draw.text((x, y + 114), f"test index {error['test_index']}", fill="#64748b", font=small)
    board.save(path)


def make_markdown(report):
    standard, pipeline = report["standard_mnist"], report["python_app_preprocessing"]
    lines = ["# 已保存模型的准确率复测", "", "本报告由 `evaluate.py` 生成，不重新训练，也不根据测试结果选择或修改权重。", "",
             "## 评估范围", "", "- 数据：官方 MNIST 测试集，完整 10,000 张图片。",
             "- 标准输入：直接使用 MNIST 的 28×28 灰度图片，像素除以 255 后使用 mean=0.1307、std=0.3081 标准化。",
             "- Python 画板流程：同一批图片先经过 `predict.py` 的极性统一、裁剪、等比缩放和重心居中，再送进同一模型。",
             "- 推理使用 `eval()` 和 `inference_mode()`；所有样本都计入准确率分母，预处理拒绝也算未识别正确。",
             "- 这是已用于第一次报告的同一测试集上的复测，不是一份新收集的独立保留集。",
             "- 画板流程结果只衡量对 MNIST 图片重复预处理后的影响，不能当作真实用户笔迹或手机照片的准确率。浏览器版的插值实现还可能与 Pillow 不同。", "",
             f"模型：`{report['checkpoint_filename']}`；SHA-256：`{report['checkpoint_sha256']}`。", "",
             f"软件：PyTorch {report['torch_version']}；评估设备：{report['device']}。", "",
             "## 总体结果", "", "| 指标 | 标准 MNIST 输入 | Python 画板预处理 |", "| --- | ---: | ---: |",
             f"| 准确率 | **{standard['accuracy']:.2%}** | **{pipeline['accuracy']:.2%}** |",
             f"| 识别正确 | {standard['correct']:,} / 10,000 | {pipeline['correct']:,} / 10,000 |",
             f"| 识别错误或拒绝 | {standard['incorrect']:,} | {pipeline['incorrect']:,} |",
             f"| 预处理拒绝 | {standard['rejected_count']} | {pipeline['rejected_count']} |",
             f"| Macro F1 | {standard['macro_f1']:.4f} | {pipeline['macro_f1']:.4f} |",
             f"| 交叉熵损失（接受的输入） | {standard['cross_entropy_on_accepted_inputs']:.6f} | {pipeline['cross_entropy_on_accepted_inputs']:.6f} |"]
    intervals = [f"{result['accuracy_wilson_95_ci'][0]:.2%}～{result['accuracy_wilson_95_ci'][1]:.2%}" for result in (standard, pipeline)]
    lines += [f"| 准确率 95% Wilson 区间 | {intervals[0]} | {intervals[1]} |", "",
              "Wilson 区间描述二项抽样不确定性；不覆盖数据分布变化、训练随机性或测试集被反复查看的影响。", "",
              f"重复预处理相对标准输入改变了 **{(pipeline['accuracy'] - standard['accuracy']) * 100:+.2f} 个百分点**。", ""]
    if "browser_js_preprocessing_pytorch_cpu" in report:
        browser = report["browser_js_preprocessing_pytorch_cpu"]
        interval = browser["accuracy_wilson_95_ci"]
        lines += ["## 部署版 JavaScript 预处理复测", "",
                  "使用 Node.js 执行网页实际使用的 `docs/preprocess.js`，对同一批 10,000 张 MNIST 图片进行处理，然后交给同一个 PyTorch CPU 模型推理。",
                  "这验证了部署版的像素预处理路径；不是浏览器 WASM 的端到端准确率测量，也不包含浏览器的图片解码、上传缩图或实际用户笔迹。", "",
                  f"- 准确率：**{browser['accuracy']:.2%}（{browser['correct']:,} / {browser['total']:,}）**；错误或拒绝 {browser['incorrect']} 张，预处理拒绝 {browser['rejected_count']} 张。",
                  f"- Macro F1：{browser['macro_f1']:.4f}；95% Wilson 区间：{interval[0]:.2%}～{interval[1]:.2%}。",
                  f"- 预处理源码 SHA-256：`{report['browser_preprocessor_sha256']}`；Node.js {report['node_version']}。", "",
                  "| 数字 | JS 预处理 Recall |", "| --- | ---: |"]
        for row in browser["per_class"]:
            lines.append(f"| {row['digit']} | {row['recall']:.2%} |")
        lines.append("")
    lines += [
              "## 各数字表现", "", "Precision：预测为这个数字的图片中，实际正确的比例。Recall：实际属于这个数字的图片中，被正确认出的比例。F1：两者的调和平均。", "",
              "| 数字 | 样本数 | 标准 Precision | 标准 Recall | 标准 F1 | 画板 Recall |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for a, b in zip(standard["per_class"], pipeline["per_class"]):
        lines.append(f"| {a['digit']} | {a['support']} | {a['precision']:.2%} | {a['recall']:.2%} | {a['f1']:.4f} | {b['recall']:.2%} |")
    lines += ["", "## 最常见的混淆", "", "| 真实数字 → 预测数字 | 标准输入次数 |", "| --- | ---: |"]
    for pair in standard["most_frequent_confusions"]:
        lines.append(f"| {pair['true_digit']} → {pair['predicted_digit']} | {pair['count']} |")
    lines += ["", "![标准输入与画板流程的混淆矩阵](confusion_matrices.png)", "",
              "下面按测试集原始顺序展示前 24 个错误，没有挑选最容易或最难的例子。置信分数高也可能识别错误。", "",
              "![标准输入的前 24 个错误](misclassified_examples.png)", "", "## 复现", "", "在项目根目录运行（完整复测额外需要 Node.js）：", "", "```bash",
              ".venv/bin/python evaluate.py --device cpu" + (" --include-browser" if "browser_js_preprocessing_pytorch_cpu" in report else ""), "```", "",
              "省略 `--include-browser` 时只评估 Python 的两条路径，不需要 Node.js。", "",
              "首次运行如本地缺少数据会下载官方 MNIST 数据；已有缓存时不需要重新下载。详细逐类指标、完整混淆矩阵、错误索引和模型校验值保存在 `evaluation.json`。", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--include-browser", action="store_true", help="额外用 Node.js 运行网页同款预处理，再用 PyTorch CPU 推理")
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("batch-size 必须是正数。")
    torch.set_num_threads(4)
    device = choose_device(args.device)
    if ssl.get_default_verify_paths().cafile is None:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    model, checkpoint = load_model(args.model, device)
    if checkpoint.get("architecture") != "DigitCNN":
        parser.error("本脚本评估 CNN，请使用 artifacts/mnist_cnn.pt；Linear 使用不同的输入标准化。")
    dataset = MNIST(args.data_dir, train=False, download=True)
    report = {"evaluated_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": "MNIST official test split",
              "checkpoint_filename": args.model.name, "checkpoint_sha256": hashlib.sha256(args.model.read_bytes()).hexdigest(),
              "checkpoint_epoch": checkpoint.get("epoch"), "device": str(device), "torch_version": str(torch.__version__),
              "retrained": False, "new_holdout": False, "normalization": {"mean": MEAN, "std": STD},
              "methodology": "Re-evaluation of the previously reported 10,000-image test split; no test-driven tuning. Python app preprocessing results do not measure real-user handwriting or browser preprocessing."}
    for mode, enabled in (("standard_mnist", False), ("python_app_preprocessing", True)):
        print(f"评估 {mode}：{len(dataset):,} 张图片……", flush=True)
        report[mode] = evaluate_mode(model, dataset, device, args.batch_size, enabled)
        result = report[mode]
        print(f"  {result['correct']:,}/{result['total']:,} = {result['accuracy']:.2%}；拒绝 {result['rejected_count']}；{result['seconds']:.2f} 秒", flush=True)
    if args.include_browser:
        print("评估 browser_js_preprocessing_pytorch_cpu：10,000 张图片……", flush=True)
        inputs, rejected, prep_seconds = prepare_browser_inputs(dataset)
        # 这一模式明确固定为 CPU，与模式名称和部署报告一致。
        model.cpu()
        result = evaluate_mode(model, dataset, torch.device("cpu"), args.batch_size, False, inputs, rejected)
        result["preprocessing_seconds"] = prep_seconds
        result["inference_device"] = "cpu"
        report["browser_js_preprocessing_pytorch_cpu"] = result
        report["browser_preprocessor_sha256"] = hashlib.sha256((ROOT / "docs" / "preprocess.js").read_bytes()).hexdigest()
        report["node_version"] = subprocess.check_output(["node", "--version"], text=True).strip()
        print(f"  {result['correct']:,}/{result['total']:,} = {result['accuracy']:.2%}；拒绝 {result['rejected_count']}；JS 预处理 {prep_seconds:.2f} 秒", flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evaluation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "evaluation.md").write_text(make_markdown(report), encoding="utf-8")
    draw_confusion_matrices(report, args.output_dir / "confusion_matrices.png")
    draw_errors(dataset, report["standard_mnist"], args.output_dir / "misclassified_examples.png")
    print(f"已保存评估报告：{args.output_dir / 'evaluation.md'}", flush=True)


if __name__ == "__main__":
    main()
