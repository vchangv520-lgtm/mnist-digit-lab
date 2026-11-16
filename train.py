"""从随机权重开始训练 MNIST：python train.py --epochs 3。"""

import argparse
import json
import os
import random
import ssl
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import certifi
from PIL import Image, ImageDraw
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from model import ROOT, MEAN, STD, DigitCNN, choose_device, load_model


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()  # 关闭 Dropout，考试时不随机丢弃信息。
    loss_sum, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * len(labels)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += len(labels)
    return {"loss": loss_sum / count, "accuracy": correct / count, "correct": correct, "count": count}


def save_chart(history, path):
    """直接生成可缩放 SVG，无需再安装绘图库。"""
    pieces = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="340" viewBox="0 0 900 340">',
              '<rect width="900" height="340" fill="#f8fafc"/>',
              '<style>text{font-family:system-ui,sans-serif;font-size:12px;fill:#334155}</style>']
    for panel, title, keys, ceiling in [
        (0, "Loss (lower is better)", ("train_loss", "validation_loss"), max(max(h["train_loss"], h["validation_loss"]) for h in history) * 1.15),
        (1, "Accuracy (higher is better)", ("train_accuracy", "validation_accuracy"), 1.0),
    ]:
        left, top, width, height = 60 + panel * 450, 55, 355, 210
        pieces.append(f'<text x="{left}" y="28" font-weight="bold">{title}</text>')
        for step in range(5):
            y = top + height * step / 4
            value = ceiling * (1 - step / 4)
            label = f"{value:.0%}" if panel else f"{value:.2f}"
            pieces.append(f'<path d="M {left} {y} h {width}" stroke="#e2e8f0"/>')
            pieces.append(f'<text x="{left-8}" y="{y+4}" text-anchor="end">{label}</text>')
        for key, color, label in zip(keys, ("#2563eb", "#e07a39"), ("Train", "Validation")):
            points = []
            for index, row in enumerate(history):
                x = left + width * index / max(1, len(history) - 1)
                y = top + height * (1 - row[key] / ceiling)
                points.append(f"{x:.1f},{y:.1f}")
                pieces.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
            pieces.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')
            legend_x = left + (130 if label == "Validation" else 0)
            pieces.append(f'<text x="{legend_x}" y="320" style="fill:{color}">{label}</text>')
        for index, row in enumerate(history):
            x = left + width * index / max(1, len(history) - 1)
            pieces.append(f'<text x="{x}" y="290" text-anchor="middle">Epoch {row["epoch"]}</text>')
    pieces.append('</svg>')
    path.write_text("\n".join(pieces), encoding="utf-8")


@torch.inference_mode()
def save_samples(model, test_data, device, output_dir):
    from PIL import ImageFont
    font = ImageFont.load_default(size=14)
    board = Image.new("RGB", (800, 430), "#f8fafc")
    draw = ImageDraw.Draw(board)
    draw.text((16, 12), "Unseen test samples | label = truth, pred = model", fill="#334155", font=font)
    images = torch.stack([test_data[i][0] for i in range(32)]).to(device)
    predicted = model(images).argmax(1).cpu().tolist()
    for i, prediction in enumerate(predicted):
        label = int(test_data.targets[i])
        raw = Image.fromarray(test_data.data[i].numpy())
        x, y = (i % 8) * 100 + 15, (i // 8) * 95 + 42
        board.paste(raw.resize((56, 56)), (x + 7, y))
        draw.text((x, y + 61), f"label {label} pred {prediction}",
                  fill="#16834b" if prediction == label else "#cf3c3c", font=font)
    board.save(output_dir / "samples.png")
    examples = output_dir / "examples"
    examples.mkdir(exist_ok=True)
    for digit in range(10):
        index = int((test_data.targets == digit).nonzero()[0].item())
        raw = Image.fromarray(test_data.data[index].numpy())
        raw.save(examples / f"digit_{digit}.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or not 0 < args.learning_rate < float("inf"):
        parser.error("epochs、batch-size 和 learning-rate 必须是正数。")
    device = choose_device(args.device)
    torch.set_num_threads(4)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    print(f"设备：{device} | PyTorch {torch.__version__} | 随机种子：{args.seed}", flush=True)
    print("准备 MNIST，首次运行需要下载约 12 MB 数据……", flush=True)
    # python.org 的 macOS 安装有时没有默认 CA 文件；补充可信证书，保留 HTTPS 验证。
    if ssl.get_default_verify_paths().cafile is None:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])
    full_train = datasets.MNIST(args.data_dir, train=True, download=True, transform=transform)
    test_data = datasets.MNIST(args.data_dir, train=False, download=True, transform=transform)
    train_data, validation_data = random_split(
        full_train, [55_000, 5_000], generator=torch.Generator().manual_seed(args.seed)
    )
    # Mac 入门项目用单进程加载，减少启动开销和额外内存占用。
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    validation_loader = DataLoader(validation_data, batch_size=256, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=256, num_workers=0)
    model = DigitCNN().to(device)  # 随机初始化，没有加载任何预训练权重。
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "mnist_cnn.pt"
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"参数：{parameter_count:,} | 训练：55,000 | 验证：5,000 | 测试：10,000", flush=True)
    history, best_accuracy, best_epoch = [], -1.0, 0
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        model.train()
        loss_sum, correct, count = 0.0, 0, 0
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()          # 1. 清空上一批梯度
            logits = model(images)         # 2. 前向传播：预测
            loss = criterion(logits, labels)  # 3. 计算与正确答案的差距
            loss.backward()                # 4. 反向传播：求梯度
            optimizer.step()               # 5. 更新参数
            loss_sum += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            count += len(labels)
            if batch_index % 100 == 0:
                print(f"  第 {epoch} 轮 | {count:>5,}/55,000 | 平均 loss {loss_sum/count:.4f}", flush=True)
        validation = evaluate(model, validation_loader, criterion, device)
        row = {"epoch": epoch, "train_loss": loss_sum / count, "train_accuracy": correct / count,
               "validation_loss": validation["loss"], "validation_accuracy": validation["accuracy"],
               "seconds": round(time.perf_counter() - epoch_started, 2)}
        history.append(row)
        print(f"第 {epoch}/{args.epochs} 轮 | loss {row['train_loss']:.4f} | "
              f"训练准确率 {row['train_accuracy']:.2%} | 验证准确率 {row['validation_accuracy']:.2%} | "
              f"{row['seconds']:.1f} 秒", flush=True)
        if validation["accuracy"] > best_accuracy:
            best_accuracy, best_epoch = validation["accuracy"], epoch
            torch.save({"format_version": 1, "architecture": "DigitCNN",
                        "model_state": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "epoch": epoch, "validation_accuracy": best_accuracy, "seed": args.seed,
                        "normalization": {"mean": MEAN, "std": STD}}, checkpoint_path)
    # 用验证集选出的最佳模型，只在训练结束后进行最终测试。
    model, _ = load_model(checkpoint_path, device)
    test_result = evaluate(model, test_loader, criterion, device)
    metrics = {"created_at": datetime.now().astimezone().isoformat(), "device": str(device),
               "torch_version": str(torch.__version__), "seed": args.seed, "epochs": args.epochs,
               "batch_size": args.batch_size, "learning_rate": args.learning_rate,
               "train_size": 55_000, "validation_size": 5_000, "test_size": 10_000,
               "parameter_count": parameter_count, "best_epoch": best_epoch,
               "best_validation_accuracy": best_accuracy, "test_accuracy": test_result["accuracy"],
               "test_loss": test_result["loss"], "test_correct": test_result["correct"],
               "elapsed_seconds": round(time.perf_counter() - started, 2), "history": history}
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    save_chart(history, args.output_dir / "training.svg")
    save_samples(model, test_data, device, args.output_dir)
    print(f"\n最终测试：{test_result['correct']:,}/10,000，准确率 {test_result['accuracy']:.2%}", flush=True)
    print(f"最佳模型来自第 {best_epoch} 轮；已保存到 {checkpoint_path}", flush=True)
    print("下一步：python app.py，然后在浏览器打开 http://127.0.0.1:7860", flush=True)


if __name__ == "__main__":
    main()
