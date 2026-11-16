"""我们自己定义的小型 CNN：把 [B, 1, 28, 28] 图片变成 [B, 10] 分数。"""

from pathlib import Path

import torch
from torch import nn

ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL = ROOT / "artifacts" / "mnist_cnn.pt"
DEFAULT_LINEAR_MODEL = ROOT / "mnist_linear_weights.pt"
MEAN, STD = 0.1307, 0.3081


class DigitCNN(nn.Module):
    normalization = (MEAN, STD)

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 28 × 28 -> 14 × 14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 14 × 14 -> 7 × 7
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
        )

    def forward(self, images):
        # 返回原始分数（logits）。训练的 CrossEntropyLoss 自带所需转换。
        return self.classifier(self.features(images))


class DigitLinear(nn.Sequential):
    """对应你在 Python 中亲手创建的 Flatten → Linear(784, 10)。"""

    normalization = (0.0, 1.0)  # 与 tutorial 的 ToTensor() 一致，保留 0～1。

    def __init__(self):
        super().__init__(nn.Flatten(), nn.Linear(784, 10))


def choose_device(requested="auto"):
    if requested == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise ValueError("当前环境无法使用 MPS，请选择 --device cpu。")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("当前环境无法使用 CUDA，请选择 --device cpu。")
    return torch.device(requested)


def load_model(path=DEFAULT_MODEL, device="cpu"):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"还没有模型：{path}。请先运行 python train.py。")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(saved, dict) and set(saved) == {"1.weight", "1.bias"}:
        # torch.save(model.state_dict(), ...) 保存的 tutorial Linear weights。
        model = DigitLinear()
        state = saved
        checkpoint = {"architecture": "Linear"}
    elif isinstance(saved, dict) and "model_state" in saved:
        model = DigitCNN()
        state = saved["model_state"]
        checkpoint = dict(saved)
        checkpoint["architecture"] = "DigitCNN"
    else:
        raise ValueError("不支持这个 checkpoint；需要 tutorial Linear state_dict 或本项目的 CNN checkpoint。")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, checkpoint
