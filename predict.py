"""识别自己的单个数字图片：python predict.py your_digit.png。"""

import argparse
import base64
from io import BytesIO

import numpy as np
from PIL import Image, ImageOps
import torch

from model import DEFAULT_MODEL, MEAN, STD, choose_device, load_model


def preprocess_image(image: Image.Image) -> Image.Image:
    """统一极性、裁掉空白、等比缩放，并按墨迹重心居中到 MNIST 的 28×28。"""
    if image.width < 2 or image.height < 2:
        raise ValueError("图片太小，请提供一张清晰的数字图片。")
    if image.width * image.height > 16_000_000:
        raise ValueError("图片过大，请缩小至 1600 万像素以内。")
    image = ImageOps.exif_transpose(image)
    # 透明背景先铺白色，避免透明 PNG 的 RGB 值影响极性判断。
    rgba = image.convert("RGBA")
    background = Image.new("RGBA", rgba.size, "white")
    background.alpha_composite(rgba)
    gray = background.convert("L")
    gray.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
    values = np.asarray(gray, dtype=np.float32)
    border = np.concatenate((values[0], values[-1], values[:, 0], values[:, -1]))
    if np.median(border) > 127:
        values = 255.0 - values  # 白底黑字 -> MNIST 的黑底白字
    values = np.maximum(values - np.median(np.concatenate(
        (values[0], values[-1], values[:, 0], values[:, -1])
    )), 0)
    peak = float(values.max())
    if peak < 25:
        raise ValueError("还没有看到数字，请先写一个 0～9 的数字。")
    mask = values > max(25, peak * 0.2)
    ys, xs = np.nonzero(mask)
    if len(xs) < 5:
        raise ValueError("笔迹太少，请写一个完整的数字。")
    values[values < peak * 0.08] = 0
    ink = Image.fromarray(np.uint8(np.clip(values * (255.0 / peak), 0, 255)))
    ink = ink.crop((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1))
    scale = 20 / max(ink.size)
    ink = ink.resize((max(1, round(ink.width * scale)), max(1, round(ink.height * scale))),
                     Image.Resampling.LANCZOS)
    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(ink, ((28 - ink.width) // 2, (28 - ink.height) // 2))
    pixels = np.asarray(canvas, dtype=np.float32)
    y, x = np.indices(pixels.shape)
    mass = float(pixels.sum())
    dx = round(13.5 - float((pixels * x).sum()) / mass)
    dy = round(13.5 - float((pixels * y).sum()) / mass)
    # paste 会裁切越界像素，而不是像 roll 一样把边缘绕回另一边。
    centered = Image.new("L", (28, 28), 0)
    centered.paste(canvas, (dx, dy))
    return centered


def image_to_tensor(processed, normalization=(MEAN, STD)):
    pixels = np.array(processed, dtype=np.float32) / 255.0
    mean, std = normalization
    return ((torch.from_numpy(pixels) - mean) / std).unsqueeze(0).unsqueeze(0)


@torch.inference_mode()
def predict_image(model, image, device="cpu"):
    processed = preprocess_image(image)
    normalization = getattr(model, "normalization", (MEAN, STD))
    logits = model(image_to_tensor(processed, normalization).to(device))
    probabilities = logits.softmax(dim=1)[0].cpu().tolist()
    prediction = max(range(10), key=probabilities.__getitem__)
    output = BytesIO()
    processed.save(output, format="PNG")
    return {
        "prediction": prediction,
        "confidence": probabilities[prediction],
        "probabilities": probabilities,
        "processed_image": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", help="只包含一个数字的图片路径")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    args = parser.parse_args()
    try:
        device = choose_device(args.device)
        model, _ = load_model(args.model, device)
        with Image.open(args.image) as image:
            result = predict_image(model, image, device)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"无法识别：{exc}\n")
    print(f"预测数字：{result['prediction']}")
    print(f"模型置信度：{result['confidence']:.2%}（不代表保证正确的概率）")
    for digit, probability in enumerate(result["probabilities"]):
        print(f"  {digit}: {probability:7.2%}")


if __name__ == "__main__":
    main()
