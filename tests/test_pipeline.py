"""离线验证：图片预处理、模型保存和预测接口，不下载数据、不训练模型。"""

import base64
from io import BytesIO
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch

from model import DigitCNN, load_model
from predict import image_to_tensor, predict_image, preprocess_image


def handwritten_one(mode="L"):
    image = Image.new(mode, (160, 120), "white")
    draw = ImageDraw.Draw(image)
    draw.line([(100, 28), (110, 18), (110, 96)], fill="black", width=9)
    return image


class ImagePipelineTests(unittest.TestCase):
    def test_blank_images_are_rejected_instead_of_guessed(self):
        for color in (0, 255):
            with self.subTest(background=color), self.assertRaises(ValueError):
                preprocess_image(Image.new("L", (160, 120), color))
        with self.assertRaises(ValueError):
            preprocess_image(Image.new("RGBA", (160, 120), (0, 0, 0, 0)))

    def test_black_and_white_ink_produce_the_same_input(self):
        dark_ink = handwritten_one()
        light_ink = ImageOps.invert(dark_ink)
        np.testing.assert_array_equal(
            np.asarray(preprocess_image(dark_ink)),
            np.asarray(preprocess_image(light_ink)),
        )

    def test_transparent_background_matches_white_background(self):
        transparent = Image.new("RGBA", (160, 120), (0, 0, 0, 0))
        draw = ImageDraw.Draw(transparent)
        draw.line([(100, 28), (110, 18), (110, 96)], fill=(0, 0, 0, 255), width=9)
        np.testing.assert_array_equal(
            np.asarray(preprocess_image(transparent)),
            np.asarray(preprocess_image(handwritten_one())),
        )

    def test_off_center_digit_stays_narrow_and_is_centered(self):
        image = Image.new("L", (200, 180), 255)
        ImageDraw.Draw(image).rectangle((150, 20, 159, 79), fill=0)
        processed = preprocess_image(image)
        self.assertEqual(processed.size, (28, 28))
        pixels = np.asarray(processed, dtype=np.float64)
        y, x = np.indices(pixels.shape)
        self.assertLessEqual(abs((pixels * x).sum() / pixels.sum() - 13.5), 0.5)
        self.assertLessEqual(abs((pixels * y).sum() / pixels.sum() - 13.5), 0.5)
        ys, xs = np.nonzero(pixels > 127)
        ratio = (xs.max() - xs.min() + 1) / (ys.max() - ys.min() + 1)
        self.assertAlmostEqual(ratio, 10 / 60, delta=0.06)

    def test_prediction_has_ten_probabilities_and_a_real_preview(self):
        model = DigitCNN().eval()
        result = predict_image(model, handwritten_one())
        probabilities = result["probabilities"]
        self.assertEqual(len(probabilities), 10)
        self.assertAlmostEqual(sum(probabilities), 1.0, places=6)
        self.assertTrue(all(0 <= value <= 1 for value in probabilities))
        self.assertEqual(result["prediction"], int(np.argmax(probabilities)))
        self.assertEqual(result["confidence"], max(probabilities))
        prefix, encoded = result["processed_image"].split(",", 1)
        self.assertEqual(prefix, "data:image/png;base64")
        with Image.open(BytesIO(base64.b64decode(encoded))) as preview:
            self.assertEqual(preview.size, (28, 28))
            tensor = image_to_tensor(preview)
        self.assertEqual(tuple(tensor.shape), (1, 1, 28, 28))
        self.assertEqual(tensor.dtype, torch.float32)
        self.assertTrue(torch.isfinite(tensor).all())

    def test_saved_weights_restore_identical_predictions(self):
        model = DigitCNN().eval()
        inputs = image_to_tensor(preprocess_image(handwritten_one()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "model.pt"
            torch.save({"model_state": model.state_dict()}, path)
            restored, _ = load_model(path, "cpu")
            self.assertFalse(restored.training)
            with torch.inference_mode():
                torch.testing.assert_close(model(inputs), restored(inputs))


class LinearCheckpointTests(unittest.TestCase):
    def test_raw_linear_weights_restore_the_users_architecture(self):
        manual_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)).eval()
        inputs = torch.linspace(0, 1, 2 * 784).reshape(2, 1, 28, 28)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "linear_weights.pt"
            torch.save(manual_model.state_dict(), path)
            restored, _ = load_model(path, "cpu")
            self.assertEqual(sum(parameter.numel() for parameter in restored.parameters()), 7_850)
            self.assertFalse(restored.training)
            with torch.inference_mode():
                torch.testing.assert_close(restored(inputs), manual_model(inputs))

    def test_linear_prediction_uses_zero_to_one_pixels(self):
        manual_model = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(784, 10)).eval()
        with torch.no_grad():
            # 不同类别具有不同权重，误用 CNN 标准化会改变这些概率。
            manual_model[1].weight.copy_(torch.linspace(-0.004, 0.004, 10)[:, None].expand(10, 784))
            manual_model[1].bias.zero_()
        image = handwritten_one()
        pixels = np.asarray(preprocess_image(image), dtype=np.float32).copy() / 255.0
        manual_input = torch.from_numpy(pixels).reshape(1, 1, 28, 28)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "linear_weights.pt"
            torch.save(manual_model.state_dict(), path)
            restored, _ = load_model(path, "cpu")
            result = predict_image(restored, image)
            with torch.inference_mode():
                expected = manual_model(manual_input).softmax(1)[0]
            torch.testing.assert_close(torch.tensor(result["probabilities"]), expected)


if __name__ == "__main__":
    unittest.main()
