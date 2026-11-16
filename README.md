# 第一个 PyTorch 项目：手写数字识别

[在线手写画板](https://vchangv520-lgtm.github.io/mnist-digit-lab/) · [公开仓库](https://github.com/vchangv520-lgtm/mnist-digit-lab) · [完整准确率评估](reports/evaluation.md)

给电脑一张只包含一个数字的图片，让它判断这个数字是 **0～9** 中的哪一个。我们从随机初始化的权重开始训练一个小型卷积神经网络（CNN），使用 MNIST 手写数字数据集学习，并在本地网页里测试自己的笔迹。

这里的“从零训练”指自己定义模型、执行训练、保存权重，不使用预训练模型。PyTorch 帮我们计算梯度和执行张量运算。

## 这台 Mac 上已经完成的第一轮实验

2026 年 11 月 16 日，已在 Apple M3、8GB 内存、Python 3.12 的独立环境中完成训练。仓库附带这次训练的模型权重和报告；原始 MNIST 数据集在首次训练或评估时下载。

| 实测项目 | 结果 |
| --- | ---: |
| 训练设备 | Apple GPU（MPS） |
| 训练轮数 / 参数数 | 3 / 206,922 |
| 最佳验证准确率 | 98.70%（第 3 轮） |
| 独立测试准确率 | **98.58%（9,858 / 10,000）** |
| 训练与最终评估计时 | 13.88 秒，不含首次安装和下载 |
| 权重文件大小 | 约 0.83 MB |

原始记录在 `artifacts/metrics.json`。浏览器手写「7」和命令行样本识别均已跑通；图片处理与 HTTP 接口的 9 项自动测试通过。这里的成绩针对 MNIST 测试集，自己的笔迹仍需要实际尝试。

**现在就试：** 打开 [在线画板](https://vchangv520-lgtm.github.io/mnist-digit-lab/)，写一个数字，然后点击「识别数字」。在线版在浏览器中运行导出的 ONNX 模型，图片不上传到服务器。若想使用本地 PyTorch 版，在项目目录运行 `.venv/bin/python app.py`，再打开 [本地画板](http://127.0.0.1:7860)。本地画板默认用 CPU 进行单张图片推理，训练默认使用 MPS。

## 准确率复测与公开部署

2026 年 11 月 17 日，对同一份已保存的 CNN 权重重新评估；没有重新训练或用测试集调参。

| 评估流程 | 正确数 / 10,000 | 准确率 |
| --- | ---: | ---: |
| 标准 MNIST 输入，PyTorch CPU | 9,858 | **98.58%** |
| Python 画板预处理后，PyTorch CPU | 9,788 | **97.88%** |
| 网页同款 JavaScript 预处理后，PyTorch CPU | 9,788 | **97.88%** |
| 标准 MNIST 输入，导出的 ONNX CPU 模型 | 9,858 | **98.58%** |

标准输入的 95% Wilson 区间为 **98.33%～98.79%**。最常见混淆是 **9 被认成 7，共 12 次**。完整的逐类 Precision、Recall、F1、混淆矩阵及错误图片见 [评估报告](reports/evaluation.md)。这是第一次报告所用测试集的复测；经过预处理的 MNIST 成绩不等于真实用户笔迹成绩。

导出前后在全部 10,000 张标准输入上的预测一致，[导出验证记录](reports/onnx_validation.json)保留了模型校验值和数值误差。JavaScript 预处理复测在 Node.js 中运行，后续推理使用 PyTorch CPU；浏览器 WASM 另做实际交互验证，两者的验证范围不同。

安装基础依赖后，复现评估：

```bash
.venv/bin/python evaluate.py --device cpu
# 可选：安装 Node.js 后，一并复测网页同款预处理。
.venv/bin/python evaluate.py --device cpu --include-browser
```

GitHub Pages 从 `main` 分支的 `docs/` 发布纯静态网站。在线版的模型、JavaScript 和 WASM 均随站点提供，访客无需安装 Python。更新模型后可重新导出并推送：

```bash
.venv/bin/python -m pip install -r requirements-web.txt
.venv/bin/python export_web.py
.venv/bin/python -m http.server 7862 --directory docs
```

导出验证使用本地 MNIST 缓存；新克隆的项目请先执行 `evaluate.py` 下载测试数据。浏览器打开 `http://localhost:7862` 可预览静态版。发布机制参考 [GitHub Pages 官方说明](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site)，推理部署参考 [ONNX Runtime Web 官方说明](https://onnxruntime.ai/docs/tutorials/web/deploy.html)。第三方运行时的许可证保留在 `docs/vendor/`。

仓库另保留学习过程中的 `Flatten → Linear(784, 10)` 模型及其本地画板，见 [Linear 画板说明](DRAWING.md)。在线版和上面的评估均使用 CNN；两种模型的权重和成绩分别保存。

## 在本地运行项目

克隆仓库后进入项目目录。下面的命令都在这个目录中执行：

```bash
git clone https://github.com/vchangv520-lgtm/mnist-digit-lab.git
cd mnist-digit-lab
```

### 1. 准备 Python 环境

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

`.venv` 是这个项目独立的 Python 环境。后面直接使用 `.venv/bin/python`，无需先执行环境激活命令。

仓库已经附带训练好的模型。只想体验本地画板，可以跳过训练，直接执行第 3 步。

### 2. 训练模型

```bash
.venv/bin/python train.py
```

首次运行会下载 MNIST 数据到 `data/`，需要联网。默认训练 3 轮，每批处理 128 张图片。程序会选择可用的计算设备：在支持 MPS 的 Apple 芯片 Mac 上使用 GPU，没有可用 GPU 时使用 CPU。

训练完成后，查看 `artifacts/`：

| 文件 | 用途 |
| --- | --- |
| `mnist_cnn.pt` | 保存的模型权重和相关信息，用于以后识别图片 |
| `metrics.json` | 训练、验证和最终测试的实际指标 |
| `training.svg` | 训练过程曲线 |
| `samples.png` | 样本预测示例 |
| `examples/digit_0.png` ～ `digit_9.png` | 各取一张测试集图片，供上传或命令行试用 |

终端输出和 `metrics.json` 中的结果是本次运行的实测值；每次训练的结果可能有小幅差别。

### 3. 在网页里写一个数字

```bash
.venv/bin/python app.py
```

打开 [本地手写数字实验室](http://127.0.0.1:7860)，在画板上写一个数字，然后点击识别。也可以上传数字图片。

先写比较大、居中的单个数字，尽量保留四周空白。网页会显示模型的判断、十个类别的预测概率，以及送入模型的 28 × 28 图片。试试自己的 `2`、`4`、`7`，观察哪些写法容易让模型困惑。

结束使用时，在运行网页服务的终端按 `Ctrl+C`。再次使用时直接启动 `app.py`，已保存的模型无需重新训练。

也可以通过命令行识别一张图片：

```bash
.venv/bin/python predict.py path/to/image.png --device cpu
```

把 `path/to/image.png` 换成真实文件路径；路径中包含空格时，用引号括起来。

## 模型究竟学到了什么？

我们没有写“有一个圆就是 0”之类的规则。训练时，模型看到图片和正确答案，先猜一个结果，再根据错误调整内部参数。这个过程重复很多次，模型逐渐学会利用笔画、转角和形状来区分数字。

MNIST 的每张图片是 **28 × 28 像素的灰度图**，因此一张图片只有一个颜色通道。训练时把多张图片放成一批：

```text
输入形状：[B, 1, 28, 28]
           │  │   │   └─ 宽度
           │  │   └───── 高度
           │  └───────── 灰度通道
           └──────────── 本批图片数，通常为 128

输出形状：[B, 10]
           │   └──────── 每张图片对应 0～9 的十个分数
           └──────────── 本批图片数
```

这十个原始分数称为 **logits**，它们还不是概率。预测时，`softmax` 可以把分数转换成总和为 1 的十个数；取分数最大的类别，就是模型的判断。

### 今天用的网络

```text
28 × 28 灰度图
  → 卷积：1 个通道变为 16 个通道 → ReLU → 最大池化
  → 卷积：16 个通道变为 32 个通道 → ReLU → 最大池化
  → 展平：32 × 7 × 7 = 1568 个数
  → 全连接层：1568 → 128 → ReLU → Dropout
  → 全连接层：128 → 10 个类别分数
```

- **卷积层**用可学习的小滤镜扫描图片，提取局部图案；后面的层可以组合前面的特征。
- **ReLU**把负值变成 0，让网络能够学习更复杂的关系。
- **池化层**缩小特征图，减少后续计算。
- **全连接层**综合特征，为十个数字分别打分。
- **Dropout**只在训练时随机屏蔽一部分中间值，帮助减少对个别特征的过度依赖。

## 五行代码理解训练

训练循环的核心是：

```python
optimizer.zero_grad()                  # 清空上一次计算的梯度
logits = model(images)                  # 看图片，给出十个类别分数
loss = loss_fn(logits, labels)          # 与正确答案比较，计算损失
loss.backward()                        # 计算每个参数应如何调整
optimizer.step()                       # 按梯度更新参数
```

这里的 `loss_fn` 是 `nn.CrossEntropyLoss()`。**传给它的是原始 logits，不要先做 softmax**；交叉熵函数内部已经包含相应的计算。标签是数字类别，例如 `3`，而不是一张数字图片。

几个常见词可以这样理解：

| 术语 | 在这个项目中的意思 |
| --- | --- |
| batch（批） | 一次用于计算损失和更新参数的一组图片 |
| epoch（轮） | 完整遍历一次训练集 |
| loss（损失） | 衡量预测与正确答案差距的数值，训练要尽量降低它 |
| accuracy（准确率） | 预测正确的图片数 ÷ 总图片数 |
| learning rate（学习率） | 控制每次参数调整的步幅 |
| gradient（梯度） | 损失相对于参数的变化信息，用来指导更新方向 |

训练前调用 `model.train()`；评估和识别前调用 `model.eval()`，让 Dropout 等层切换到正确的工作状态。评估时还要关闭梯度计算，避免不必要的开销。

## 为什么要分训练、验证和测试三部分？

MNIST 提供 60,000 张训练图片和 10,000 张测试图片。本项目进一步拆分原始训练集：

| 数据划分 | 数量 | 用途 |
| --- | ---: | --- |
| 训练集 | 55,000 | 计算梯度，更新模型参数 |
| 验证集 | 5,000 | 每轮检查学习效果，选择保存哪一轮模型 |
| 测试集 | 10,000 | 训练结束后，评估最终选出的模型 |

三部分互不重叠。程序保存验证准确率最高的模型，再用它进行最终测试。测试集不参与参数更新或最佳轮次选择。

如果不断根据测试结果调整模型，就会逐渐把测试集也当成“练习题”。学习项目中可以尝试改进，但应清楚区分验证结果和最终测试结果。

## 接着做三个小实验

1. **观察训练过程。** 打开 `training.svg`，看看训练损失是否下降，验证准确率是否上升。如果训练越来越好而验证没有改善，继续训练未必有帮助。
2. **只改变训练轮数。** 先训练 1 轮，再训练 5 轮，比较各自的验证指标、耗时和手写体验。使用不同输出目录保存实验结果：

   ```bash
   .venv/bin/python train.py --epochs 1 --output-dir artifacts/one_epoch
   .venv/bin/python train.py --epochs 5 --output-dir artifacts/five_epochs
   ```

3. **研究一个错误。** 在画板里找到一个被认错的数字，稍微改变笔画、位置或大小后再试。观察 28 × 28 预处理图，而不只是看原图。

常用训练参数：

```bash
.venv/bin/python train.py \
  --epochs 3 \
  --batch-size 128 \
  --learning-rate 0.001 \
  --device auto \
  --seed 42 \
  --data-dir data \
  --output-dir artifacts
```

`--seed 42` 固定随机种子，便于比较实验；不同设备和软件版本之间不一定得到完全相同的数值。指定其他输出目录后，默认网页仍使用 `artifacts/mnist_cnn.pt`；体验其他实验时可运行 `.venv/bin/python app.py --model artifacts/five_epochs/mnist_cnn.pt`。切换模型前先停止旧的网页服务。

## 遇到问题时

- **提示找不到 `torch`：** 检查是否使用了 `.venv/bin/python`，并重新执行依赖安装命令。
- **下载失败：** 检查网络连接后重新运行训练命令。已经下载完成的数据会被复用。
- **MPS 运行出错：** 使用 `.venv/bin/python train.py --device cpu` 跑通流程；CPU 也能完成这个小模型的训练。
- **内存或其他程序压力较大：** 关闭暂时不用的应用，并尝试 `--batch-size 64`。
- **网页提示没有模型：** 先完成默认的 `train.py` 训练，再重新启动网页服务。
- **自己的数字经常被认错：** 先用白色背景、深色粗笔画、单个数字试验。手机照片中的阴影、复杂背景、旋转和多个数字会增加难度。

MNIST 测试准确率描述的是模型对这份测试集的表现，不能直接代表它对所有人的笔迹或手机照片的准确率。网页里的概率是模型输出的相对判断，没有经过置信度校准；即使显示很高的概率，也可能认错。

## 建议的阅读顺序

维护代码后，可运行 `.venv/bin/python -m unittest discover -s tests -v` 检查图片预处理、模型保存和本地接口。这些测试使用临时随机权重，不需要重新训练或联网。

网页版的预处理回归测试运行 `node --test tests/test_preprocess.mjs`。测试使用合成像素，覆盖空白图、黑白极性、透明背景、比例与居中、舍入边界以及 softmax 数值稳定性。

先读 `model.py` 看网络结构，再读 `train.py` 找到上面五行训练代码，最后读 `predict.py` 看图片如何变成模型输入。`app.py` 把识别功能接到本地网页，`web/index.html` 提供画板和结果展示。

需要查原始说明时，可以阅读 [PyTorch 入门教程](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)、[Apple 芯片上的 MPS 后端说明](https://docs.pytorch.org/docs/stable/notes/mps.html)和 [torchvision 的 MNIST 数据集文档](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)。
