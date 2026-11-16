# 用你训练的 Linear model 手写 prediction

画板读取项目中的 `mnist_linear_weights.pt`，恢复你在 Python interactive mode 里创建的 `Flatten → Linear(784, 10)`。共 7,850 个 parameters，input values 保持 `0～1`，与当时的 `ToTensor()` 一致。

保存的 weights 已重新 load，并在原始 MNIST test dataset 上复现 **9,084 / 10,000，test accuracy 90.84%**。这次 evaluation 没有执行 parameter updates。记录保存在 `mnist_linear_weights.metrics.json`，只有记录中的文件 hash 与当前 weights 一致时，画板才显示这份 test accuracy。

## 打开画板

启动服务后打开 [本地 Linear 画板](http://127.0.0.1:7861)，确认页面显示 `Linear classifier` 和 `mnist_linear_weights.pt`。点击「清空」，用鼠标、触控板或触屏写一个数字，再点击「识别数字」。

如果服务已经关闭，在一个新的 Terminal 窗口输入下面两行。它们是 Terminal commands，不是在 Python 的 `>>>` 中输入：

```bash
cd /path/to/mnist-digit-lab
.venv/bin/python draw.py
```

保留这个 Terminal 窗口即可使用画板。结束时按 Control + C 停止服务。正在学习的 Python interactive window 可以继续保留。

将 `/path/to/mnist-digit-lab` 换成你克隆项目的位置；依赖安装方法见 README。

## 一次 prediction 如何发生

```text
手写 canvas
  → preprocessing：统一为黑底白字、裁掉空白、保持比例缩放、居中到 28 × 28
  → tensor：pixel values / 255，形成 [1, 1, 28, 28] 的一个 batch
  → Flatten：784 个 input features
  → Linear：使用你保存的 weights 和 biases 计算 10 个 logits
  → softmax：10 个 probabilities
  → argmax：选出 prediction
```

右侧的小图就是 preprocessing 后送给 model 的 image。画板进行 inference，不执行 `backward()` 或 `optimizer.step()`；画很多次也不会改变 parameters。

页面的 Probability 是这一张 image 的 model output，Test accuracy 是独立 test dataset 上的整体结果。自己的笔迹仍可能得到错误 prediction。

以后你再次 training 并保存新的 `mnist_linear_weights.pt`，停止并重新启动画板，才会 load 新的 weights。旧的 evaluation 记录如果不再匹配，页面会显示 `Test accuracy 未载入`。

`draw.py` 默认使用你的 Linear weights，端口为 `7861`。原来的 `app.py` 默认使用 `artifacts/mnist_cnn.pt`，端口为 `7860`；页面会显示各自实际加载的 model 信息。
