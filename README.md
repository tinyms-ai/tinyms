# TinyMS

[![Build Status](https://travis-ci.org/tinyms-ai/tinyms.svg?branch=main)](https://travis-ci.org/tinyms-ai/tinyms)
[![Documentation Status](https://readthedocs.org/projects/tinyms/badge/?versoin=latest)](https://readthedocs.org/projects/tinyms)
[![Releases](https://img.shields.io/github/release/tinyms-ai/tinyms/all.svg?style=flat-square)](https://github.com/tinyms-ai/tinyms/releases)
[![LICENSE](https://img.shields.io/github/license/tinyms-ai/tinyms.svg?style=flat-square)](https://github.com/tinyms-ai/tinyms/blob/main/LICENSE)

TinyMS is an Easy-to-Use deep learning development toolkit.

> **注意：** TinyMS当前是基于`v1.1.1`版本的[MindSpore](https://github.com/mindspore-ai/mindspore/tree/v1.1.1)开发完成的。

## Codebase

| 模块名称 | 功能介绍 | 样例代码 |
| :------ | :------- | :------ |
| data | 数据集一键下载和加载 | `from tinyms.data import MnistDataset, download_dataset` |
| model | Model高阶API以及预置网络 | `from tinyms.model import Model, lenet5` |
| serving | 模型部署模块 | `from tinyms.serving import predict` |
| vision | CV领域相关的数据处理 | `from tinyms.vision import mnist_transform, Resize` |
| callbacks | 模型训练过程的回调处理 | `from tinyms.callbacks import ModelCheckpoint` |
| common | 基础组件，包括Tensor、numpy风格的函数 | `from tinyms import Tensor, array` |
| context | 全局上下文 | `from tinyms import context` |
| initializers | 算子权重初始化 | `from tinyms.initializers import Normal` |
| layers | 构建网络的算子清单 | `from tinyms.layers import Layer, Conv2d` |
| losses | 模型训练的损失函数 | `from tinyms.losses import SoftmaxCrossEntropyWithLogits` |
| metrics | 模型验证的指标收集 | `from tinyms.metrics import Accuracy` |
| optimizers | 模型训练的优化器 | `from tinyms.optimizers import Momentum` |
| primitives | 基础算子 | `from tinyms.primitives import Add, tensor_add` |

## TinyMS guideline（以LeNet5为例）

### 数据处理

<table>
<tr>
<td>

```python
from tinyms.data import MnistDataset, download_dataset
from tinyms.vision import mnist_transform

data_path = download_dataset('mnist')
mnist_ds = MnistDataset(data_path, shuffle=True)
mnist_ds = mnist_transform.apply_ds(mnist_ds)
```

</td>
</tr>
</table>

### 网络构建

<table>
<tr>
<td>

```python
from tinyms.model import lenet5

net = lenet5(class_num=10)
```

</td>
</tr>
</table>

### 模型训练/验证

<table>
<tr>
<td>

```python
from tinyms.model import Model

model = Model(net)
model.compile(loss_fn=net_loss, optimizer=net_opt, metrics=net_metrics)
model.train(epoch_size, train_dataset)
model.save_checkpoint('./checkpoint_lenet.ckpt')
···
model.load_checkpoint('./checkpoint_lenet.ckpt')
model.eval(eval_dataset)
```

</td>
</tr>
</table>

### 模型推理/数据后处理

<table>
<tr>
<td>

```python
from PIL import Image
import tinyms as ts
from tinyms.model import Model, lenet5
from tinyms.vision import mnist_transform

img = Image.open(img_path)
img = mnist_transform(img)

net = lenet5(class_num=10)
model = Model(net)
model.load_checkpoint('./checkpoint_lenet.ckpt')

input = ts.expand_dims(ts.array(img), 0)
res = model.predict(input).asnumpy()
print("The label is:", mnist_transform.postprocess(res))
```

</td>
</tr>
</table>

## License

This work is licensed under [Apache License 2.0](LICENSE).
