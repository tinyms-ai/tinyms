# TinyMS

TinyMS is an Easy-to-Use deep learning development toolkit.

> **注意：** TinyMS当前是基于`v1.1`版本的[MindSpore](https://github.com/mindspore-ai/mindspore/tree/r1.1)开发完成的。

## Codebase

| 模块名称 | 功能介绍 | 样例代码 |
| :------ | :------- | :------ |
| data | 数据集一键下载和基础处理 | `from tinyms.data import MnistDataset, download_dataset` |
| model | Model高阶API以及预置网络 | `from tinyms.model import Model, lenet5` |
| serving | 模型部署模块 | `from tinyms.serving import predict` |
| vision | CV领域相关的数据处理 | `from tinyms.vision import Resize, Rescale` |
| callbacks | 模型训练过程的回调处理 | `from tinyms.callbacks import ModelCheckpoint` |
| common | 基础组件，包括Tensor、numpy风格的函数 | `from tinyms import Tensor, array` |
| context | 全局上下文 | `from tinyms import context` |
| initializers | 算子权重初始化 | `from tinyms.initializers import Normal` |
| layers | 构建网络的算子清单 | `from tinyms.layers import Layer, Conv2d` |
| losses | 模型训练的损失函数 | `from tinyms.losses import SoftmaxCrossEntropyWithLogits` |
| metrics | 模型验证的指标收集 | `from tinyms.metrics import Accuracy` |
| optimizers | 模型训练的优化器 | `from tinyms.optimizers import Momentum` |
| primitives | 基础算子 | `from tinyms.primitives import TensorAdd` |

## TinyMS vs MindSpore（以LeNet5为例）

### 数据处理

<table>
<tr>
<td style="text-align:center"> TinyMS </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td>

```python
from tinyms.data import MnistDataset, download_dataset
from tinyms.data.transforms import TypeCast
from tinyms.vision import Inter, Resize, Rescale, HWC2CHW
```

</td>
<td>

```python
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
```

</td>
</tr>
</table>

### 网络构建

<table>
<tr>
<td style="text-align:center"> TinyMS </td> <td style="text-align:center"> MindSpore </td>
</tr>
<tr>
<td>

```python
from tinyms.model import lenet5

net = lenet5(class_num=10)
```

</td>
<td>

```python
import mindspore.nn as nn

class LeNet5(nn.Cell):
    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = nn.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = nn.Dense(84, num_class, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = LeNet(class_num=10)
```

</td>
</tr>
</table>

### 模型训练/验证

<table>
<tr>
<td style="text-align:center"> TinyMS </td> <td style="text-align:center"> MindSpore </td>
</tr>
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
<td>

```python
from mindspore import Model
from mindspore.train.serialization import load_checkpoint, save_checkpoint

model = Model(net, loss_fn=net_loss, optimizer=net_opt)
model.train(epoch_size, train_dataset)
# 不支持直接保存checkpoint
···
load_checkpoint('./checkpoint_lenet.ckpt', net=net)
model = Model(net, metrics=net_metrics)
model.eval(eval_dataset)
```

</td>
</tr>
</table>

## License

[Apache License 2.0](./LICENSE)
