# TinyMS

TinyMS is an Easy-to-Use deep learning development toolkit.

> **注意：** TinyMS当前是基于`v1.1.1`版本的[MindSpore](https://github.com/mindspore-ai/mindspore/tree/r1.1.1)开发完成的。

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
<td style="text-align:center"> TinyMS </td> <td style="text-align:center"> MindSpore </td>
</tr>
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
<td>

```python
import mindspore.dataset as ds
import mindspore.dataset.vision.c_transforms as CV
import mindspore.dataset.transforms.c_transforms as C
from mindspore.dataset.vision import Inter
from mindspore.common import dtype as mstype

# 不支持直接下载Mnist数据集
mnist_ds = ds.MnistDataset(data_path, shuffle=True)
# define map operations
resize_op = CV.Resize((32, 32), interpolation=Inter.LINEAR)
rescale_nml_op = CV.Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081)
rescale_op = CV.Rescale(1.0 / 255.0, 0.0)
hwc2chw_op = CV.HWC2CHW()
c_trans = [resize_op, rescale_op, rescale_nml_op, hwc2chw_op]
# apply map operations on dataset
mnist_ds = mnist_ds.map(operations=C.TypeCast(mstype.int32), input_columns="label")
mnist_ds = mnist_ds.map(operations=c_trans, input_columns="image")

mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
mnist_ds = mnist_ds.repeat(repeat_size)
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

### 模型推理/数据后处理

<table>
<tr>
<td style="text-align:center"> TinyMS </td> <td style="text-align:center"> MindSpore </td>
</tr>
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
<td>

```python
from mindspore import Model
from mindspore.train.serialization import load_checkpoint

# 不支持对单张图片进行预处理
# 需要用户自行构建网络
load_checkpoint('./checkpoint_lenet.ckpt', net=net)
model = Model(net)
res = model.predict(img).asnumpy()
# 不支持对推理结果进行后处理
```

</td>
</tr>
</table>

## License

[Apache License 2.0](./LICENSE)
