<p align="center"><img src="https://github.com/tinyms-ai/tinyms/raw/main/docs/pic/tinyms-logo.png" alt="TinyMS logo" width="300"/></p>

# TinyMS

[![Pypi](https://img.shields.io/pypi/v/tinyms.svg)](https://pypi.org/project/tinyms)
[![Python](https://img.shields.io/pypi/pyversions/tinyms.svg)](https://pypi.org/project/tinyms)
[![Downloads](https://pepy.tech/badge/tinyms)](https://pepy.tech/project/tinyms)
[![Build Status](https://travis-ci.org/tinyms-ai/tinyms.svg?branch=main)](https://travis-ci.org/tinyms-ai/tinyms)
[![Documentation Status](https://readthedocs.org/projects/tinyms/badge/?versoin=latest)](https://readthedocs.org/projects/tinyms)
[![Releases](https://img.shields.io/github/release/tinyms-ai/tinyms/all.svg?style=flat-square)](https://github.com/tinyms-ai/tinyms/releases)
[![LICENSE](https://img.shields.io/github/license/tinyms-ai/tinyms.svg?style=flat-square)](https://github.com/tinyms-ai/tinyms/blob/main/LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

English | [查看中文](./README_CN.md)

TinyMS is an Easy-to-Use deep learning framework development toolkit based on [MindSpore](https://www.mindspore.cn/en/), designed to provide quick-start guidelines for machine learning beginners.

## Installation

Please checkout the [install document](https://tinyms.readthedocs.io/en/latest/quickstart/install.html) to quickly install or upgrade TinyMS project.

## Quick start

Have no idea what to do with TinyMS❓ See the [Quick Start](https://tinyms.readthedocs.io/en/latest/quickstart/quickstart_in_one_minute.html) to implement the image classification application in one minutes❗

Besides, here are some use cases listed to demonstrate how TinyMS simplifies the code flow for users.

### Data loading and preprocess

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

### Network construction

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

### Model train/evaluation

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

### Model prediction

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

## API documentation

If you are interested in learning TinyMS API, please find TinyMS Python API in [API Documentation](https://tinyms.readthedocs.io/en/latest/tinyms/tinyms.html).

## Community

For any developers who are not familiar with how TinyMS community works, please find the [Contributing Guidelines](https://tinyms.readthedocs.io/en/latest/community/contributing.html) to get started.

## Release Notes

The release notes, see our [RELEASE](https://github.com/tinyms-ai/tinyms/blob/main/RELEASE.md).

## License

[Apache License 2.0](https://github.com/tinyms-ai/tinyms/blob/main/LICENSE)
