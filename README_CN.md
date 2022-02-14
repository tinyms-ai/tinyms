<p align="center"><img src="https://github.com/tinyms-ai/tinyms/raw/main/docs/pic/tinyms-logo.png" alt="TinyMS logo" width="300"/></p>

# TinyMS

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/tinyms.svg)](https://pypi.org/project/tinyms)
[![PyPI](https://badge.fury.io/py/tinyms.svg)](https://badge.fury.io/py/tinyms)
[![Downloads](https://pepy.tech/badge/tinyms)](https://pepy.tech/project/tinyms)
[![DockerHub](https://img.shields.io/docker/pulls/tinyms/tinyms.svg)](https://hub.docker.com/r/tinyms/tinyms)
[![Build Status](https://github.com/tinyms-ai/tinyms/actions/workflows/install_and_test.yml/badge.svg?branch=main)](https://github.com/tinyms-ai/tinyms/actions/workflows/install_and_test.yml)
[![Documentation Status](https://readthedocs.org/projects/tinyms/badge/?versoin=latest)](https://readthedocs.org/projects/tinyms)
[![Releases](https://img.shields.io/github/release/tinyms-ai/tinyms/all.svg?style=flat-square)](https://github.com/tinyms-ai/tinyms/releases)
[![LICENSE](https://img.shields.io/github/license/tinyms-ai/tinyms.svg?style=flat-square)](https://github.com/tinyms-ai/tinyms/blob/main/LICENSE)
[![Slack](https://img.shields.io/badge/slack-chat-green.svg?logo=slack)](https://join.slack.com/t/mindspore/shared_invite/zt-dgk65rli-3ex4xvS4wHX7UDmsQmfu8w)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](https://github.com/tinyms-ai/tinyms/pulls)

中文 | [View English](./README.md)

TinyMS是基于[MindSpore AI框架](https://www.mindspore.cn/)开发，面向上层用户的一个高级API开发库，目的是让小白用户能够更加轻松地上手开发深度学习应用。

<p align="center"><img src="docs/pic/tinyms-architecture.png" alt="TinyMS Architecture" width="800" /></p>

## 安装

| 版本分发工具 | 版本 | 安装命令 |
| :---------- | :--- | :------ |
| PyPI | x.y.z | `pip install tinyms==x.y.z` |
|  | latest | `pip install git+https://github.com/tinyms-ai/tinyms.git` |
| Docker | x.y.z | `docker pull tinyms==x.y.z` |
|  | latest | - |

> **注意：** 表格中的`x.y.z`需替换成具体的版本号。

欢迎查阅[安装文档](https://tinyms.readthedocs.io/zh_CN/latest/quickstart/install.html)实现一键安装TinyMS。

## 快速上手

不知道用TinyMS做什么❓ 通过[快速上手指南](https://tinyms.readthedocs.io/zh_CN/latest/quickstart/quickstart_in_one_minute.html)，您可以在一分钟内快速开发一个图形分类应用❗

当然，我们在这里也为您提供了一些TinyMS常用的使用场景，您可以快速体验到TinyMS开发的简易和流畅性。

### 数据加载/处理

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

### 模型推理部署

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

## API文档

如果您想了解TinyMS的Python API接口，请访问[TinyMS API文档](https://tinyms.readthedocs.io/zh_CN/latest/tinyms/tinyms.html)。

## 教程

TinyMS配套了完善的教程，适合纯小白入手。

| 集数 | 标题 | 主题内容 | 学习文档 | 状态 | 更新时间 |
| ---- | ---- | ------- | -------- | ---- | -------- |
| EP01 | [最适合小白的深度学习入门课程来了!](https://www.bilibili.com/video/BV1MB4y1P79S) | 老师简介+深度学习科普+课程介绍 | 无 | 已发布 | 2021.3.30 |
| EP02 | [计算机是如何识别图像的?](https://www.bilibili.com/video/BV18v41187fX) | 计算机识别图像科普+TinyMS安装（Ubuntu、Win10、Docker安装） | [TinyMS快速安装](https://tinyms.readthedocs.io/zh_CN/latest/quickstart/install.html) | 已发布 | 2021.3.31 |
| EP03 | [30min速成Shell脚本命令](https://www.bilibili.com/video/BV1vy4y1b7jh) | shell基本命令、变量与传参、数组、运算符、test命令、使用函数、输入输出重定向、流程控制函数以及常见的文件操作命令 | [Shell脚本30min速成指南](https://github.com/tinyms-ai/tinyms/blob/main/tutorials/EP03/30min速成Shell脚本.md) | 已发布 | 2021.4.1 |
| EP04 | [30min速成Python指南（上）](https://www.bilibili.com/video/BV1Tp4y1b7UG?spm_id_from=333.999.0.0) | Python安装、基础语法、原始数据类型及运算符介绍 | [Python30min速成指南(上)](https://github.com/tinyms-ai/tinyms/blob/main/tutorials/EP04/Quickstart-for-Python-in-30-minutes.md) | 已发布 | 2021.4.23  |
| EP05 | [30min速成Python指南（下）](https://www.bilibili.com/video/BV1XS4y1Z7yp?spm_id_from=333.999.0.0) | Python条件语句、循环语句、迭代器、生成器、函数、类、模块、高级用法、及深度学习中常用的Python库 | [Python30min速成指南(下)](https://github.com/tinyms-ai/tinyms/blob/main/tutorials/EP04/Quickstart-for-Python-in-30-minutes.md) | 已发布 | 2022.1.10  |
## 社区

欢迎加入TinyMS社区进行贡献，如果您还不太清楚TinyMS社区的运作流程，可通过[贡献指南](https://tinyms.readthedocs.io/zh_CN/latest/community/contributing.html)的学习快速上手社区贡献。

## 版本说明

版本说明请参阅[RELEASE](https://github.com/tinyms-ai/tinyms/blob/main/RELEASE.md)。

## 许可证

[Apache License 2.0](https://github.com/tinyms-ai/tinyms/blob/main/LICENSE)
