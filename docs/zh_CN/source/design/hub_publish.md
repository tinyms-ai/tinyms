# TinyMS Hub预训练模型贡献指南

## 标准规范

### UID

模型uid是一串自定义的统一标识符，并且按照`模型发布者/tinyms版本号/yaml文件名称`的规则对uid进行约束，其中`yaml文件名称`遵从了`模型名称_模型版本号_数据集名称`的命名规则。由于uid可以唯一标识模型的分类，因此其可以作为`hub.load`接口的参数来进行模型的加载。

举例如下：

| UID | 资源描述路径 |
| --- | --- |
| tinyms/0.2/lenet5_v1_mnist | https://github.com/tinyms-ai/tinyms/blob/main/tinyms/hub/assets/tinyms/0.2/lenet5_v1_mnist.yaml |
| tinyms/0.2/resnet50_v1_cifar10 | https://github.com/tinyms-ai/tinyms/blob/main/tinyms/hub/assets/tinyms/0.2/resnet50_v1_cifar10.yaml |

### 资源描述目录

```shell
assets/
|-- README_CN.md
|-- README.md
|-- {model_publisher}
    |-- {tinyms-version}
        |-- {model-name_01}_{model-version}_{train-dataset_01}.yaml
        |-- {model-name_02}_{model-version}_{train-dataset_02}.yaml
```

### 资源描述属性

| 属性名称 | 数据类型 | 详细介绍 |
| :------- | :------ | :------- |
| * `model-name` | string | 模型名称的全小写简称，例如`lenet5` |
| * `backbone-name` | string | 预训练模型的骨干网络，一般用于目标检测领域 |
| * `module-type` | string | 用于表示预训练模型的使用场景，当前只支持`audio`、`cv`、`nlp`、`recommend`以及`others` |
| * `fine-tunable` | bool | 用于表示该模型是否可用于迁移学习场景 |
| * `input-shape` | list[int] | `input-shape`是一个包含了`[C, H, W]`三个元素的列表，用于描述输入数据的形状信息，例如`[1, 32, 32]`或`[3, 224, 224]` |
| * `model-version` | string | 预训练模型的版本号 |
| * `train-dataset` | string | `train-dataset`可用来帮助用户来更好地理解和学习模型的使用场景，当前只支持`mnist`、`cifar10`、`mushroom`以及`voc2007` |
| `train-backend` | string | 用于表示模型训练过程中指定的后端设备 |
| `accuracy` | float | 模型评估的精度（0 < x < 1） |
| * `author` | string | 模型发布者的名称 |
| * `update-time` | datetime | 最近更新模型的时间（包括元数据更新） |
| * `user-id` | string | 用于认证的用户ID信息 |
| * `used-for` | string | 预训练模型的使用场景，当前只支持`inference`以及`transfer-learning` |
| * `infer-backend` | list[string] | 用于模型推理支持的后端 |
| * `tinyms-version` | string | TinyMS的版本号，例如`v0.2` |
| `asset` | dict | 资源信息，包括资源类型和下载链接 |
| * `file-format` | string | 预训练模型的资源类型，当前支持`ckpt`、`mindir`以及`onnx` |
| * `asset-link` | string | 预训练模型的资源下载链接 |
| * `asset-sha256` | string | 预训练模型文件的sha256码，用于检测资源是否损坏 |
| * `license` | string | 预训练模型的许可证类型 |
| * `summary` | string | 预训练模型的简短描述 |

> **注意**：上表中带`*`前缀的属性名称意味着必选。

如下提供一个基于`Mnist`数据集训练的`LeNet5`模型规范的参考实现：

```yaml
model-name: lenet5
backbone-name: lenet5
module-type: cv-classification
fine-tunable: False
input-shape: [1, 32, 32]
model-version: v1
train-dataset: mnist
train-backend: CPU
accuracy: 0.981
author: TinyMS team
update-time: 2021-05-07
user-id: TinyMS
used-for: inference
infer-backend:
  - CPU
tinyms-version: 0.2
asset:
  file-format: ckpt
  asset-link: https://tinyms-hub.obs.cn-north-4.myhuaweicloud.com/tinyms/0.2/lenet5_v1_mnist/lenet5.ckpt
  asset-sha256: b0f748227734236960b97750665cab696d60d88cb7436743b1d8a9f431ff85f1
license: Apache-2.0
summary: LeNet5 model used to classify the 10 classes of Mnist dataset.
```

## 如何发布

预训练模型发布的流程如下：

1. 创建模型（包括权重文件和网络脚本）、检查精度，然后上传至可供下载的地方；
2. 创建模型资源yaml文件，资源属性详见[上文](#资源描述属性)；
3. 使用`tools/asset_validator.py`脚本进行验证，用来核实yaml文件是否符合要求；
4. 创建一个发布请求，并通过Pull Request发布到[TinyMS社区](https://github.com/tinyms-ai/tinyms/pulls)上去。

## 许可证

所有发布到TinyMS Hub的模型都使用了[Apache 2.0 License](https://github.com/tinyms-ai/tinyms/blob/main/LICENSE)。
