# TinyMS Hub Assets Contributing Guidelines

## Specification

### UID

Model uid is alphanumeric token, in the pattern of `publisher/tinyms_version/yaml_file_name`, in which `yaml_file_name` is consisted of `[model_name]_[model_version]_[dataset_name]`. Which could be the input of the `hub.load` interface.

See example below:

| UID | Asset Url |
| --- | --- |
| tinyms/0.2/lenet5_v1_mnist | https://github.com/tinyms-ai/tinyms/blob/main/tinyms/hub/assets/tinyms/0.2/lenet5_v1_mnist.yaml |
| tinyms/0.2/resnet50_v1_cifar10 | https://github.com/tinyms-ai/tinyms/blob/main/tinyms/hub/assets/tinyms/0.2/resnet50_v1_cifar10.yaml |

### Assets folder structure

```shell
assets/
|-- README.md
|-- {model_publisher}
    |-- {tinyms-version}
        |-- {model-name_01}_{model-version}_{train-dataset_01}.yaml
        |-- {model-name_02}_{model-version}_{train-dataset_02}.yaml
```
### Asset attributes

| Attribute Name | Data Type | Introduction |
| :------------- | :-------- | :----------- |
| * `model-name` | string | The model name for short with all lower letters, such as `lenet5`. |
| * `backbone-name` | string | The backbone name of pretrained model, usually work in object detection scenario. |
| * `module-type` | string | Specify the usage for pretrained model, only support `audio`, `cv`, `nlp`, `recommend` and `others`. |
| * `fine-tunable` | bool | Specify if the model can be used for transfer learning. |
| * `input-shape` | list[int] | The input-shape is a list with three elements of `[C, H, W]`, such as `[1, 32, 32]` and `[3, 224, 224]`. |
| * `model-version` | string | Specify the version of pretrained model. |
| * `train-dataset` | string | Train dataset is REQUIRED for users to better learn the scenarios, currently only support `mnist`, `cifar10`, `mushroom` and `voc2007`. |
| `train-backend` | string | Specify the backend name when training the model. |
| `accuracy` | float | The accuracy of model evaluation with between zero and one. |
| * `author` | string | The author name of model publisher. |
| * `update-time` | datetime | The datetime of latest update. |
| * `user-id` | string | The user ID for identity. |
| * `used-for` | string | The usage of pretrained model, currently only support `inference` and `transfer-learning`. |
| * `infer-backend` | list[string] | The list of backend names supported for model inference. |
| * `tinyms-version` | string | The version of TinyMS with example of `0.2`. |
| `asset` | dict | The asset information shown with asset type and links. |
| * `file-format` | string | The asset format of pretrained model, only support `ckpt`, `mindir` and `onnx`. |
| * `asset-link` | string | The asset links of pretrained model provided for downloading. |
| * `asset-sha256` | string | The asset sha256 code provided to check if the asset corrupted. |
| * `license` | string | The license type of pretrained model. |
| * `summary` | string | The short summary of pretrained model. |

> **NOTICE:** The attribute name with `*` prefix is REQUIRED.

A referenced implementation of `LeNet5` model with `Mnist` dataset would be below:

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

## How to publish

The full process of publishing consists of:

1. Creating the model, verify the accuracy and upload to a place which can be accessed.
2. Create the model asset yaml file with [Asset attributes](#asset-attributes).
3. Using the check script in ``tools/asset_validator.py`` to self-checking the Yaml file's pattern is qualified.
4. Creating a publishing request by pulling a request to [TinyMS Community](https://github.com/tinyms-ai/tinyms/pulls).

## License

All the Model committed to the TinyMS Hub should use the [Apache 2.0 License](../../../LICENSE).
