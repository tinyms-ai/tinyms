# Design Concept

## Background

In recent years, with the vigorous development of AI technology, many excellent deep learning frameworks have emerged in the industry: such as TensorFlow, PyTorch, MindSpore, etc. These frameworks have been proposed to meet the needs of various industries for AI model/application development. However, due to the high threshold of AI technology, the API of the native framework cannot meet the needs of all users/developers. This also leads to the emergence of many high-level APIs customized and developed for AI frameworks in the later period: such as Keras (TF), fastai (PyTorch) etc. Then the TinyMS introduced to you today is a high-level API module specially proposed for optimizing MindSpore.

Users who know MindSpore may be puzzled: The MindSpore project itself has defined high-level APIs, mid-level APIs, and low-level APIs. Why do we need to define a set of high-level APIs? What kind of problems does this high-level API solve? Indeed, if you compare MindSpore with TF+Keras, MindSpore's high-level and mid-level Python APIs have already implemented most of the functions of Keras, and there is no need to package an additional layer of Keras-like APIs; fastai itself is designed for PyTorch flexibility And Shengsheng's design concept of rapid development and verification is also difficult to apply to MindSpore. But if we compare Keras and fastai horizontally, their original design intentions are basically based on the characteristics or deficiencies of the underlying framework for further optimization, then if we think about the unique characteristics of MindSpore, that is, the whole scene of the end-side cloud is unified. AI framework, so the consistent development experience of the end-side cloud must be one of TinyMS's design goals.

In addition to the consistent development experience of the end-side cloud, another major goal of TinyMS is to further reduce the technical threshold of deep learning, so that more "little white" users can quickly get started with the AI ​​application development experience. With TinyMS, we hope to be able to do so. The ultimate experience to the following two points:

* Get started with AI application development in one minute
* Master the free switching of AI models and data sets within one hour

## Architecture

For most projects, architecture (module) design can be said to be the soul and core of the entire project, and a good architecture design can play a multiplier role in terms of user experience and scalability. As a high-level API project, the architecture design of TinyMS hopes to achieve the following points:

* Divide and design modules from the perspective of users
* Do a good job of collaboration with MindSpore: TinyMS focuses on user developer experience, MindSpore is responsible for function implementation
* Decoupling between modules

### Use-case analysis

For most AI model application development scenarios, they can basically be summarized into the following steps:

* **Data acquisition**: including data set download, decompression, loading, etc.
* **Data processing**: In order to achieve better performance of the model, data preprocessing (enhancement) operations are generally performed on the original data
* **Model Construction**: In addition to the construction of the main body of the network, it also includes the definition of Loss loss function, Optimizer optimizer, etc.
* **Model training**: Responsible for the process of model training, including the definition of callbacks
* **Accuracy Verification**: Responsible for the process of model accuracy verification, including the definition of metrics
* **Model deployment**: Provide AI model application services by building a server

### Module design

In response to the steps mentioned in the above scenario, currently TinyMS has created the following modules:

| Name | Introduction | Example Code |
| :--- | :----------- | :----------- |
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

## Implementation

Seeing this, I believe that everyone has a preliminary understanding of the overall architecture of TinyS, and then we will explain the design and implementation ideas of each module one by one.

### Data loading (*data*)

The data loading module is mainly divided into two parts: data set download and loading. Through TinyMS's data loading API, only two lines of code can complete the process of downloading, decompressing, formatting, and loading common data sets.

Most AI frameworks do not provide an interface for data set downloading. Users need to prepare the data set in advance, and at the same time, adjust the format (training/validation data set division, etc.) according to the data loading API format provided by the framework itself. This is for the beginning of AI. For scholars, the threshold is still quite high. Based on this problem, TinyMS provides the `download_dataset` interface, which supports users to complete the download, decompression and format adjustment operations of the data set with one click; use [Mnist](http://yann.lecun.com/exdb/mnist/) dataset as an example:
```python
from tinyms.data import download_dataset

mnist_path = download_dataset('mnist', local_path='./')
```

For data loading operations, TinyMS completely inherits MindSpore's native [Data Loading API](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.dataset.html), so Users can use the xxxDataset interface to instantiate different data sets very conveniently; with [MnistDataset](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/dataset/mindspore.dataset.MnistDataset.html#mindspore.dataset.MnistDataset) as an example:
```python
from tinyms.data import MnistDataset

mnist_ds = MnistDataset(mnist_path, shuffle=True)
```

### Data preprocessing (*vision*)

Usually when constructing AI model development and application, data processing is the first big challenge we face: insufficient data, heavy manual labeling workload, irregular data format and other issues may cause the network accuracy after training to not meet the standard, so it is absolutely necessary. Most AI frameworks provide related modules for data processing. Take MindSpore as an example. MindSpore currently provides data processing functions for common scenarios such as CV and NLP (for relevant interface definitions, please refer to [`mindspore.dataset.vision`](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.dataset.vision.html) and [`mindspore.dataset.text`](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.dataset.text.html)), the user can directly call the preset data processing operator to process pictures or text, and then construct a data processing pipeline to efficiently parallelize massive data (see [here](https://www.mindspore.cn/news/newschildren?id=399)).

Although the functions and performance of MindSpore in the data processing part have been excellent, we found that the semantics of operators is still too low-level for junior deep learning developers. For these user groups, learn the operator expression and pipeline of data processing. The cost of construction is still high, so TinyMS has made further abstraction and encapsulation on the basis of MindSpore, and directly corresponds to the processing of the data set itself through the `DatasetTransform` interface, allowing users to realize a single piece of data or the entire data set with one line of code Preprocessing operation; take `MnistTransform` as an example:
```python
from PIL import Image
from tinyms.vision import mnist_transform

# Preprocessing a single one picture
img = mnist_transform(Image.open('picture.jpg'))
# Apply preprocessing to MnistDataset class instance
mnist_ds = mnist_transform.apply_ds(mnist_ds)
```

### Model construction (*model*)

As the core of deep learning model development, network structure is a skill that AI practitioners must master: from CNN in the CV field, to RNN and Transformer in the NLP field, to GNN, GAN, etc., which are very popular in the academic circle, every time Changes in the network structure will cause an uproar worldwide. For the AI ​​framework, its main responsibility is to provide complete operator expressions to build different network structures. Therefore, the interfaces at the AI ​​framework level focus more on functional completeness and flexibility; in addition to cultivating user ecology, Generally speaking, the AI ​​framework will specifically provide ModelZoo components to meet the needs of users and commercial landing. So for the TinyMS high-level API, what special functions can it provide in terms of network construction?

If you have checked the [ModelZoo](https://gitee.com/mindspore/mindspore/tree/r1.1/model_zoo) component provided by the MindSpore community, you will find that many network model construction scripts have been natively provided to upper-level users, So for TinyMS, you only need to encapsulate the relevant network call API on the native script; take the `LeNet5` network as an example:
```python
from tinyms.model import lenet5

net = lenet5(class_num=10)
```

In addition to encapsulating the commonly used network structure, TinyMS also provides a `Model` high-level API interface (based on [MindSpore Model](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.html#mindspore.Model) interface package), by drawing on the design idea of ​​[Keras Model](https://keras.io/api/models/model/#model-class) interface, it not only improves the original API interface The functions of, also provide a consistent development experience for Keras users:
```python
from tinyms.model import Model

model = Model(net)
model.compile(loss_fn=net_loss, optimizer=net_opt)
```

### Model training (*losses*, *optimizers*, *callbacks*)

For the model training phase, the most important factors are the definition of loss function, optimizer, and callback function. The basic definitions of these three are not repeated here. For beginners, it is not difficult to understand the basic principles of loss functions and optimizers, but a strong mathematical background is required to understand the principles of implementation. Therefore, the TinyMS high-level API encapsulates the loss function and optimizer at the network level, so that users can complete the initialization work with one line of code whether they are training simple or complex networks; take the `LeNet5` network as an example:
```python
from tinyms.losses import SoftmaxCrossEntropyWithLogits
from tinyms.optimizers import Momentum

lr = 0.01
momentum = 0.9
net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
net_opt = Momentum(net.trainable_params(), lr, momentum)
```

Regarding the definition of callback functions, in addition to commonly used callback functions (such as `TimeMonitor`, `LossMonitor`, etc.), MindSpore itself provides [Callback](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.train.html#mindspore.train.callback.Callback) interface to facilitate user-defined callback functions. The TinyMS high-level API also provides network-level encapsulation, so that users can complete the initialization of the callback function with one line of code; take the `MobileNetV2` network as an example:
```python
from tinyms.callbacks import mobilenetv2_cb

net_cb = mobilenetv2_cb(device_target, lr, is_saving_checkpoint, save_checkpoint_epochs, step_size)
```

### Model evaluating (*metrics*)

Model accuracy verification is an indispensable process to verify whether the model accuracy is up to standard. MindSpore natively provides measurement interfaces for indicators such as `Accuracy` and `Precision` (see [here](https://www.mindspore.cn/doc/api_python/zh-CN/r1.1/mindspore/mindspore.nn.html#metrics)), while providing users with a custom measurement interface `Metric`. In terms of index measurement, TinyMS directly inherits the native MindSpore API, and users can follow the habits of MindSpore for accuracy verification:
```python
from tinyms.model import Model
from tinyms.metrics import Accuracy

model = Model(net)
model.compile(metrics={"Accuracy": Accuracy())
```

### Model deployment (*serving*)

Model deployment reasoning refers to the process of servicing pre-trained models so that they can quickly and efficiently process data input by users and obtain results. MindSpore provides the [predict](https://mindspore.cn/doc/api_python/zh-CN/r1.1/_modules/mindspore/train/model.html#Model.predict) function for reasoning, similarly, TinyMS Corresponding encapsulation is carried out for this function, and different back-end networks are connected with the same interface. In order to achieve servicing, TinyMS provides a complete set of start server (`start_server`), check backend (`list_servables`), check based on [Flask](https://flask.palletsprojects.com/en/1.1.x/) Whether to start (`server_started`) and shut down the server (`shutdown`) and other functions; Take the `LeNet5` network as an example:
```python
from tinyms.serving import start_server, predict, list_servables, server_started, shutdown

# Start prediction server
start_server()
# List all servables available
list_servables()
# Call predict interface
if server_started():
    res = predict(image_path, 'lenet5', 'mnist')
# Shutdown the prediction server
shutdown()
```
