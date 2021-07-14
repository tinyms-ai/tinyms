# TinyMS Release Notes

## 0.2.1

Released 2021-07-15.

### Major Features and Improvements

* Fix `load_checkpoint` interface bug in TinyMS 0.2.0 hub module. [#96](https://github.com/tinyms-ai/tinyms/pull/96)

## 0.2.0

Released 2021-06-07.

### Major Features and Improvements

* Add `text` module to provide the basic dataset loading and preprocessing in NLP scenarios. [#53](https://github.com/tinyms-ai/tinyms/pull/53) [#73](https://github.com/tinyms-ai/tinyms/pull/73)
* Upgrade the version of `mindspore` module dependencies from `v1.1.1` to `v1.2.0`. [#81](https://github.com/tinyms-ai/tinyms/pull/81) [#84](https://github.com/tinyms-ai/tinyms/pull/84)
* Refactor the `Client` and `Server` communication interface in `serving` module. [#76](https://github.com/tinyms-ai/tinyms/pull/76)
* Added server_path, start FlaskServer and add host and port parameters. [#77](https://github.com/tinyms-ai/tinyms/pull/77)
* Implement TinyMS `hub` module to enable loading lots of pre-trained models, incluidng `lenet5_v1`, `resnet50_v1`, `alexnet_v1`, `vgg16_v1`, `mobilenet_v2` and `ssd300_v1`. [#86](https://github.com/tinyms-ai/tinyms/pull/86) [#93](https://github.com/tinyms-ai/tinyms/pull/93)
* Publish the TinyMS Hub contributing guidelines in public to welcome pre-trained model from the comunity. [#91](https://github.com/tinyms-ai/tinyms/pull/91)
* Refactor the model network entrypoint method to provide the unified interface. [#85](https://github.com/tinyms-ai/tinyms/pull/85)

#### Model Park

* Add **5** models support: `AlexNet`, `DenseNet100`, `VGG16`, `SentimentNet`, `Bert`. [#59](https://github.com/tinyms-ai/tinyms/pull/59) [#89](https://github.com/tinyms-ai/tinyms/pull/89) [#63](https://github.com/tinyms-ai/tinyms/pull/63) [#67](https://github.com/tinyms-ai/tinyms/pull/67)

### API Change

* Refactor the `serving` entrypoint function with `Client` and `Server` class interface.

<table>
<tr>
<td style="text-align:center"> v0.1.0 </td> <td style="text-align:center"> v0.2.0 </td>
</tr>
<tr>
<td>

```python
from tinyms.serving import start_server, server_started, list_servables, predict, shutdown

start_server()
if server_started():
    list_servables()
    predict('example.jpg', 'servable_name', dataset_name='mnist')
shutdown()
```

</td>
<td>

```python
from tinyms.serving import Client, Server

server = Server()
server.start_server()
client = Client()
client.list_servables()
client.predict('example.jpg', 'servable_name', dataset_name='mnist')
server.shutdown()
```

</td>
</tr>
</table>

* Add a new interface load in model module to support load MindIR graph directly to perform model inference operation.

<table>
<tr>
<td style="text-align:center"> v0.2.0 </td>
</tr>
<tr>
<td>

```python
>>> import tinyms as ts
>>> import tinyms.layers as layers
>>> from tinyms.model import Model, load
>>>
>>> net = layers.Conv2d(1, 1, kernel_size=3)
>>> model = Model(net)
>>> input = ts.ones([1, 1, 3, 3])
>>> model.export(input, "net", file_format="MINDIR")
...
>>> net = load("net.mindir")
>>> print(net(input))
[[[[ 0.02548009  0.04010789  0.03120251]
    [ 0.00268656  0.02353744  0.03807815]
    [-0.00896441 -0.00303641  0.01502199]]]]
```

</td>
</tr>
</table>

* Add `hub.load` method to easily load pretrained model and apply model evaluation and inference operation.


<table>
<tr>
<td style="text-align:center"> v0.2.0 </td>
</tr>
<tr>
<td>

```python
from PIL import Image
from tinyms import hub
from tinyms.vision import mnist_transform
from tinyms.model import Model

img = Image.open(img_path)
img = mnist_transform(img)

# load LeNet5 pretrained model
net= hub.load('tinyms/0.2/lenet5_v1_mnist', class_num=10)
model = Model(net)

res = model.predict(ts.expand_dims(ts.array(img), 0)).asnumpy()
print("The label is:", mnist_transform.postprocess(res))
```

</td>
</tr>
</table>

For the detailed API changes, please find TinyMS Python API in [API Documentation](https://tinyms.readthedocs.io/en/latest/tinyms/tinyms.html).

#### Backwards Incompatible Change

None

### Bug fixes

* Fix some bugs when serving in Windows operating system. [#74](https://github.com/tinyms-ai/tinyms/pull/74)
* Set `batch_norm` as `True` by default in VGG16 to fix the converge problem of accuracy. [#90](https://github.com/tinyms-ai/tinyms/pull/90)

### Contributors

Great thanks go to these wonderful people:

[@zjuter0126](https://github.com/zjuter0126), [@Mickls](https://github.com/Mickls), [@leonwanghui](https://github.com/leonwanghui), [@hannibalhuang](https://github.com/hannibalhuang), [@hellowaywewe](https://github.com/hellowaywewe), [@huxiaoman7](https://github.com/huxiaoman7)


## 0.1.0

Released 2021-03-28.

### Major Features and Improvements

* Design the overall framework of TinyMS development toolkit. [#3](https://github.com/tinyms-ai/tinyms/pull/3) [#5](https://github.com/tinyms-ai/tinyms/pull/5) [#12](https://github.com/tinyms-ai/tinyms/pull/12) [#13](https://github.com/tinyms-ai/tinyms/pull/13)
* Support install TinyMS binary in `Linux Ubuntu 18.04` and `Window 10` environment, also provide TinyMS docker image to users. [#2](https://github.com/tinyms-ai/tinyms/pull/2) [#45](https://github.com/tinyms-ai/tinyms/pull/45)
* Enable document auto-generation using Sphinx. [#35](https://github.com/tinyms-ai/tinyms/pull/35)
* Provide several end to end model development and deployment [tutorials](https://tinyms.readthedocs.io/en/latest/quickstart/quickstart_in_one_minute.html) for machine learning beginners. [#11](https://github.com/tinyms-ai/tinyms/pull/11) [#24](https://github.com/tinyms-ai/tinyms/pull/24) [#26](https://github.com/tinyms-ai/tinyms/pull/26) [#34](https://github.com/tinyms-ai/tinyms/pull/34)
* Set up the initial CI pipeline (including cla-assistant, GitHub Actions, readthedocs) for TinyMS project. [#1](https://github.com/tinyms-ai/tinyms/pull/1) [#49](https://github.com/tinyms-ai/tinyms/pull/49) [#50](https://github.com/tinyms-ai/tinyms/pull/50)

#### Model Park

* Add **5** models support: `LeNet5`, `ResNet50`, `MobileNetV2`, `SSD300`, `CycleGAN`. [#5](https://github.com/tinyms-ai/tinyms/pull/5) [#14](https://github.com/tinyms-ai/tinyms/pull/14) [#17](https://github.com/tinyms-ai/tinyms/pull/17) [#32](https://github.com/tinyms-ai/tinyms/pull/32)

### API Change

There is no API change for the first version of TinyMS. Please find TinyMS Python API in [API Documentation](https://tinyms.readthedocs.io/en/latest/tinyms/tinyms.html).

#### Backwards Incompatible Change

None

### Bug fixes

None

### Contributors

Great thanks go to these wonderful people:

[@leonwanghui](https://github.com/leonwanghui), [@lyd911](https://github.com/lyd911), [@hannibalhuang](https://github.com/hannibalhuang), [@hellowaywewe](https://github.com/hellowaywewe), [@Yikun](https://github.com/Yikun), [@huxiaoman7](https://github.com/huxiaoman7)
