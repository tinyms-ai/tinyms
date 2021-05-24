# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

import numpy as np

from tinyms import layers, Tensor
from tinyms.layers import BatchNorm2d, ReLU, Conv2d, MaxPool2d, \
    Flatten, SequentialLayer, Layer


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return Conv2d(in_channel, out_channel,
                         kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)

class VGG(Layer):

    def __init__(self, features, class_num=10):
        super(VGG, self).__init__()
        self.features = features
        self.flatten = Flatten()
        self.classifier = SequentialLayer([
            layers.Dense(512 * 7 * 7, 4096),
            layers.ReLU(),
            layers.Dropout(),
            layers.Dense(4096, 4096),
            layers.ReLU(),
            layers.Dropout(),
            layers.Dense(4096, class_num),
        ])

    def construct(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = _conv3x3(in_channels, v)
            if batch_norm:
                layers += [conv2d, BatchNorm2d(v), ReLU()]
            else:
                layers += [conv2d, ReLU()]
            in_channels = v
    return SequentialLayer(layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(class_num=10):
    """
    Get vgg11 neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg11 neural network.

    Examples:
        >>> from tinyms.model import vgg11
        >>>
        >>> net = vgg11(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['A'], batch_norm=False), class_num=class_num)


def vgg11_bn(class_num=10):
    """
    Get vgg11_bn neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg11_bn neural network.

    Examples:
        >>> from tinyms.model import vgg11_bn
        >>>
        >>> net = vgg11_bn(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['A'], batch_norm=True), class_num=class_num)


def vgg13(**kwargs):
    """
    Get vgg13 neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg13 neural network.

    Examples:
        >>> from tinyms.model import vgg13
        >>>
        >>> net = vgg13(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['B'], batch_norm=False), class_num=kwargs.get('class_num', 10))


def vgg13_bn(**kwargs):
    """
    Get vgg13_bn neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg13_bn neural network.

    Examples:
        >>> from tinyms.model import vgg13_bn
        >>>
        >>> net = vgg13_bn(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['B'], batch_norm=True), class_num=kwargs.get('class_num', 10))


def vgg16(**kwargs):
    """
    Get vgg16 neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg16 neural network.

    Examples:
        >>> from tinyms.model import vgg16
        >>>
        >>> net = vgg16(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['D'], batch_norm=False), class_num=kwargs.get('class_num', 10))


def vgg16_bn(**kwargs):
    """
    Get vgg16_bn neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg16_bn neural network.

    Examples:
        >>> from tinyms.model import vgg16_bn
        >>>
        >>> net = vgg16_bn(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['D'], batch_norm=True), class_num=kwargs.get('class_num', 10))


def vgg19(**kwargs):
    """
    Get vgg19 neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg19 neural network.

    Examples:
        >>> from tinyms.model import vgg19
        >>>
        >>> net = vgg19(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['E'], batch_norm=False), class_num=kwargs.get('class_num', 10))


def vgg19_bn(**kwargs):
    """
    Get vgg19_bn neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of vgg19_bn neural network.

    Examples:
        >>> from tinyms.model import vgg19_bn
        >>>
        >>> net = vgg19_bn(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['E'], batch_norm=True), class_num=kwargs.get('class_num', 10))