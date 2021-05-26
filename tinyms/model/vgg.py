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


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channel, out_channel,
                         kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)

class VGG(layers.Layer):
    """
    Get VGG neural network.

    Args:
        features (layers.Layer): Feature extractor.
        class_num (int): Class number. Default: 1000.

    Returns:
        layers.Layer, layer instance of AlexNet neural network.

    Examples:
        >>> from tinyms.model import VGG
        >>>
        >>> net = VGG(features=make_layers(cfg=cfgs['A']),class_num=1000)
    """

    def __init__(self, features, class_num=1000):
        super(VGG, self).__init__()
        self.features = features
        self.flatten = layers.Flatten()
        self.classifier = layers.SequentialLayer([
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
    Layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            Layers += [layers.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = _conv3x3(in_channels, v)
            if batch_norm:
                Layers += [conv2d, layers.BatchNorm2d(v), layers.ReLU()]
            else:
                Layers += [conv2d, layers.ReLU()]
            in_channels = v
    return layers.SequentialLayer(Layers)


cfgs = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(**kwargs):
    """
    Get vgg11 neural network.

    Args:
        class_num (int): Class number. Default: 10.
        batch_norm (bool): Whether to use BatchNormalization. Default: True

    Returns:
        layers.Layer, layer instance of vgg11 neural network.

    Examples:
        >>> from tinyms.model import vgg11
        >>>
        >>> net = vgg11(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['A'], batch_norm=kwargs.get('batch_norm', True)), class_num=kwargs.get('class_num', 10))


def vgg13(**kwargs):
    """
    Get vgg13 neural network.

    Args:
        class_num (int): Class number. Default: 10.
        batch_norm (bool): Whether to use BatchNormalization. Default: True

    Returns:
        layers.Layer, layer instance of vgg13 neural network.

    Examples:
        >>> from tinyms.model import vgg13
        >>>
        >>> net = vgg13(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['B'], batch_norm=kwargs.get('batch_norm', True)), class_num=kwargs.get('class_num', 10))


def vgg16(**kwargs):
    """
    Get vgg16 neural network.

    Args:
        class_num (int): Class number. Default: 10.
        batch_norm (bool): Whether to use BatchNormalization. Default: True

    Returns:
        layers.Layer, layer instance of vgg16 neural network.

    Examples:
        >>> from tinyms.model import vgg16
        >>>
        >>> net = vgg16(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['D'], batch_norm=kwargs.get('batch_norm', True)), class_num=kwargs.get('class_num', 10))


def vgg19(**kwargs):
    """
    Get vgg19 neural network.

    Args:
        class_num (int): Class number. Default: 10.
        batch_norm (bool): Whether to use BatchNormalization. Default: True

    Returns:
        layers.Layer, layer instance of vgg19 neural network.

    Examples:
        >>> from tinyms.model import vgg19
        >>>
        >>> net = vgg19(class_num=10)
    """
    return VGG(make_layers(cfg=cfgs['E'], batch_norm=kwargs.get('batch_norm', True)), class_num=kwargs.get('class_num', 10))
