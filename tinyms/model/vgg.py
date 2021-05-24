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

from tinyms import layers


class VGG(layers.Layer):

    def __init__(self, features, class_num=10):
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = layers.AvgPool2d(kernel_size=2,stride=2)
        self.flatten = layers.Flatten()
        self.classifier = layers.Sequential(
            layers.Dense(512 * 7 * 7, 4096),
            layers.ReLU(),
            layers.Dropout(),
            layers.Dense(4096, 4096),
            layers.ReLU(),
            layers.Dropout(),
            layers.Dense(4096, class_num),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [layers.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = layers.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, layers.BatchNorm2d(v), layers.ReLU()]
            else:
                layers += [conv2d, layers.ReLU()]
            in_channels = v
    return layers.Sequential(*layers)


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
