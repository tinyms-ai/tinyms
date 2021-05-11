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
from scipy.stats import truncnorm

import tinyms as ts
from tinyms import layers, Tensor
from tinyms.layers import ReLU, MaxPool2d, Flatten, Dropout


def _conv_variance_scaling_initializer(in_channel, out_channel, kernel_size):
    fan_in = in_channel * kernel_size * kernel_size
    scale = 1.0
    scale /= max(1., fan_in)
    stddev = (scale ** 0.5) / .87962566103423978
    mu, sigma = 0, stddev
    weight = truncnorm(-2, 2, loc=mu, scale=sigma).rvs(out_channel * in_channel * kernel_size * kernel_size)
    return ts.reshape(weight, (out_channel, in_channel, kernel_size, kernel_size))


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def _conv3x3(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 3, 3)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channel, out_channel,
                         kernel_size=3, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv1x1(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 1, 1)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channel, out_channel,
                         kernel_size=1, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv7x7(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 7, 7)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channel, out_channel,
                         kernel_size=7, stride=stride, padding=0, pad_mode='same', weight_init=weight)


def _conv11x11(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 11, 11)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channel, out_channel,
                         kernel_size=11, stride=stride, padding=2, pad_mode='pad', weight_init=weight)


def _conv5x5(in_channel, out_channel, stride=1):
    weight_shape = (out_channel, in_channel, 5, 5)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channel, out_channel,
                         kernel_size=5, stride=stride, padding=2, pad_mode='pad', weight_init=weight)


def _bn(channel):
    return layers.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _bn_last(channel):
    return layers.BatchNorm2d(channel, eps=1e-4, momentum=0.9,
                              gamma_init=0, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _fc(in_channel, out_channel):
    weight_shape = (out_channel, in_channel)
    weight = _weight_variable(weight_shape)
    return layers.Dense(in_channel, out_channel, has_bias=True, weight_init=weight, bias_init=0)


class AlexNet(layers.Layer):
    """
    Get AlexNet neural network.

    Args:
        class_num (int): Class number. Default: 1000.

    Returns:
        layers.Layer, layer instance of AlexNet neural network.

    Examples:
        >>> from tinyms.model import AlexNet
        >>>
        >>> net = AlexNet(class_num=1000)
    """

    def __init__(self, class_num=1000):
        super(AlexNet, self).__init__()

        self.features = layers.SequentialLayer(
            [
                _conv11x11(3, 64, 4),
                ReLU(),
                MaxPool2d(kernel_size=3, stride=2),
                _conv5x5(64, 192),
                ReLU(),
                MaxPool2d(kernel_size=3, stride=2),
                _conv3x3(192, 384),
                ReLU(),
                _conv3x3(384, 256),
                ReLU(),
                _conv3x3(256, 256),
                ReLU(),
                MaxPool2d(kernel_size=3, stride=2),
                Flatten(),
                Dropout(),
                _fc(256*6*6, 4096),
                ReLU(),
                Dropout(),
                _fc(4096, 4096),
                ReLU(),
                _fc(4096, class_num)
            ]

        )

    def construct(self, x):
        x = self.features(x)
        return x


def alexnet(**kwargs):
    """
    Get AlexNet neural network.

    Args:
        class_num (int): Class number. Default: 10.

    Returns:
        layers.Layer, layer instance of AlexNet neural network.

    Examples:
        >>> from tinyms.model import alexnet
        >>>
        >>> net = alexnet(class_num=10)
    """
    return AlexNet(class_num=kwargs.get('class_num', 10))
