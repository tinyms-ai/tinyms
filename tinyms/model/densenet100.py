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
from tinyms.primitives import ReduceMean, Concat
from tinyms.layers import AvgPool2d, ReLU, MaxPool2d


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


class _DenseLayer(layers.Layer):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.layer = layers.SequentialLayer(
            [_bn(in_channels),
             layers.ReLU(),
             _conv1x1(in_channels, bn_size * growth_rate),
             _bn(bn_size*growth_rate),
             layers.ReLU(),
             _conv3x3(bn_size*growth_rate, growth_rate),
             ])
        self.ops = Concat(axis=1)

    def construct(self, x):
        new_features = self.layer(x)
        return self.ops((x, new_features))


class _DenseBlock(layers.Layer):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(_DenseBlock, self).__init__()
        self.layer = layers.SequentialLayer()
        for i in range(num_layers):
            self.layer.append(_DenseLayer(in_channels+growth_rate*i,
                                          growth_rate, bn_size))

    def construct(self, x):
        out = self.layer(x)
        return out


class _Transition(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(_Transition, self).__init__()
        self.layer = layers.SequentialLayer(
            [_bn(in_channels),
             layers.ReLU(),
             _conv1x1(in_channels, out_channels),
             AvgPool2d(kernel_size=2, stride=2, pad_mode='same', data_format='NCHW')
             ])

    def construct(self, x):
        out = self.layer(x)
        return out


class DenseNet(layers.Layer):
    def __init__(self, class_num=1000, growth_rate=12, block_config=(6, 12, 24, 16),
                 bn_size=4, theta=0.5, bc=False):
        super(DenseNet, self).__init__()

        num_init_feature = 2 * growth_rate
        if bc:
            self.features = layers.SequentialLayer(
                [
                    _conv3x3(3, num_init_feature, 1),
                    _bn(num_init_feature),
                    ReLU()
                ]
            )
        else:
            self.features = layers.SequentialLayer(
                [
                    _conv7x7(3, num_init_feature, 2),
                    _bn(num_init_feature),
                    ReLU(),
                    MaxPool2d(kernel_size=2, stride=2, pad_mode='same', data_format='NCHW')
                ]
            )

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):

            self.features.append(_DenseBlock(num_layers, num_feature,
                                             bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.append(_Transition(num_feature, int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.norm = _bn(num_feature)
        self.relu = ReLU()
        # self.features.append([_bn(num_feature),ReLU()])
        self.mean = ReduceMean(keep_dims=True)
        self.flatten = layers.Flatten()
        self.end_point = _fc(num_feature, class_num)

    def construct(self, x):
        features = self.features(x)
        features = self.relu(self.norm(features))
        features = self.end_point(self.flatten(self.mean(features, (2, 3))))
        return features


def densenet100(**kwargs):
    """
    Get DenseNet instance for model training, evaluation and prediction.

    Args:
        class_num (int): The number of classes. Default: 10.

    Returns:
        model.DenseNet, DenseNet instance.
    """
    return DenseNet(class_num=kwargs.get('class_num', 10),
                    growth_rate=12, block_config=(16, 16, 16))
