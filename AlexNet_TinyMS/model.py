import numpy as np
from scipy.stats import truncnorm

import tinyms as ts
from tinyms import layers, Tensor
from tinyms.primitives import tensor_add, ReduceMean
from tinyms import primitives as P
from tinyms.layers import AvgPool2d, ReLU, MaxPool2d, Flatten, Dropout




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


class _DenseLayer(layers.Layer):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(_DenseLayer, self).__init__()
        self.layer = layers.SequentialLayer(
            [_bn(in_channels),
             layers.ReLU(),
             _conv1x1(in_channels, bn_size * growth_rate),
             _bn(bn_size*growth_rate),
             layers.ReLU(),
             _conv3x3(bn_size*growth_rate,growth_rate),
             ])
        self.ops = P.Concat(axis=1)
    # 重载forward函数
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

class AlexNet(layers.Layer):
    def __init__(self, num_classes=1000):
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
                _fc(4096, num_classes)
            ]

        )
    def construct(self, x):
        x = self.features(x)
        return x


# data =  ts.ones([10, 3, 224, 224])
# print(data.shape)
# model = AlexNet()
# print(model(data).shape)