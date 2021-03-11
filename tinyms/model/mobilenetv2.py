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

import tinyms as ts
from tinyms import layers, Tensor
from tinyms.primitives import tensor_add, Softmax, ReduceMean


def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class GlobalAvgPooling(layers.Layer):
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class ConvBNReLU(layers.Layer):
    """
    Convolution/Depthwise fused with Batchnorm and ReLU block definition.

    Args:
        in_channels (int): Input channel.
        out_channels (int): Output channel.
        kernel_size (int): Input kernel size. Default: 3.
        stride (int): Stride size for the first convolutional layer. Default: 1.
        groups (int): channel group. Convolution is 1 while Depthwise is input channel. Default: 1.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> ConvBNReLU(16, 256, kernel_size=1, stride=1, groups=1)
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        padding = (kernel_size - 1) // 2
        if groups == 1:
            conv = layers.Conv2d(in_channels, out_channels, kernel_size, stride,
                                 pad_mode='pad', padding=padding)
        else:
            conv = layers.Conv2d(in_channels, in_channels, kernel_size, stride,
                                 pad_mode='pad', padding=padding, group=in_channels)
        self.features = layers.SequentialLayer([conv,
                                                layers.BatchNorm2d(out_channels),
                                                layers.ReLU6()])

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(layers.Layer):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        residual_layers = []
        if expand_ratio != 1:
            residual_layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        residual_layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            layers.Conv2d(hidden_dim, oup, kernel_size=1, stride=1, has_bias=False),
            layers.BatchNorm2d(oup),
        ])
        self.conv = layers.SequentialLayer(residual_layers)

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return tensor_add(identity, x)
        return x


class MobileNetV2Backbone(layers.Layer):
    def __init__(self, width_mult=1., round_nearest=8, input_channel=32, last_channel=1280):
        super(MobileNetV2Backbone, self).__init__()
        # setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        # building first layer
        input_channel = _make_divisible(input_channel * width_mult, round_nearest)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult), round_nearest)
        backbone_layers = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in self.cfgs:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                backbone_layers.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        backbone_layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.backbone = layers.SequentialLayer(backbone_layers)
        self._initialize_weights()

    def construct(self, x):
        x = self.backbone(x)
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, layers.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.set_data(Tensor(np.random.normal(0, np.sqrt(2. / n),
                                                          m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(ts.zeros(m.bias.data.shape))
            elif isinstance(m, layers.BatchNorm2d):
                m.gamma.set_data(ts.ones(m.gamma.data.shape))
                m.beta.set_data(ts.zeros(m.beta.data.shape))


class MobileNetV2Head(layers.Layer):
    def __init__(self, input_channel=1280, class_num=1000, use_activation=False):
        super(MobileNetV2Head, self).__init__()
        # mobilenet head
        self.head = layers.SequentialLayer(([GlobalAvgPooling(),
                                             layers.Dense(input_channel, class_num)]))
        self.use_activation = use_activation
        self.activation = Softmax()
        self._initialize_weights()

    def construct(self, x):
        x = self.head(x)
        if self.use_activation:
            x = self.activation(x)
        return x

    def _initialize_weights(self):
        self.init_parameters_data()
        for _, m in self.cells_and_names():
            if isinstance(m, layers.Dense):
                m.weight.set_data(Tensor(np.random.normal(
                    0, 0.01, m.weight.data.shape).astype("float32")))
                if m.bias is not None:
                    m.bias.set_data(ts.zeros(m.bias.data.shape))


class MobileNetV2(layers.Layer):
    """
    MobileNetV2 architecture.

    Args:
        class_num (int): number of classes.
        width_mult (float): Channels multiplier for round to 8/16 and others. Default is 1.0.
        round_nearest (int): Channel round to. Default is 8.
        input_channel (int): Input channel. Default is 32.
        last_channel (int): The channel of last layer. Default is 1280.
    Returns:
        Tensor, output tensor.
    """

    def __init__(self, class_num=1000, width_mult=1.,
                 round_nearest=8, input_channel=32, last_channel=1280, is_training=True):
        super(MobileNetV2, self).__init__()
        self.backbone = MobileNetV2Backbone(width_mult=width_mult,
                                            round_nearest=round_nearest,
                                            input_channel=input_channel,
                                            last_channel=last_channel)
        self.head = MobileNetV2Head(input_channel=self.backbone.last_channel,
                                    class_num=class_num, use_activation=not is_training)

    def construct(self, x):
        x = self.backbone(x)
        x = self.head(x)
        return x


def mobilenetv2(class_num=1000, is_training=True):
    return MobileNetV2(class_num=class_num, is_training=is_training)


def mobilenetv2_infer(class_num=1000):
    return MobileNetV2(class_num=class_num, is_training=False)
