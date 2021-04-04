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

"""ResNet Generator."""
from tinyms import layers
from tinyms.primitives import Tanh
from .common_net import ConvNormReLU, ConvTransposeNormReLU


class ResidualBlock(layers.Layer):
    """
    ResNet residual block definition.

    Args:
        dim (int): Input and output channel.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        dropout (bool): Use dropout or not. Default: False.
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, dim, norm_mode='batch', dropout=True, pad_mode="CONSTANT"):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode)
        self.conv2 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode, use_relu=False)
        self.dropout = dropout
        if dropout:
            self.dropout = layers.Dropout(0.5)

    def construct(self, x):
        out = self.conv1(x)
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        return x + out


class ResNetGenerator(layers.Layer):
    """
    ResNet Generator of GAN.

    Args:
        in_planes (int): Input channel.
        ngf (int): Output channel.
        n_layers (int): The number of ConvNormReLU blocks.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        dropout (bool): Use dropout or not. Default: False.
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".

    Returns:
        Tensor, output tensor.
    """
    def __init__(self, in_planes=3, ngf=64, n_layers=9, alpha=0.2, norm_mode='batch', dropout=True,
                 pad_mode="CONSTANT"):
        super(ResNetGenerator, self).__init__()
        self.conv_in = ConvNormReLU(in_planes, ngf, 7, 1, alpha=alpha, norm_mode=norm_mode, pad_mode=pad_mode)
        self.down_1 = ConvNormReLU(ngf, ngf * 2, 3, 2, alpha, norm_mode)
        self.down_2 = ConvNormReLU(ngf * 2, ngf * 4, 3, 2, alpha, norm_mode)
        layer_list = [ResidualBlock(ngf * 4, norm_mode, dropout=dropout, pad_mode=pad_mode)] * n_layers
        self.residuals = layers.SequentialLayer(layer_list)
        self.up_2 = ConvTransposeNormReLU(ngf * 4, ngf * 2, 3, 2, alpha, norm_mode)
        self.up_1 = ConvTransposeNormReLU(ngf * 2, ngf, 3, 2, alpha, norm_mode)
        if pad_mode == "CONSTANT":
            self.conv_out = layers.Conv2d(ngf, 3, kernel_size=7, stride=1, pad_mode='pad', padding=3)
        else:
            pad = layers.Pad(paddings=((0, 0), (0, 0), (3, 3), (3, 3)), mode=pad_mode)
            conv = layers.Conv2d(ngf, 3, kernel_size=7, stride=1, pad_mode='pad')
            self.conv_out = layers.SequentialLayer([pad, conv])
        self.activate = Tanh()

    def construct(self, x):
        x = self.conv_in(x)
        x = self.down_1(x)
        x = self.down_2(x)
        x = self.residuals(x)
        x = self.up_2(x)
        x = self.up_1(x)
        output = self.conv_out(x)
        return self.activate(output)
