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

"""Common network."""
from tinyms.initializers import initializer, Normal, XavierUniform
from tinyms import layers


def init_weights(net, init_type='normal', init_gain=0.02):
    """
    Initialize network weights.

    Parameters:
        net (layer): Network to be initialized
        init_type (str): The name of an initialization method: normal | xavier.
        init_gain (float): Gain factor for normal and xavier.

    """
    for _, layer in net.cells_and_names():
        if isinstance(layer, (layers.Conv2d, layers.Conv2dTranspose)):
            if init_type == 'normal':
                layer.weight.set_data(initializer(Normal(init_gain), layer.weight.shape))
            elif init_type == 'xavier':
                layer.weight.set_data(initializer(XavierUniform(init_gain), layer.weight.shape))
            elif init_type == 'constant':
                layer.weight.set_data(initializer(0.001, layer.weight.shape))
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif isinstance(layer, layers.BatchNorm2d):
            layer.gamma.set_data(initializer('ones', layer.gamma.shape))
            layer.beta.set_data(initializer('zeros', layer.beta.shape))


class ConvNormReLU(layers.Layer):
    """
    Convolution fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".
        use_relu (bool): Use relu or not. Default: True.
        padding (int): Pad size, if it is None, it will calculate by kernel_size. Default: None.

    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=4,
                 stride=2,
                 alpha=0.2,
                 norm_mode='batch',
                 pad_mode='CONSTANT',
                 use_relu=True,
                 padding=None):
        super(ConvNormReLU, self).__init__()
        self.norm = layers.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
            norm = layers.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            conv = layers.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad',
                                 has_bias=has_bias, padding=padding)
            layer_list = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = layers.Pad(paddings=paddings, mode=pad_mode)
            conv = layers.Conv2d(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layer_list = [pad, conv, norm]
        if use_relu:
            relu = layers.ReLU()
            if alpha > 0:
                relu = layers.LeakyReLU(alpha)
            layer_list.append(relu)
        self.features = layers.SequentialLayer(layer_list)

    def construct(self, x):
        output = self.features(x)
        return output


class ConvTransposeNormReLU(layers.Layer):
    """
    ConvTranspose2d fused with BatchNorm/InstanceNorm and ReLU/LackyReLU block definition.

    Args:
        in_planes (int): Input channel.
        out_planes (int): Output channel.
        kernel_size (int): Input kernel size. Default: 4.
        stride (int): Stride size for the first convolutional layer. Default: 2.
        alpha (float): Slope of LackyReLU. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
                        Default: "CONSTANT".
        use_relu (bool): use relu or not. Default: True.
        padding (int): pad size, if it is None, it will calculate by kernel_size. Default: None.

    Returns:
        Tensor, output tensor.
    """
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=4,
                 stride=2,
                 alpha=0.2,
                 norm_mode='batch',
                 pad_mode='CONSTANT',
                 use_relu=True,
                 padding=None):
        super(ConvTransposeNormReLU, self).__init__()
        conv = layers.Conv2dTranspose(in_planes, out_planes, kernel_size, stride=stride, pad_mode='same')
        norm = layers.BatchNorm2d(out_planes)
        if norm_mode == 'instance':
            # Use BatchNorm2d with batchsize=1, affine=False, training=True instead of InstanceNorm2d
            norm = layers.BatchNorm2d(out_planes, affine=False)
        has_bias = (norm_mode == 'instance')
        if padding is None:
            padding = (kernel_size - 1) // 2
        if pad_mode == 'CONSTANT':
            conv = layers.Conv2dTranspose(in_planes, out_planes, kernel_size, stride, pad_mode='same', has_bias=has_bias)
            layer_list = [conv, norm]
        else:
            paddings = ((0, 0), (0, 0), (padding, padding), (padding, padding))
            pad = layers.Pad(paddings=paddings, mode=pad_mode)
            conv = layers.Conv2dTranspose(in_planes, out_planes, kernel_size, stride, pad_mode='pad', has_bias=has_bias)
            layer_list = [pad, conv, norm]
        if use_relu:
            relu = layers.ReLU()
            if alpha > 0:
                relu = layers.LeakyReLU(alpha)
            layer_list.append(relu)
        self.features = layers.SequentialLayer(layer_list)

    def construct(self, x):
        output = self.features(x)
        return output
