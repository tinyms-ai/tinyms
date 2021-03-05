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
"""SSD300 network based on MobileNetV2 backbone."""
import tinyms as ts
from .. import layers, primitives as P, Tensor
from ..vision.utils import ssd_default_boxes


def _make_divisible(v, divisor, min_value=None):
    """ensures that all layers have a channel number that is divisible by 8."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def _conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same'):
    return layers.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride,
                         padding=0, pad_mode=pad_mod, has_bias=True)


def _bn(channel):
    return layers.BatchNorm2d(channel, eps=1e-3, momentum=0.97,
                              gamma_init=1, beta_init=0, moving_mean_init=0, moving_var_init=1)


def _last_conv2d(in_channel, out_channel, kernel_size=3, stride=1, pad_mod='same', pad=0):
    in_channels = in_channel
    out_channels = in_channel
    depthwise_conv = layers.Conv2d(in_channels, out_channels, kernel_size, stride, pad_mode=pad_mod,
                                   padding=pad, group=in_channels)
    conv = _conv2d(in_channel, out_channel, kernel_size=1)
    return layers.SequentialLayer([depthwise_conv, _bn(in_channel), layers.ReLU6(), conv])


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
        padding = 0
        if groups == 1:
            conv = layers.Conv2d(in_channels, out_channels, kernel_size, stride,
                                 pad_mode='same', padding=padding)
        else:
            conv = layers.Conv2d(in_channels, in_channels, kernel_size, stride,
                                 pad_mode='same', padding=padding, group=in_channels)
        self.features = layers.SequentialLayer([conv,
                                                _bn(out_channels),
                                                layers.ReLU6()])

    def construct(self, x):
        output = self.features(x)
        return output


class InvertedResidual(layers.Layer):
    def __init__(self, inp, oup, stride, expand_ratio, use_relu=False):
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
            _bn(oup),
        ])
        self.conv = layers.SequentialLayer(residual_layers)
        self.use_relu = use_relu
        self.relu = layers.ReLU6()

    def construct(self, x):
        identity = x
        x = self.conv(x)
        if self.use_res_connect:
            return P.tensor_add(identity, x)
        if self.use_relu:
            x = self.relu(x)
        return x


class FlattenConcat(layers.Layer):
    """
    Concatenate predictions into a single tensor.

    Args:
        num_ssd_boxes (int): number of ssd boxes. Default is 1917.

    Returns:
        Tensor, flatten predictions.
    """

    def __init__(self, num_ssd_boxes=1917):
        super(FlattenConcat, self).__init__()
        self.num_ssd_boxes = num_ssd_boxes
        self.concat = P.Concat(axis=1)
        self.transpose = P.Transpose()

    def construct(self, inputs):
        output = ()
        batch_size = P.shape(inputs[0])[0]
        for x in inputs:
            x = self.transpose(x, (0, 2, 3, 1))
            output += (P.reshape(x, (batch_size, -1)),)
        res = self.concat(output)
        return P.reshape(res, (batch_size, self.num_ssd_boxes, -1))


class MultiBox(layers.Layer):
    """
    Multibox conv layers. Each multibox layer contains class conf scores
    and localization predictions.

    Args:
        class_num (int): number of classes. Default is 21.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, class_num=21):
        super(MultiBox, self).__init__()
        out_channels = [576, 1280, 512, 256, 256, 128]
        num_default = [3, 6, 6, 6, 6, 6]

        loc_layers = []
        cls_layers = []
        for k, out_channel in enumerate(out_channels):
            loc_layers += [_last_conv2d(out_channel, 4 * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]
            cls_layers += [_last_conv2d(out_channel, class_num * num_default[k],
                                        kernel_size=3, stride=1, pad_mod='same', pad=0)]

        self.multi_loc_layers = layers.LayerList(loc_layers)
        self.multi_cls_layers = layers.LayerList(cls_layers)
        self.flatten_concat = FlattenConcat()

    def construct(self, inputs):
        loc_outputs = ()
        cls_outputs = ()
        for i in range(len(self.multi_loc_layers)):
            loc_outputs += (self.multi_loc_layers[i](inputs[i]),)
            cls_outputs += (self.multi_cls_layers[i](inputs[i]),)
        return self.flatten_concat(loc_outputs), self.flatten_concat(cls_outputs)


class SSDWithMobileNetV2(layers.Layer):
    """
    MobileNetV2 backbone.

    Args:
        width_mult (float): Channels multiplier for round to 8/16 and others. Default is 1.0.
        round_nearest (int): Channel round to. Default is 8.
        input_channel (int): Input channel. Default is 32.
        last_channel (int): The channel of last layer. Default is 1280.

    Returns:
        Tensor, output feature maps.
        Tensor, output tensor.
    """

    def __init__(self, width_mult=1.0, round_nearest=8, input_channel=32, last_channel=1280):
        super(SSDWithMobileNetV2, self).__init__()
        # setting of inverted residual blocks
        cfg = [
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
        layer_index = 0
        for t, c, n, s in cfg:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                if layer_index == 13:
                    hidden_dim = int(round(input_channel * t))
                    self.expand_layer_conv_13 = ConvBNReLU(input_channel, hidden_dim, kernel_size=1)
                stride = s if i == 0 else 1
                backbone_layers.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
                layer_index += 1
        # building last several layers
        backbone_layers.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))

        self.features_1 = layers.SequentialLayer(backbone_layers[:14])
        self.features_2 = layers.SequentialLayer(backbone_layers[14:])

    def construct(self, x):
        out = self.features_1(x)
        expand_layer_conv_13 = self.expand_layer_conv_13(out)
        out = self.features_2(out)
        return expand_layer_conv_13, out


class SSD300(layers.Layer):
    """
    SSD300 Network. Default backbone is MobileNetV2.

    Args:
        backbone (Layer): backbone of ssd300 model.
        class_num (int): number of classes. Default is 21.
        is_training (Bool): Specify if in training step. Default is True.

    Returns:
        Tensor, localization predictions.
        Tensor, class conf scores.
    """

    def __init__(self, backbone, class_num=21, is_training=True):
        super(SSD300, self).__init__()

        self.backbone = backbone
        in_channels = [256, 576, 1280, 512, 256, 256]
        out_channels = [576, 1280, 512, 256, 256, 128]
        ratios = [0.2, 0.2, 0.2, 0.25, 0.5, 0.25]
        strides = [1, 1, 2, 2, 2, 2]
        residual_list = []
        for i in range(2, len(in_channels)):
            residual = InvertedResidual(in_channels[i], out_channels[i], stride=strides[i],
                                        expand_ratio=ratios[i], use_relu=True)
            residual_list.append(residual)
        self.multi_residual = layers.LayerList(residual_list)
        self.multi_box = MultiBox(class_num=class_num)
        self.is_training = is_training
        self.activation = P.Sigmoid()

    def construct(self, x):
        layer_out_13, output = self.backbone(x)
        multi_feature = (layer_out_13, output)
        feature = output
        for residual in self.multi_residual:
            feature = residual(feature)
            multi_feature += (feature,)
        pred_loc, pred_label = self.multi_box(multi_feature)
        if not self.is_training:
            pred_label = self.activation(pred_label)
        pred_loc = P.cast(pred_loc, ts.float32)
        pred_label = P.cast(pred_label, ts.float32)
        return pred_loc, pred_label


def ssd300_mobilenetv2(class_num=21, is_training=True):
    return SSD300(SSDWithMobileNetV2(), class_num=class_num, is_training=is_training)


class SSDInferWithDecoder(layers.Layer):
    """
    SSD300 infer wrapper to decode the bbox locations.

    Args:
        network (Layer): the origin ssd300 infer network without bbox decoder.
        default_boxes (Tensor): the default_boxes from anchor generator

    Returns:
        Tensor, the locations for bbox after decoder representing (y0, x0, y1, x1)
        Tensor, the prediction labels.
    """

    def __init__(self, network):
        super(SSDInferWithDecoder, self).__init__()
        self.network = network
        self.default_boxes = Tensor(ssd_default_boxes, dtype=ts.float32)
        prior_scaling_tensor = Tensor((0.1, 0.2), dtype=ts.float32)
        self.prior_scaling_xy = prior_scaling_tensor[0]
        self.prior_scaling_wh = prior_scaling_tensor[1]

    def construct(self, x):
        pred_loc, pred_label = self.network(x)

        default_bbox_xy = self.default_boxes[..., :2]
        default_bbox_wh = self.default_boxes[..., 2:]
        pred_xy = pred_loc[..., :2] * self.prior_scaling_xy * default_bbox_wh + default_bbox_xy
        pred_wh = P.Exp()(pred_loc[..., 2:] * self.prior_scaling_wh) * default_bbox_wh

        pred_xy_0 = pred_xy - pred_wh / 2.0
        pred_xy_1 = pred_xy + pred_wh / 2.0
        pred_xy = P.Concat(-1)((pred_xy_0, pred_xy_1))
        pred_xy = P.Maximum()(pred_xy, 0)
        pred_xy = P.Minimum()(pred_xy, 1)
        return pred_xy, pred_label


def ssd300_infer(class_num=21):
    net = SSD300(SSDWithMobileNetV2(), class_num=class_num, is_training=False)
    return SSDInferWithDecoder(net)
