"""
Qiu Kunfeng. ZJUT;CETC36.
model architecture of densenet
"""

import math
import numpy as np

import tinyms as ts
from tinyms import layers, Tensor
from tinyms.primitives import tensor_add, Softmax, ReduceMean, Concat

# __all__ = ["DenseNet121", "DenseNet100"]


class GlobalAvgPooling(layers.Layer):
    """
    Pooling Layer, do average pooling operation.

    Returns:
        Tensor, output tensor.
    """
    def __init__(self):
        super(GlobalAvgPooling, self).__init__()
        self.mean = ReduceMean(keep_dims=False)

    def construct(self, x):
        x = self.mean(x, (2, 3))
        return x


class CommonHead(layers.Layer):
    def __init__(self, num_classes, out_channels):
        super(CommonHead, self).__init__()
        self.avgpool = GlobalAvgPooling()
        self.fc = layers.Dense(out_channels, num_classes)

    def construct(self, x):
        x = self.avgpool(x)
        x = self.fc(x)
        return x


def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    weight_shape = (out_channels, in_channels, 1, 1)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride,
                         padding=padding, pad_mode='pad', weight_init=weight)


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    weight_shape = (out_channels, in_channels, 3, 3)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride,
                         padding=padding, pad_mode='pad', weight_init=weight)


def conv7x7(in_channels, out_channels, stride=1, padding=3):
    weight_shape = (out_channels, in_channels, 7, 7)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channels, out_channels, kernel_size=7, stride=stride,
                         padding=padding, pad_mode='pad', weight_init=weight)


class _DenseLayer(layers.Layer):
    """
    the dense layer, include 2 conv layer
    """
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(_DenseLayer, self).__init__()
        self.norm1 = layers.BatchNorm2d(num_input_features)
        self.relu1 = layers.ReLU()
        self.conv1 = conv1x1(num_input_features, bn_size*growth_rate)

        self.norm2 = layers.BatchNorm2d(bn_size*growth_rate)
        self.relu2 = layers.ReLU()
        self.conv2 = conv3x3(bn_size*growth_rate, growth_rate)

        # nn.Dropout in MindSpore use keep_prob, diff from Pytorch
        self.keep_prob = 1.0 - drop_rate
        self.dropout = layers.Dropout(keep_prob=self.keep_prob)

    def construct(self, features):
        bottleneck = self.conv1(self.relu1(self.norm1(features)))
        new_features = self.conv2(self.relu2(self.norm2(bottleneck)))
        if self.keep_prob < 1:
            new_features = self.dropout(new_features)
        return new_features


class _DenseBlock(layers.Layer):
    """
    the dense block
    """
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(_DenseBlock, self).__init__()
        self.cell_list = layers.SequentialLayer()
        # self.cell_list = nn.CellList()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate
            )
            self.cell_list.append(layer)

        self.concate = Concat(axis=1)

    def construct(self, init_features):
        features = init_features
        for layer in self.cell_list:
            new_features = layer(features)
            features = self.concate((features, new_features))
        return features


class _Transition(layers.Layer):
    """
    the transition layer
    """
    def __init__(self, num_input_features, num_output_features, avgpool=False):
        super(_Transition, self).__init__()
        if avgpool:
            poollayer = layers.AvgPool2d
        else:
            poollayer = layers.MaxPool2d
        # self.features = layers.SequentialLayer(([
            # ('norm', layers.BatchNorm2d(num_input_features)),
            # ('relu', layers.ReLU()),
            # ('conv', conv1x1(num_input_features, num_output_features)),
            # ('pool', poollayer)
        # ]))
        self.features = layers.SequentialLayer([
            layers.BatchNorm2d(num_input_features),
            layers.ReLU(),
            conv1x1(num_input_features, num_output_features),
            poollayer(kernel_size=2, stride=2)
        ])

    def construct(self, x):
        x = self.features(x)
        return x


class Densenet(layers.Layer):
    """
    the densenet architecture
    """
    # __constants__ = ['features']

    def __init__(self, growth_rate, block_config, num_init_features=None, bn_size=4, drop_rate=0):
        super(Densenet, self).__init__()

        # layers = OrderedDict()
        if num_init_features:
            self.features = layers.SequentialLayer([
                conv7x7(3, num_init_features, stride=2, padding=3),
                layers.BatchNorm2d(num_init_features),
                layers.ReLU(),
                layers.MaxPool2d(kernel_size=3, stride=2)
            ])    
            # layers_.append(conv7x7(3, num_init_features, stride=2, padding=3))
            # layers_.append(layers.BatchNorm2d(num_init_features))
            # layers_.append(layers.ReLU())
            # layers_.append(layers.MaxPool2d(kernel_size=3, stride=2, pad_mode='same'))
            num_features = num_init_features
        else:
            self.features = layers.SequentialLayer([
                conv3x3(3, growth_rate*2, stride=1, padding=1),
                layers.BatchNorm2d(growth_rate*2),
                layers.ReLU()
            ])
            # layers_.append(conv3x3(3, growth_rate*2, stride=1, padding=1))
            # layers_.append(layers.BatchNorm2d(growth_rate*2))
            # layers_.append(layers.ReLU())
            num_features = growth_rate * 2

        # Each denseblock
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate
            )
            self.features.append(block)
            num_features = num_features + num_layers*growth_rate

            if i != len(block_config)-1:
                if num_init_features:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                        avgpool=False)
                else:
                    trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2,
                                        avgpool=True)
                self.features.append(trans)
                num_features = num_features // 2

        # Final batch norm
        # self.features.append([layers.BatchNorm2d(num_features), layers.ReLU()])  # ////
        self.batchnorm_ = layers.BatchNorm2d(num_features)
        self.relu_ = layers.ReLU()
        self.out_channels = num_features

    def construct(self, x):
        x = self.features(x)
        x = self.batchnorm_(x)
        x = self.relu_(x)
        return x

    def get_out_channels(self):
        return self.out_channels


def densenet100(**kwargs):
    return Densenet(growth_rate=12, block_config=(16, 16, 16), **kwargs)


def _densenet121(**kwargs):
    return Densenet(growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, **kwargs)


def _densenet161(**kwargs):
    return Densenet(growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, **kwargs)


def _densenet169(**kwargs):
    return Densenet(growth_rate=32, block_config=(6, 12, 32, 32), num_init_features=64, **kwargs)


def _densenet201(**kwargs):
    return Densenet(growth_rate=32, block_config=(6, 12, 48, 32), num_init_features=64, **kwargs)


class DenseNet100(layers.Layer):
    """
    the densenet100 architecture
    """
    def __init__(self, num_classes, include_top=True):
        super(DenseNet100, self).__init__()
        self.backbone = densenet100(num_init_features=24)
        out_channels = self.backbone.get_out_channels()  # 342
        self.include_top = include_top
        if self.include_top:
            self.head = CommonHead(num_classes, out_channels)

        # default_recurisive_init(self)
        # for _, cell in self.cells_and_names():
            # if isinstance(cell, nn.Conv2d):
                # cell.weight.set_data(init.initializer(KaimingNormal(a=math.sqrt(5), mode='fan_out',
                                                                    # nonlinearity='relu'),
                                                      # cell.weight.shape,
                                                      # cell.weight.dtype))
            # elif isinstance(cell, nn.BatchNorm2d):
                # cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
                # cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
            # elif isinstance(cell, nn.Dense):
                # cell.bias.set_data(init.initializer('zeros', cell.bias.shape))

    def construct(self, x):
        x = self.backbone(x)
        if not self.include_top:
            return x
        x = self.head(x)
        return x


class DenseNet121(layers.Layer):
    """
    the densenet121 architecture
    """
    def __init__(self, num_classes, include_top=True):
        super(DenseNet121, self).__init__()
        self.backbone = _densenet121()
        out_channels = self.backbone.get_out_channels()
        self.include_top = include_top
        if self.include_top:
            self.head = CommonHead(num_classes, out_channels)

        # default_recurisive_init(self)
        # for _, cell in self.cells_and_names():
            # if isinstance(cell, nn.Conv2d):
                # cell.weight.set_data(init.initializer(KaimingNormal(a=math.sqrt(5), mode='fan_out',
                                                                    # nonlinearity='relu'),
                                                      # cell.weight.shape,
                                                      # cell.weight.dtype))
            # elif isinstance(cell, nn.BatchNorm2d):
                # cell.gamma.set_data(init.initializer('ones', cell.gamma.shape))
                # cell.beta.set_data(init.initializer('zeros', cell.beta.shape))
            # elif isinstance(cell, nn.Dense):
                # cell.bias.set_data(init.initializer('zeros', cell.bias.shape))

    def construct(self, x):
        x = self.backbone(x)
        if not self.include_top:
            return x
        x = self.head(x)
        return x
