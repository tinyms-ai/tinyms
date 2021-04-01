import numpy as np
from scipy.stats import truncnorm

import tinyms as ts
from tinyms import layers, Tensor
from tinyms.primitives import tensor_add, ReduceMean
from tinyms import primitives as P
from tinyms.layers import AvgPool2d, ReLU, MaxPool2d, Conv2d, BatchNorm2d, Dense




class DenseUnit(layers.Layer):

    def __init__(self, in_channels, growth_rate, bn_size):
        super(DenseUnit, self).__init__()
        self.bn1 = BatchNorm2d(in_channels)
        self.relu1 = ReLU()
        self.conv1 = Conv2d(in_channels,bn_size * growth_rate,1)
        self.bn2 = BatchNorm2d(bn_size * growth_rate)
        self.relu2 = ReLU()
        self.conv2 = Conv2d(bn_size*growth_rate,growth_rate,3)
        self.concat = P.Concat(axis=1)

    def construct(self, x):
        new_features = self.conv1(self.relu1(self.bn1(x)))
        new_features = self.conv2(self.relu2(self.bn2(new_features)))

        return self.concat((x, new_features))





class DenseBlock(layers.Layer):
    def __init__(self, num_layers, in_channels, bn_size, growth_rate):
        super(DenseBlock, self).__init__()
        self.layer = layers.LayerList()
        for i in range(num_layers):
            self.layer.append(DenseUnit(in_channels+growth_rate*i,
                                        growth_rate, bn_size))
    def construct(self, x):

        for layer in self.layer:
            x = layer(x)
        return x

class TransitionBlock(layers.Layer):
    def __init__(self, in_channels, out_channels):
        super(TransitionBlock, self).__init__()

        self.bn = BatchNorm2d(in_channels)
        self.relu = ReLU()
        self.conv = Conv2d(in_channels, out_channels, 1)
        self.pool = AvgPool2d(kernel_size=2, stride=2, pad_mode='same')

    def construct(self, x):

        out = self.conv(self.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNet(layers.Layer):
    def __init__(self, num_classes=1000, growth_rate=12, block_config=(6,12,24,16),
                 bn_size=4, theta=0.5, use_for_cifar10=False):
        super(DenseNet, self).__init__()


        num_init_feature = 2 * growth_rate
        if use_for_cifar10:
             self.features = layers.LayerList(
                [
                    Conv2d(3, num_init_feature, 3, pad_mode='same'),
                    BatchNorm2d(num_init_feature),
                    ReLU()
                ]
                )           
        else:
            self.features = layers.LayerList(
                [
                    Conv2d(3, num_init_feature, 7, stride=2, pad_mode='same'),
                    BatchNorm2d(num_init_feature),
                    ReLU(),
                    MaxPool2d(kernel_size=2, stride=2, pad_mode='same', data_format='NCHW')
                ]
                )

        num_feature = num_init_feature
        for i, num_layers in enumerate(block_config):

            self.features.append(DenseBlock(num_layers, num_feature,
                                bn_size, growth_rate))
            num_feature = num_feature + growth_rate * num_layers
            if i != len(block_config)-1:
                self.features.append(TransitionBlock(num_feature,int(num_feature * theta)))
                num_feature = int(num_feature * theta)

        self.features.append(BatchNorm2d(num_feature))
        self.features.append(ReLU())


        self.globalpool = ReduceMean(keep_dims=False)
        self.classifier = Dense(num_feature, num_classes)

    def construct(self, x):

        for layer in self.features:
            x = layer(x)

        features = self.classifier(self.globalpool(x, (2, 3)))


        return features

def densenetBC_100(num_classes=10):

    return DenseNet(num_classes=num_classes, growth_rate=12, block_config=(16, 16, 16))

