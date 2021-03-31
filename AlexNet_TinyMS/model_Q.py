"""
Qiu Kunfeng. ZJUT;CETC36.
Alexnet. According to MindSpore.
"""
import numpy as np
import tinyms as ts
from tinyms import layers, Tensor
from tinyms.primitives import tensor_add, ReduceMean


def conv(in_channels, out_channels, kernel_size, stride=1, padding=0, pad_mode="same"):
    weight_shape = (out_channels, in_channels, kernel_size, kernel_size)
    weight = _weight_variable(weight_shape)
    return layers.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                         pad_mode=pad_mode, weight_init=weight)
                         

def _weight_variable(shape, factor=0.01):
    init_value = np.random.randn(*shape).astype(np.float32) * factor
    return Tensor(init_value)


def fc_with_initialize(input_channels, out_channels, has_bias=True):
    weight_shape = (out_channels, input_channels)
    weight = _weight_variable(weight_shape)
    return layers.Dense(input_channels, out_channels, has_bias=has_bias, weight_init=weight, bias_init=0)


class AlexNet(layers.Layer):
    """
    Alexnet
    """
    def __init__(self, num_classes=10, channel=3, phase='train', include_top=True):
        super(AlexNet, self).__init__()
        self.conv1 = conv(channel, 64, 11, stride=4, pad_mode="same")
        self.conv2 = conv(64, 128, 5, pad_mode="same")
        self.conv3 = conv(128, 192, 3, pad_mode="same")
        self.conv4 = conv(192, 256, 3, pad_mode="same")
        self.conv5 = conv(256, 256, 3, pad_mode="same")
        self.relu = layers.ReLU()  # ////
        self.max_pool2d = layers.MaxPool2d(kernel_size=3, stride=2, pad_mode='valid')
        self.include_top = include_top
        if self.include_top:
            dropout_ratio = 0.65
            if phase == 'test':
                dropout_ratio = 1.0
            self.flatten = layers.Flatten()
            self.fc1 = fc_with_initialize(6 * 6 * 256, 4096)
            self.fc2 = fc_with_initialize(4096, 4096)
            self.fc3 = fc_with_initialize(4096, num_classes)
            self.dropout = layers.Dropout(dropout_ratio)

    def construct(self, x):
        """define network"""
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.relu(x)
        x = self.max_pool2d(x)
        if not self.include_top:
            return x
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
