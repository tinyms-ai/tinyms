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

from .. import layers
from ..initializers import Normal


class LeNet(layers.Layer):
    """
    LeNet architecture.

    Args:
        class_num (int): The number of classes that the training images are belonging to.
        channel_num (int): The channel number.
    Returns:
        Tensor, output tensor.

    Examples:
        >>> LeNet(class_num=10)
    """

    def __init__(self, class_num=10, channel_num=1):
        super(LeNet, self).__init__()
        self.conv1 = layers.Conv2d(channel_num, 6, 5, pad_mode='valid')
        self.conv2 = layers.Conv2d(6, 16, 5, pad_mode='valid')
        self.relu = layers.ReLU()
        self.max_pool2d = layers.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = layers.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = layers.Dense(84, class_num, weight_init=Normal(0.02))

    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def lenet5(class_num=10):
    """
    Get LeNet5 neural network.

    Args:
        class_num (int): Class number.

    Returns:
        Layer, layer instance of LeNet5 neural network.

    Examples:
        >>> net = lenet5(class_num=10)
    """
    return LeNet(class_num=class_num)
