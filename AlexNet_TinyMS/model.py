import numpy as np
from scipy.stats import truncnorm

import tinyms as ts
from tinyms import layers, Tensor
from tinyms.primitives import tensor_add, ReduceMean
from tinyms import primitives as P
from tinyms.layers import AvgPool2d, ReLU, MaxPool2d, Flatten, Dropout, Conv2d, Dense




class alexnet(layers.Layer):
    def __init__(self, num_classes=1000):
        super(alexnet, self).__init__()

        self.conv1 = Conv2d(3, 64, 11, 4, pad_mode='pad', padding=2)
        self.relu1 = ReLU()
        self.pool1 = MaxPool2d(kernel_size=3, stride=2)


        self.conv2 = Conv2d(64, 192, 5, stride=1, pad_mode='same')
        self.relu2 = ReLU()
        self.pool2 = MaxPool2d(kernel_size=3, stride=2)


        self.conv3 = Conv2d(192, 384, 3, pad_mode='same')
        self.relu3 = ReLU()

        self.conv4 = Conv2d(384, 256, 3, pad_mode='same')
        self.relu4 = ReLU()
        self.conv5 = Conv2d(256, 256, 3, pad_mode='same')
        self.relu5 = ReLU()
        self.pool5 = MaxPool2d(kernel_size=3, stride=2)
        self.classifier = layers.SequentialLayer(
            [
                Flatten(),
                Dropout(),
                Dense(256*6*6, 4096),
                ReLU(),
                Dropout(),
                Dense(4096, 4096),
                ReLU(),
                Dense(4096, num_classes)
            ]
        )


    def construct(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.pool5(self.relu5(self.conv5(x)))
        x = self.classifier(x)
        return x

def AlexNet(num_classes=10):
    return alexnet(num_classes=num_classes)