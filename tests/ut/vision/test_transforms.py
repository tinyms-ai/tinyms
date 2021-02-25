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
from tinyms.vision import mnist_transform, cifar10_transform, imagefolder_transform


def test_mnist_transform():
    img = np.ones((32, 32))
    img = mnist_transform(img)
    print(img)

    img = np.ones((32, 32, 1))
    img = mnist_transform(img)
    print(img)

    img = np.ones((32, 32, 3))
    img = mnist_transform(img)
    print(img)


def test_mnist_transform_postprocess():
    input = np.array([[10, 1, 4, 2, 5, 18, -10, -4, 3, 7]])
    label = mnist_transform.postprocess(input)
    assert label == {'prediction: ': 5, 'score': 0.9996444101857648}

    label = mnist_transform.postprocess(input, 'TOP5_CLASS')
    print(label)
    expected = "TOP1: 5, score: 0.99964441018576477571\n"+"TOP2: 0, score: 0.00033534334080897316\n"+"TOP3: 9, score: 0.00001669576183556424\n"+"TOP4: 4, score: 0.00000225952565686712\n"+"TOP5: 2, score: 0.00000083123303596081\n"
    print(expected)
    assert label == expected


def test_cifar10_transform():
    img = np.ones((720, 720, 3))
    img = cifar10_transform(img)
    print(img)


def test_cifar10_transform_postprocess():
    input = np.array([[10, 1, 4, 2, 5, 18, -10, -4, 3, 7]])
    label = cifar10_transform.postprocess(input)
    assert label == {'prediction: ': 'dog', 'score': 0.9996444101857648}

    label = cifar10_transform.postprocess(input, strategy='TOP5_CLASS')
    expected = "TOP1: dog, score: 0.99964441018576477571\nTOP2: airplane, score: 0.00033534334080897316\nTOP3: truck, score: 0.00001669576183556424\nTOP4: deer, score: 0.00000225952565686712\nTOP5: bird, score: 0.00000083123303596081\n"
    assert label == expected


def test_imagefolder_transform():
    img = np.ones((1080, 1080, 3))
    img = imagefolder_transform(img)
    print(img)


def test_imagefolder_transform_postprocess():
    input = np.array([[10, 4, 2, 5, 18, -10, -4, 3, 7]])
    label = imagefolder_transform.postprocess(input)
    assert label == {'prediction: ': 'Entoloma霍氏粉褶菌,伞菌目,粉褶菌科,粉褶菌属,主要分布于新西兰北岛和南岛西部,有毒', 'score': 0.9996444515557065}

    label = imagefolder_transform.postprocess(input, strategy='TOP5_CLASS')
    expected = "TOP1: Entoloma霍氏粉褶菌,伞菌目,粉褶菌科,粉褶菌属,主要分布于新西兰北岛和南岛西部,有毒, score: 0.99964445155570647028\nTOP2: Agaricus双孢蘑菇,伞菌目,蘑菇科,蘑菇属,广泛分布于北半球温带,无毒, score: 0.00033534335468704253\nTOP3: Suillus乳牛肝菌,牛肝菌目,乳牛肝菌科,乳牛肝菌属,分布于吉林、辽宁、山西、安徽、江西、浙江、湖南、四川、贵州等地,无毒, score: 0.00001669576252651263\nTOP4: Cortinarius掷丝膜菌,伞菌目,丝膜菌科,丝膜菌属,分布于湖南等地(夏秋季在山毛等阔叶林地上生长), score: 0.00000225952575037681\nTOP5: Amanita毒蝇伞,伞菌目,鹅膏菌科,鹅膏菌属,主要分布于我国黑龙江、吉林、四川、西藏、云南等地,有毒, score: 0.00000083123307036111\n"
    assert label == expected
