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


def test_cifar10_transform():
    img = np.ones((720, 720, 3))
    img = cifar10_transform(img)
    print(img)


def test_imagefolder_transform():
    img = np.ones((1080, 1080, 3))
    img = imagefolder_transform(img)
    print(img)
