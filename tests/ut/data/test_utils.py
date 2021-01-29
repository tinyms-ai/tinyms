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
import os
import pytest
import numpy as np

from tinyms.data import download_dataset, ImageViewer


@pytest.mark.skip(reason="no way of currently testing this")
def test_download_dataset_mnist():
    download_dataset(dataset_name='mnist', local_path='/tmp')

    assert os.path.exists('/tmp/mnist/train')
    assert os.path.exists('/tmp/mnist/test')


@pytest.mark.skip(reason="no way of currently testing this")
def test_download_dataset_cifar10():
    download_dataset(dataset_name='cifar10', local_path='/tmp')

    assert os.path.exists('/tmp/cifar10/cifar-10-batches-bin/batches.meta.txt')


@pytest.mark.skip(reason="no way of currently testing this")
def test_download_dataset_cifar100():
    download_dataset(dataset_name='cifar100', local_path='/tmp')

    assert os.path.exists('/tmp/cifar100/cifar-100-bin/train.bin')
    assert os.path.exists('/tmp/cifar100/cifar-100-bin/test.bin')


def test_imageviewer():
    fake_img = np.random.uniform(0.0, 1.0, size=[3, 224, 224])
    image_viewer = ImageViewer(fake_img, 'cat')

    assert np.all(image_viewer.image == fake_img)
    assert image_viewer.label == 'cat'
