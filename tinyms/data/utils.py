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
""".. TinyMS data utils package."""
import os
import sys
import gzip
import matplotlib.pyplot as plt
import numpy as np
import requests


def _unzip(gzip_path):
    """unzip dataset file
    Args:
        gzip_path: dataset file path
    """
    with open(gzip_path.replace('.gz', ''), 'wb') as f:
        gz_file = gzip.GzipFile(gzip_path)
        f.write(gz_file.read())


def _fetch_and_unzip(url, file_name):
    """download the dataset from remote url
    Args:
        url: str, remote download url
        file_name: str, local path of downloaded file
    """
    res = requests.get(url, stream=True, verify=False)
    # get dataset size
    total_size = int(res.headers["Content-Length"])
    temp_size = 0
    with open(file_name, "wb+") as f:
        for chunk in res.iter_content(chunk_size=1024):
            temp_size += len(chunk)
            f.write(chunk)
            f.flush()
            done = int(100 * temp_size / total_size)
            # show download progress
            sys.stdout.write("\r[{}{}] {:.2f}%".format("█" * done, " " * (100 - done), 100 * temp_size / total_size))
            sys.stdout.flush()
    print("\n============== {} is already ==============".format(file_name))
    _unzip(file_name)
    os.remove(file_name)


def _download_mnist(local_path):
    """Download the dataset from http://yann.lecun.com/exdb/mnist/."""
    train_path = os.path.join(local_path, 'mnist', 'train')
    test_path = os.path.join(local_path, 'mnist', 'test')
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    print("************** Downloading the MNIST dataset **************")
    train_url = {"http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
                 "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"}
    test_url = {"http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
                "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"}
    for url in train_url:
        # split the file name from url
        file_name = os.path.join(train_path, url.split('/')[-1])
        if not os.path.exists(file_name.replace('.gz', '')):
            _fetch_and_unzip(url, file_name)
    for url in test_url:
        # split the file name from url
        file_name = os.path.join(test_path, url.split('/')[-1])
        if not os.path.exists(file_name.replace('.gz', '')):
            _fetch_and_unzip(url, file_name)


def _download_cifar10(local_path):
    '''Download the dataset from http://www.cs.toronto.edu/~kriz/cifar.html.'''
    dataset_path = os.path.join(local_path, 'cifar10')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print("************** Downloading the Cifar10 dataset **************")
    remote_url = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    file_name = os.path.join(dataset_path, remote_url.split('/')[-1])
    _fetch_and_unzip(remote_url, file_name)


def _download_cifar100(local_path):
    '''Download the dataset from http://www.cs.toronto.edu/~kriz/cifar.html.'''
    dataset_path = os.path.join(local_path, 'cifar100')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print("************** Downloading the Cifar100 dataset **************")
    remote_url = "http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    file_name = os.path.join(dataset_path, remote_url.split('/')[-1])
    _fetch_and_unzip(remote_url, file_name)


def download_dataset(dataset_name='mnist', local_path='.'):
    r'''
    This function is defined to easily download any public dataset
    without specifing much details.

    Args:
        dataset_name: str, the official name of dataset, currently supports `mnist` and `cifar10`.
            Default: `mnist`.
        local_path: str, specifies the local location of dataset to be downloaded.
            Default: `.`.
    '''
    if dataset_name not in ('mnist', 'cifar10', 'cifar100'):
        print("Currently dataset_name only supports `mnist`, `cifar10` and `cifar100`!")
        sys.exit(0)

    if dataset_name == 'mnist':
        return _download_mnist(local_path)
    elif dataset_name == 'cifar10':
        return _download_cifar10(local_path)
    else:
        return _download_cifar100(local_path)


class ImageViewer():
    r'''
    ImageViewer is a class defined for visualizing the input image.

    Args:
        image: PIL.Image, image input.
        label: str, specifies the label of this image.
    Examples:
        >>> form PIL import Image
        >>> img = Image.open('example.jpg')
        >>> img_viewer = ImageViewer(img, 'cat')
        >>> img_viewer.show()
        >>> print(img_viewer.label)
    '''

    def __init__(self, image, label):
        self._image = np.array(image)
        self._label = label

    def show(self):
        plt.imshow(np.squeeze(self._image))
        plt.title("label: %s" % self._label)
        plt.show()

    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label
