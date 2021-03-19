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
import tarfile
import requests


__all__ = ['download_dataset']


def _unzip(gzip_path):
    """unzip dataset file
    Args:
        gzip_path: dataset file path
    """
    # decompress the file if gzip_path ends with `.tar`
    if gzip_path.endswith('.tar'):
        with tarfile.open(gzip_path) as f:
            f.extractall(gzip_path[:gzip_path.rfind('/')])
    elif gzip_path.endswith('.gz'):
        gzip_file = gzip_path.replace('.gz', '')
        with open(gzip_file, 'wb') as f:
            gz_file = gzip.GzipFile(gzip_path)
            f.write(gz_file.read())
        # decompress the file if gz_file ends with `.tar`
        if gzip_file.endswith('.tar'):
            with tarfile.open(gzip_file) as f:
                f.extractall(gzip_file[:gzip_file.rfind('/')])
    else:
        print("Currently the format of unzip dataset only supports `*.tar`, `*.gz` and `*.tar.gz`!")
        sys.exit(0)


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
    print("\n============== {} is ready ==============".format(file_name))
    _unzip(file_name)
    os.remove(file_name)


def _download_mnist(local_path):
    """Download the dataset from http://yann.lecun.com/exdb/mnist/."""
    dataset_path = os.path.join(local_path, 'mnist')
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')
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

    return dataset_path


def _download_cifar10(local_path):
    '''Download the dataset from http://www.cs.toronto.edu/~kriz/cifar.html.'''
    dataset_path = os.path.join(local_path, 'cifar10')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print("************** Downloading the Cifar10 dataset **************")
    remote_url = "http://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    file_name = os.path.join(dataset_path, remote_url.split('/')[-1])
    if not os.path.exists(file_name.replace('.gz', '')):
        _fetch_and_unzip(remote_url, file_name)

    return os.path.join(dataset_path, 'cifar-10-batches-bin')


def _download_cifar100(local_path):
    '''Download the dataset from http://www.cs.toronto.edu/~kriz/cifar.html.'''
    dataset_path = os.path.join(local_path, 'cifar100')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print("************** Downloading the Cifar100 dataset **************")
    remote_url = "http://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
    file_name = os.path.join(dataset_path, remote_url.split('/')[-1])
    if not os.path.exists(file_name.replace('.gz', '')):
        _fetch_and_unzip(remote_url, file_name)

    return os.path.join(dataset_path, 'cifar-100-binary')


def _download_voc(local_path):
    '''Download the dataset from http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html.'''
    dataset_path = os.path.join(local_path, 'voc')
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    print("************** Downloading the VOC2007 dataset **************")
    remote_url = "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar"
    file_name = os.path.join(dataset_path, remote_url.split('/')[-1])
    if not os.path.exists(os.path.join(dataset_path, 'VOCdevkit', 'VOC2007')):
        _fetch_and_unzip(remote_url, file_name)

    return os.path.join(dataset_path, 'VOCdevkit', 'VOC2007')


download_checker = {
    'mnist': _download_mnist,
    'cifar10': _download_cifar10,
    'cifar100': _download_cifar100,
    'voc': _download_voc,
}


def download_dataset(dataset_name, local_path='.'):
    r'''
    This function is defined to easily download any public dataset
    without specifing much details.
    Args:
        dataset_name: str, the official name of dataset, currently supports `mnist`, `cifar10` and `cifar100`.
        local_path: str, specifies the local location of dataset to be downloaded.
            Default: `.`.
    Returns:
        dataset_path: str, the source location of dataset downloaded.
    '''
    download_func = download_checker.get(dataset_name)
    if download_func is None:
        print("Currently dataset_name only supports {}!".format(list(download_checker.keys())))
        sys.exit(0)

    return download_func(local_path)
