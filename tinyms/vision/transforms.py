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

import cv2
import numpy as np
import tinyms as ts
from PIL import Image

from . import _transform_ops
from ._transform_ops import *
from ..data import MnistDataset, Cifar10Dataset, ImageFolderDataset
from ..data.transforms import TypeCast

__all__ = [
    'mnist_transform', 'MnistTransform',
    'cifar10_transform', 'Cifar10Transform',
    'imagefolder_transform', 'ImageFolderTransform',
]
__all__.extend(_transform_ops.__all__)


class MnistTransform():
    def __init__(self):
        self.resize = Resize((32, 32))
        self.normalize = Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081)
        self.rescale = Rescale(1.0 / 255.0, 0.0)
        self.type_cast = TypeCast(ts.int32)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image to be transformed in Mnist-style.

        Returns:
            img (NumPy), Transformed image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        if not img.ndim == 2:
            raise TypeError("Input should be 2-D Numpy, got {}.".format(img.ndim))
        img = np.expand_dims(img, 2)
        img = self.resize(img)
        img = self.normalize(img)
        img = self.rescale(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, mnist_ds, repeat_size=1, batch_size=32, num_parallel_workers=None):
        if not isinstance(mnist_ds, MnistDataset):
            raise TypeError("Input should be MnistDataset, got {}.".format(type(mnist_ds)))

        c_trans = [self.resize, self.normalize, self.rescale, hwc2chw]
        # apply map operations on images
        mnist_ds = mnist_ds.map(operations=self.type_cast, input_columns="label",
                                num_parallel_workers=num_parallel_workers)
        mnist_ds = mnist_ds.map(operations=c_trans, input_columns="image", num_parallel_workers=num_parallel_workers)
        # apply batch operations
        mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
        # apply repeat operations
        mnist_ds = mnist_ds.repeat(repeat_size)

        return mnist_ds


class Cifar10Transform():
    def __init__(self):
        self.random_crop = RandomCrop((32, 32), (4, 4, 4, 4))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5)
        self.resize = Resize((224, 224))
        self.rescale = Rescale(1.0 / 255.0, 0.0)
        self.normalize = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
        self.type_cast = TypeCast(ts.int32)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image to be transformed in Cifar10-style.

        Returns:
            img (NumPy), Transformed image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self.rescale(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, cifar10_ds, repeat_size=1, batch_size=32,
                 num_parallel_workers=None, training=True):
        if not isinstance(cifar10_ds, Cifar10Dataset):
            raise TypeError("Input should be Cifar10Dataset, got {}.".format(type(cifar10_ds)))

        c_trans = []
        if training:
            c_trans += [self.random_crop, self.random_horizontal_flip]
        c_trans += [self.resize, self.rescale, self.normalize, hwc2chw]
        # apply map operations on images
        cifar10_ds = cifar10_ds.map(operations=self.type_cast, input_columns="label",
                                    num_parallel_workers=num_parallel_workers)
        cifar10_ds = cifar10_ds.map(operations=c_trans, input_columns="image",
                                    num_parallel_workers=num_parallel_workers)
        # apply batch operations
        cifar10_ds = cifar10_ds.batch(batch_size, drop_remainder=True)
        # apply repeat operations
        cifar10_ds = cifar10_ds.repeat(repeat_size)

        return cifar10_ds


class ImageFolderTransform():
    def __init__(self):
        self.random_crop_decode_resize = RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5)
        self.resize = Resize(256)
        self.center_crop = CenterCrop(224)
        self.normalize = Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255],
                                   [0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.type_cast = TypeCast(ts.int32)

    def _center_crop(self, img, cropx, cropy):
        y, x, _ = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty:starty + cropy, startx:startx + cropx, :]

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image to be transformed in ImageFolder-style.

        Returns:
            img (NumPy), Transformed image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self._center_crop(img, 224, 224)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, imagefolder_ds, repeat_size=1, batch_size=32,
                 num_parallel_workers=None, training=True):
        if not isinstance(imagefolder_ds, ImageFolderDataset):
            raise TypeError("Input should be ImageFolderDataset, got {}.".format(type(imagefolder_ds)))

        if training:
            c_trans = [self.random_crop_decode_resize, self.random_horizontal_flip]
        else:
            c_trans = [decode, self.resize, self.center_crop]
        c_trans += [self.normalize, hwc2chw]
        # apply map operations on images
        imagefolder_ds = imagefolder_ds.map(operations=self.type_cast, input_columns="label",
                                            num_parallel_workers=num_parallel_workers)
        imagefolder_ds = imagefolder_ds.map(operations=c_trans, input_columns="image",
                                            num_parallel_workers=num_parallel_workers)
        # apply batch operations
        imagefolder_ds = imagefolder_ds.batch(batch_size, drop_remainder=True)
        # apply repeat operations
        imagefolder_ds = imagefolder_ds.repeat(repeat_size)

        return imagefolder_ds


mnist_transform = MnistTransform()
cifar10_transform = Cifar10Transform()
imagefolder_transform = ImageFolderTransform()
