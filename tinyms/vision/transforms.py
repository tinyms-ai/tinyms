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
import tinyms as ts
from PIL import Image
from tinyms.primitives import Softmax

from . import _transform_ops
from ._transform_ops import *
from ..data import MnistDataset, Cifar10Dataset, ImageFolderDataset, GeneratorDataset

__all__ = [
    'mnist_transform', 'MnistTransform',
    'cifar10_transform', 'Cifar10Transform',
    'imagefolder_transform', 'ImageFolderTransform',
    'cyclegan_transform', 'CycleGanDatasetTransform',
]
__all__.extend(_transform_ops.__all__)


class DatasetTransform():
    def __init__(self, labels=None):
        self.labels = labels
        self.transform_strategy = ['TOP1_CLASS', 'TOP5_CLASS']

    def apply_ds(self, ds, trans_func=None, repeat_size=1, batch_size=32,
                 num_parallel_workers=None):
        if not isinstance(trans_func, list):
            raise TypeError('trans_func must be list')

        # apply map operations on datasets
        ds = ds.map(operations=TypeCast(ts.int32), input_columns="label",
                    num_parallel_workers=num_parallel_workers)
        ds = ds.map(operations=trans_func, input_columns="image", num_parallel_workers=num_parallel_workers)
        # apply batch operations
        ds = ds.batch(batch_size, drop_remainder=True)
        # apply repeat operations
        ds = ds.repeat(repeat_size)

        return ds

    def postprocess(self, input, strategy='TOP1_CLASS'):
        if not isinstance(input, np.ndarray):
            raise TypeError("Input should be NumPy, got {}.".format(type(input)))
        if not input.ndim == 2:
            raise TypeError("Input should be 2-D Numpy, got {}.".format(input.ndim))
        if strategy not in self.transform_strategy:
            raise ValueError("Strategy should be one of {}, got {}.".format(self.transform_strategy, strategy))
        
        softmax = Softmax()
        score_list = softmax(ts.array(input)).asnumpy()
        if strategy == 'TOP1_CLASS':
            score = max(score_list[0])
            return ('TOP1: '+ str(self.labels[input[0].argmax()]) + ', score: ' + str(format(score, '.20f')))
        else:
            label_index = np.argsort(input[0])[::-1]
            score_index = np.sort(score_list[0])[::-1]
            top5_labels = []
            res = ''
            top5_scores = score_index[:5].tolist()
            for i in range(5):
                top5_labels.append(self.labels[label_index[i]])  
                res += 'TOP' + str(i+1) + ": " + str(top5_labels[i]) + ", score: " + str(format(top5_scores[i], '.20f')) + '\n'
            return res


class MnistTransform(DatasetTransform):
    def __init__(self):
        labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        super().__init__(labels=labels)
        self.grayscale = Grayscale()
        self.resize = Resize((32, 32))
        self.normalize = Rescale(1 / 0.3081, -1 * 0.1307 / 0.3081)
        self.rescale = Rescale(1.0 / 255.0, 0.0)

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
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode='RGB')
        img = np.asarray(self.grayscale(img), dtype=np.float32)
        img = np.expand_dims(img, 2)
        img = self.resize(img)
        img = self.normalize(img)
        img = self.rescale(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, mnist_ds, repeat_size=1, batch_size=32, num_parallel_workers=None):
        if not isinstance(mnist_ds, MnistDataset):
            raise TypeError("Input should be MnistDataset, got {}.".format(type(mnist_ds)))

        trans_func = [self.resize, self.normalize, self.rescale, hwc2chw]
        # apply transform functions on mnist dataset
        mnist_ds = super().apply_ds(mnist_ds, trans_func=trans_func, repeat_size=repeat_size,
                                    batch_size=batch_size, num_parallel_workers=num_parallel_workers)

        return mnist_ds


class Cifar10Transform(DatasetTransform):
    def __init__(self):
        labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                  'dog', 'frog', 'horse', 'ship', 'truck']
        super().__init__(labels=labels)
        self.random_crop = RandomCrop((32, 32), (4, 4, 4, 4))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5)
        self.resize = Resize((224, 224))
        self.rescale = Rescale(1.0 / 255.0, 0.0)
        self.normalize = Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])

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

        trans_func = []
        if training:
            trans_func += [self.random_crop, self.random_horizontal_flip]
        trans_func += [self.resize, self.rescale, self.normalize, hwc2chw]
        # apply transform functions on cifar10 dataset
        cifar10_ds = super().apply_ds(cifar10_ds, trans_func=trans_func, repeat_size=repeat_size,
                                      batch_size=batch_size, num_parallel_workers=num_parallel_workers)

        return cifar10_ds


class ImageFolderTransform(DatasetTransform):
    def __init__(self):
        labels = ["Agaricus双孢蘑菇,伞菌目,蘑菇科,蘑菇属,广泛分布于北半球温带,无毒",
                  "Amanita毒蝇伞,伞菌目,鹅膏菌科,鹅膏菌属,主要分布于我国黑龙江、吉林、四川、西藏、云南等地,有毒",
                  "Boletus丽柄牛肝菌,伞菌目,牛肝菌科,牛肝菌属,分布于云南、陕西、甘肃、西藏等地,有毒",
                  "Cortinarius掷丝膜菌,伞菌目,丝膜菌科,丝膜菌属,分布于湖南等地(夏秋季在山毛等阔叶林地上生长)",
                  "Entoloma霍氏粉褶菌,伞菌目,粉褶菌科,粉褶菌属,主要分布于新西兰北岛和南岛西部,有毒",
                  "Hygrocybe浅黄褐湿伞,伞菌目,蜡伞科,湿伞属,分布于香港(见于松仔园),有毒",
                  "Lactarius松乳菇,红菇目,红菇科,乳菇属,广泛分布于亚热带松林地,无毒",
                  "Russula褪色红菇,伞菌目,红菇科,红菇属,分布于河北、吉林、四川、江苏、西藏等地,无毒",
                  "Suillus乳牛肝菌,牛肝菌目,乳牛肝菌科,乳牛肝菌属,分布于吉林、辽宁、山西、安徽、江西、浙江、湖南、四川、贵州等地,无毒",
                  ]
        super().__init__(labels=labels)
        self.random_crop_decode_resize = RandomCropDecodeResize(224, scale=(0.08, 1.0), ratio=(0.75, 1.333))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5)
        self.resize = Resize(256)
        self.center_crop = CenterCrop(224)
        self.normalize = Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255],
                                   [0.229 * 255, 0.224 * 255, 0.225 * 255])

    def _center_crop(self, img):
        y, x, _ = img.shape
        startx = x // 2 - (224 // 2)
        starty = y // 2 - (224 // 2)
        return img[starty:starty + 224, startx:startx + 224, :]

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
        img = self._center_crop(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, imagefolder_ds, repeat_size=1, batch_size=32,
                 num_parallel_workers=None, training=True):
        if not isinstance(imagefolder_ds, ImageFolderDataset):
            raise TypeError("Input should be ImageFolderDataset, got {}.".format(type(imagefolder_ds)))

        if training:
            trans_func = [self.random_crop_decode_resize, self.random_horizontal_flip]
        else:
            trans_func = [decode, self.resize, self.center_crop]
        trans_func += [self.normalize, hwc2chw]
        # apply transform functions on imagefolder dataset
        imagefolder_ds = super().apply_ds(imagefolder_ds, trans_func=trans_func, repeat_size=repeat_size,
                                          batch_size=batch_size, num_parallel_workers=num_parallel_workers)

        return imagefolder_ds


class CycleGanDatasetTransform():
    def __init__(self):
        self.random_resized_crop = RandomResizedCrop(256, scale=(0.5, 1.0), ratio=(0.75, 1.333))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5)
        self.resize = Resize((256, 256))
        self.normalize = Normalize(mean=[0.5 * 255] * 3, std=[0.5 * 255] * 3)

    def __call__(self, img):
        """
        Call method.

        Args:
            img (NumPy or PIL image): Image to be transformed in city_scape.

        Returns:
            img (NumPy), Transformed image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        img = self.random_resized_crop(img)
        img = self.random_horizontal_flip(img)
        img = self.resize(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, gan_generator_ds, repeat_size=1, batch_size=1,
                 num_parallel_workers=1, shuffle=True, phase='train'):
        if not isinstance(gan_generator_ds, GeneratorDataset):
            raise TypeError("Input should be GeneratorDataset, got {}.".format(type(gan_generator_ds)))

        trans_func = []
        if phase == 'train':
            if shuffle:
                trans_func += [self.random_resized_crop, self.random_horizontal_flip, self.normalize, hwc2chw]
            else:
                trans_func += [self.resize, self.normalize, hwc2chw]

            # apply transform functions on gan_generator_ds dataset
            gan_generator_ds = gan_generator_ds.map(operations=trans_func,
                                                    input_columns=["image_A"],
                                                    num_parallel_workers=num_parallel_workers)
            gan_generator_ds = gan_generator_ds.map(operations=trans_func,
                                                    input_columns=["image_B"],
                                                    num_parallel_workers=num_parallel_workers)
        else:
            trans_func += [self.resize, self.normalize, hwc2chw]
            gan_generator_ds = gan_generator_ds.map(operations=trans_func,
                                                    input_columns=["image"],
                                                    num_parallel_workers=num_parallel_workers)
        gan_generator_ds = gan_generator_ds.batch(batch_size, drop_remainder=True)
        gan_generator_ds = gan_generator_ds.repeat(repeat_size)
        return gan_generator_ds


mnist_transform = MnistTransform()
cifar10_transform = Cifar10Transform()
imagefolder_transform = ImageFolderTransform()
cyclegan_transform = CycleGanDatasetTransform()

