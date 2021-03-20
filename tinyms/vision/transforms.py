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
from .utils import ssd_bboxes_encode, ssd_bboxes_filter, jaccard_numpy
from ..data import MnistDataset, Cifar10Dataset, ImageFolderDataset, VOCDataset, GeneratorDataset

__all__ = [
    'mnist_transform', 'MnistTransform',
    'cifar10_transform', 'Cifar10Transform',
    'imagefolder_transform', 'ImageFolderTransform',
    'voc_transform', 'VOCTransform',
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
            return ('TOP1: ' + str(self.labels[input[0].argmax()]) + ', score: ' + str(format(score, '.20f')))
        else:
            label_index = np.argsort(input[0])[::-1]
            score_index = np.sort(score_list[0])[::-1]
            top5_labels = []
            res = ''
            top5_scores = score_index[:5].tolist()
            for i in range(5):
                top5_labels.append(self.labels[label_index[i]])
                res += 'TOP' + str(i+1) + ": " + str(top5_labels[i]) + \
                    ", score: " + str(format(top5_scores[i], '.20f')) + '\n'
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
        Call method for model prediction.

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
        Call method for model prediction.

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
                 num_parallel_workers=None, is_training=True):
        if not isinstance(cifar10_ds, Cifar10Dataset):
            raise TypeError("Input should be Cifar10Dataset, got {}.".format(type(cifar10_ds)))

        trans_func = []
        if is_training:
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
        self.random_crop_decode_resize = RandomCropDecodeResize((224, 224), scale=(0.08, 1.0), ratio=(0.75, 1.333))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5)
        self.resize = Resize((256, 256))
        self.center_crop = CenterCrop((224, 224))
        self.normalize = Normalize([0.485 * 255, 0.456 * 255, 0.406 * 255],
                                   [0.229 * 255, 0.224 * 255, 0.225 * 255])

    def _center_crop(self, img):
        y, x, _ = img.shape
        startx = x // 2 - (224 // 2)
        starty = y // 2 - (224 // 2)
        return img[starty:starty + 224, startx:startx + 224, :]

    def __call__(self, img):
        """
        Call method for model prediction.

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
                 num_parallel_workers=None, is_training=True):
        if not isinstance(imagefolder_ds, ImageFolderDataset):
            raise TypeError("Input should be ImageFolderDataset, got {}.".format(type(imagefolder_ds)))

        if is_training:
            trans_func = [self.random_crop_decode_resize, self.random_horizontal_flip]
        else:
            trans_func = [decode, self.resize, self.center_crop]
        trans_func += [self.normalize, hwc2chw]
        # apply transform functions on imagefolder dataset
        imagefolder_ds = super().apply_ds(imagefolder_ds, trans_func=trans_func, repeat_size=repeat_size,
                                          batch_size=batch_size, num_parallel_workers=num_parallel_workers)

        return imagefolder_ds


def _rand(a=0., b=1.):
    """Generate random."""
    return np.random.rand() * (b - a) + a


class VOCTransform(DatasetTransform):
    def __init__(self):
        labels = ['background',
                  'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                  'bus', 'car', 'cat', 'chair', 'cow',
                  'diningtable', 'dog', 'horse', 'motorbike', 'person',
                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        super().__init__(labels=labels)
        self.resize = Resize((300, 300))
        self.horizontal_flip = PILRandomHorizontalFlip(1.0)
        self.normalize = Normalize(mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                   std=[0.229 * 255, 0.224 * 255, 0.225 * 255])
        self.random_color_adjust = RandomColorAdjust(brightness=0.4, contrast=0.4, saturation=0.4)

    def _preprocess_fn(self, image, boxes, labels, is_training=True):
        """Preprocess function for voc dataset."""
        def _random_sample_crop(image, boxes):
            """Random Crop the image and boxes"""
            height, width, _ = image.shape
            min_iou = np.random.choice([None, 0.1, 0.3, 0.5, 0.7, 0.9])
            if min_iou is None:
                return image, boxes
            # max trails (50)
            for _ in range(50):
                image_t = image
                w = _rand(0.3, 1.0) * width
                h = _rand(0.3, 1.0) * height
                # aspect ratio constraint b/t .5 & 2
                if h / w < 0.5 or h / w > 2:
                    continue
                left = _rand() * (width - w)
                top = _rand() * (height - h)
                rect = np.array([int(top), int(left), int(top + h), int(left + w)])
                overlap = jaccard_numpy(boxes, rect)
                # dropout some boxes
                drop_mask = overlap > 0
                if not drop_mask.any():
                    continue
                if overlap[drop_mask].min() < min_iou and overlap[drop_mask].max() > (min_iou + 0.2):
                    continue
                image_t = image_t[rect[0]:rect[2], rect[1]:rect[3], :]
                centers = (boxes[:, :2] + boxes[:, 2:4]) / 2.0
                m1 = (rect[0] < centers[:, 0]) * (rect[1] < centers[:, 1])
                m2 = (rect[2] > centers[:, 0]) * (rect[3] > centers[:, 1])
                # mask in that both m1 and m2 are true
                mask = m1 * m2 * drop_mask
                # have any valid boxes? try again if not
                if not mask.any():
                    continue
                # take only matching gt boxes
                boxes_t = boxes[mask, :].copy()
                boxes_t[:, :2] = np.maximum(boxes_t[:, :2], rect[:2])
                boxes_t[:, :2] -= rect[:2]
                boxes_t[:, 2:4] = np.minimum(boxes_t[:, 2:4], rect[2:4])
                boxes_t[:, 2:4] -= rect[:2]
                return image_t, boxes_t
            return image, boxes

        # Only perform resize operation of data evaluation step
        if not is_training:
            img_h, img_w, _ = image.shape
            image = self.resize(image)
            return image, np.array((img_h, img_w), dtype=np.float32), labels
        # Merge [x, y, w, h] and cls to [x, y, w, h, cls]
        boxes = np.hstack((boxes, labels)).astype(np.float32)
        # Change [x, y, w, h, cls] to [ymin, xmin, ymax, xmax, cls]
        boxes_yxyx = np.zeros_like(boxes)
        boxes_yxyx[:, 4] = boxes[:, 4]
        boxes_yxyx[:, [1, 0]] = boxes[:, [0, 1]]
        boxes_yxyx[:, [3, 2]] = boxes[:, [0, 1]] + boxes[:, [2, 3]]
        # Random crop image and bbox
        image, boxes_yxyx = _random_sample_crop(image, boxes_yxyx)
        # Resize image and bbox
        ih, iw, _ = image.shape
        image = self.resize(image)
        boxes_yxyx[:, [0, 2]] = boxes_yxyx[:, [0, 2]] / ih
        boxes_yxyx[:, [1, 3]] = boxes_yxyx[:, [1, 3]] / iw
        # Flip image and bbox or not
        flip = _rand() < .5
        if flip:
            image = np.asarray(self.horizontal_flip(Image.fromarray(image, mode='RGB')))
            boxes_yxyx[:, [1, 3]] = 1 - boxes_yxyx[:, [3, 1]]
        # When the channels of image is 1
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
            image = np.concatenate([image, image, image], axis=-1)

        boxes_yxyx, label, num_match = ssd_bboxes_encode(boxes_yxyx)
        return image, boxes_yxyx, label, num_match

    def __call__(self, img):
        """
        Call method for model prediction.

        Args:
            img (NumPy or PIL image): Image to be transformed in VOC-style.

        Returns:
            img (NumPy), Transformed image.
        """
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, voc_ds, repeat_size=1, batch_size=32,
                 num_parallel_workers=None, is_training=True):
        if not isinstance(voc_ds, VOCDataset):
            raise TypeError("Input should be VOCDataset, got {}.".format(type(voc_ds)))

        compose_map_func = (lambda image, boxes, labels: self._preprocess_fn(image, boxes, labels, is_training))
        if is_training:
            output_columns = ["image", "bbox", "label", "num_match"]
            trans_func = [self.random_color_adjust, self.normalize, hwc2chw]
        else:
            output_columns = ["image", "image_shape", "label"]
            trans_func = [self.normalize, hwc2chw]
        # apply transform functions on voc dataset
        voc_ds = voc_ds.map(operations=compose_map_func,
                            input_columns=["image", "bbox", "label"],
                            output_columns=output_columns,
                            column_order=output_columns,
                            num_parallel_workers=num_parallel_workers)
        voc_ds = super().apply_ds(voc_ds, trans_func=trans_func, repeat_size=repeat_size,
                                  batch_size=batch_size, num_parallel_workers=num_parallel_workers)

        return voc_ds

    def postprocess(self, input, image_shape, strategy='TOP1_CLASS'):
        if not isinstance(input, np.ndarray):
            raise TypeError("Input should be NumPy, got {}.".format(type(input)))
        if not input.ndim == 3:
            raise TypeError("Input should be 3-D Numpy, got {}.".format(input.ndim))
        if not strategy == 'TOP1_CLASS':
            raise ValueError("Currently VOC transform only supports 'TOP1_CLASS' strategy!")

        pred_res = []
        pred_loc, pred_cls, pred_label = ssd_bboxes_filter(input[0, :, :4], input[0, :, 4:], image_shape)
        for loc, score, label in zip(pred_loc, pred_cls, pred_label):
            pred_res.append({
                'bbox': [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]],
                'score': score,
                'category_id': self.labels[label],
            })

        return pred_res


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
        if not isinstance(img, np.ndarray):
            raise TypeError("Input should be NumPy, got {}.".format(type(img)))
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
voc_transform = VOCTransform()
cyclegan_transform = CycleGanDatasetTransform()
