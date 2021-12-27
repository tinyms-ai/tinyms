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
from PIL import Image
import tinyms as ts

from . import _transform_ops
from ._transform_ops import *
from .utils import ssd_bboxes_encode, ssd_bboxes_filter, jaccard_numpy
from .. import Tensor
from ..data import MnistDataset, Cifar10Dataset, ImageFolderDataset, VOCDataset, GeneratorDataset
from ..primitives import Softmax
from .transform_config import get_specified_config

__all__ = [
    'mnist_transform', 'MnistTransform',
    'cifar10_transform', 'Cifar10Transform',
    'imagefolder_transform', 'ImageFolderTransform',
    'voc_transform', 'VOCTransform',
    'shanshui_tranform', 'ShanshuiTransform',
    'cyclegan_transform', 'CycleGanDatasetTransform'
]
__all__.extend(_transform_ops.__all__)


class DatasetTransform(object):
    r'''
    Base class for all dataset transforms.
    '''

    def __init__(self, configs=None):
        if configs:
            self.configs = configs
        else:
            self.configs = get_specified_config('DatasetTransform')
        self.labels = self.configs['labels']
        self.transform_strategy = self.configs['transform_strategy']

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
        r'''
        Apply postprocess operation for prediction result.

        Args:
            input (numpy.ndarray): Prediction result.
            strategy (str): Specifies the postprocess strategy. Default: TOP1_CLASS.

        Returns:
            str, the postprocess result.
        '''
        if not isinstance(input, np.ndarray):
            raise TypeError("Input should be NumPy, got {}.".format(type(input)))
        if not input.ndim == 2:
            raise TypeError("Input should be 2-D Numpy, got {}.".format(input.ndim))
        if strategy not in self.transform_strategy:
            raise ValueError("Strategy should be one of {}, got {}.".format(self.transform_strategy, strategy))

        softmax = Softmax()
        score_list = softmax(Tensor(input, dtype=ts.float32)).asnumpy()
        if strategy == 'TOP1_CLASS':
            score = max(score_list[0])
            return ('TOP1: {}, score: {}').format(str(self.labels[input[0].argmax()]), str(round(score, 5)))
        else:
            label_index = np.argsort(input[0])[::-1]
            score_index = np.sort(score_list[0])[::-1]
            top5_labels = []
            res = ''
            top5_scores = score_index[:5].tolist()
            top_num = int(strategy.split('_')[0].split('TOP')[-1])
            for i in range(top_num):
                top5_labels.append(self.labels[label_index[i]])
                res += 'TOP' + str(i+1) + ": " + str(top5_labels[i]) + \
                       ", score: " + str(format(top5_scores[i], '.7f')) + '\t'
            return res


class MnistTransform(DatasetTransform):
    r'''
    Mnist dataset transform class.

    Inputs:
        img (Union[numpy.ndarray, PIL.Image]): Image to be transformed in Mnist-style.

    Outputs:
        numpy.ndarray, transformed image.

    Examples:
        >>> from PIL import Image
        >>> from tinyms.vision import MnistTransform
        >>>
        >>> mnist_transform = MnistTransform()
        >>> img = Image.open('object_detection.jpg')
        >>> img = mnist_transform(img)
    '''

    def __init__(self, configs=None):
        if configs:
            self.configs = configs
        else:
            self.configs = get_specified_config('MnistTransform')
        super().__init__(configs=self.configs)
        self.labels = self.configs['labels']
        self.transform_strategy = self.configs['transform_strategy']
        self.grayscale = Grayscale()
        self.resize = Resize(self.configs['resize'])
        self.normalize = Rescale(eval(self.configs['rescale1']['rescale_factor']),
                                 eval(self.configs['rescale1']['shift_factor']))
        self.rescale = Rescale(eval(self.configs['rescale2']['rescale_factor']),
                               self.configs['rescale2']['shift_factor'])

    def __call__(self, img):
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input type should be numpy.ndarray or PIL.Image, got {}.".format(type(img)))
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
        img = np.asarray(self.grayscale(img), dtype=np.float32)
        img = np.expand_dims(img, 2)
        img = self.resize(img)
        img = self.normalize(img)
        img = self.rescale(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, mnist_ds, repeat_size=1, batch_size=32, num_parallel_workers=None):
        r'''
        Apply preprocess operation on MnistDataset instance.

        Args:
            mnist_ds (data.MnistDataset): MnistDataset instance.
            repeat_size (int): The repeat size of dataset. Default: 1.
            batch_size (int): Batch size. Default: 32.
            num_parallel_workers (int): The number of concurrent workers. Default: None.

        Returns:
            data.MnistDataset, the preprocessed MnistDataset instance.

        Examples:
            >>> from tinyms.vision import MnistTransform
            >>>
            >>> mnist_transform = MnistTransform()
            >>> mnist_ds = mnist_transform.apply_ds(mnist_ds)
        '''
        if not isinstance(mnist_ds, MnistDataset):
            raise TypeError("Input type should be MnistDataset, got {}.".format(type(mnist_ds)))

        trans_func = [self.resize, self.normalize, self.rescale, hwc2chw]
        # apply transform functions on mnist dataset
        mnist_ds = super().apply_ds(mnist_ds, trans_func=trans_func, repeat_size=repeat_size,
                                    batch_size=batch_size, num_parallel_workers=num_parallel_workers)

        return mnist_ds


class Cifar10Transform(DatasetTransform):
    r'''
    Cifar10 dataset transform class.

    Inputs:
        img (Union[numpy.ndarray, PIL.Image]): Image to be transformed in Cifar10-style.

    Outputs:
        numpy.ndarray, Transformed image.

    Examples:
        >>> from PIL import Image
        >>> from tinyms.vision import Cifar10Transform
        >>>
        >>> cifar10_transform = Cifar10Transform()
        >>> img = Image.open('object_detection.jpg')
        >>> img = cifar10_transform(img)
    """
    '''

    def __init__(self, configs=None):
        if configs:
            self.configs = configs
        else:
            self.configs = get_specified_config('Cifar10Transform')
        super().__init__(configs=self.configs)
        self.random_crop = RandomCrop(self.configs['random_crop']['size'], self.configs['random_crop']['padding'])
        self.random_horizontal_flip = RandomHorizontalFlip(prob=self.configs['random_horizontal_flip']['prob'])
        self.resize = Resize(self.configs['resize'])
        self.rescale = Rescale(eval(self.configs['rescale']['rescale_factor']),
                               self.configs['rescale']['shift_factor'])
        self.normalize = Normalize(self.configs['normalize']['mean'], self.configs['normalize']['std'])

    def __call__(self, img):
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input type should be numpy.ndarray or PIL.Image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self.rescale(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, cifar10_ds, repeat_size=1, batch_size=32,
                 num_parallel_workers=None, is_training=True):
        r'''
        Apply preprocess operation on Cifar10Dataset instance.

        Args:
            cifar10_ds (data.Cifar10Dataset): Cifar10Dataset instance.
            repeat_size (int): The repeat size of dataset. Default: 1.
            batch_size (int): Batch size. Default: 32.
            num_parallel_workers (int): The number of concurrent workers. Default: None.
            is_training (bool): Specifies if is in training step. Default: True.

        Returns:
            data.Cifar10Dataset, the preprocessed Cifar10Dataset instance.

        Examples:
            >>> from tinyms.vision import Cifar10Transform
            >>>
            >>> cifar10_transform = Cifar10Transform()
            >>> cifar10_ds = cifar10_transform.apply_ds(cifar10_ds)
        '''
        if not isinstance(cifar10_ds, Cifar10Dataset):
            raise TypeError("Input type should be Cifar10Dataset, got {}.".format(type(cifar10_ds)))

        trans_func = []
        if is_training:
            trans_func += [self.random_crop, self.random_horizontal_flip]
        trans_func += [self.resize, self.rescale, self.normalize, hwc2chw]
        # apply transform functions on cifar10 dataset
        cifar10_ds = super().apply_ds(cifar10_ds, trans_func=trans_func, repeat_size=repeat_size,
                                      batch_size=batch_size, num_parallel_workers=num_parallel_workers)

        return cifar10_ds


class ImageFolderTransform(DatasetTransform):
    r'''
    ImageFolder dataset transform class.

    Inputs:
        img (Union[numpy.ndarray, PIL.Image]): Image to be transformed in ImageFolder-style.

    Outputs:
        numpy.ndarray, transformed image.

    Examples:
        >>> from PIL import Image
        >>> from tinyms.vision import ImageFolderTransform
        >>>
        >>> imagefolder_transform = ImageFolderTransform()
        >>> img = Image.open('object_detection.jpg')
        >>> img = imagefolder_transform(img)
    '''

    def __init__(self, configs=None):
        if configs:
            self.configs = configs
        else:
            self.configs = get_specified_config('ImageFolderTransform')
        super().__init__(configs=self.configs)
        self.random_crop_decode_resize = RandomCropDecodeResize(
            self.configs['random_crop_decode_resize']['size'],
            scale=tuple(self.configs['random_crop_decode_resize']['scale']),
            ratio=tuple(self.configs['random_crop_decode_resize']['ratio']))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=self.configs['random_horizontal_flip']['prob'])
        self.resize = Resize(self.configs['resize'])
        self.center_crop = CenterCrop(self.configs['center_crop'])
        self.normalize = Normalize(self.configs['normalize']['mean'], self.configs['normalize']['std'])

    def _center_crop(self, img):
        y, x, _ = img.shape
        startx = x // 2 - (224 // 2)
        starty = y // 2 - (224 // 2)
        return img[starty:starty + 224, startx:startx + 224, :]

    def __call__(self, img):
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input type should be numpy.ndarray or PIL.Image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self._center_crop(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, imagefolder_ds, repeat_size=1, batch_size=32,
                 num_parallel_workers=None, is_training=True):
        r'''
        Apply preprocess operation on ImageFolderDataset instance.

        Args:
            cifar10_ds (data.ImageFolderDataset): ImageFolderDataset instance.
            repeat_size (int): The repeat size of dataset. Default: 1.
            batch_size (int): Batch size. Default: 32.
            num_parallel_workers (int): The number of concurrent workers. Default: None.
            is_training (bool): Specifies if is in training step. Default: True.

        Returns:
            data.ImageFolderDataset, the preprocessed ImageFolderDataset instance.

        Examples:
            >>> from tinyms.vision import ImageFolderTransform
            >>>
            >>> imagefolder_transform = ImageFolderTransform()
            >>> imagefolder_ds = imagefolder_transform.apply_ds(imagefolder_ds)
        '''
        if not isinstance(imagefolder_ds, ImageFolderDataset):
            raise TypeError("Input type should be ImageFolderDataset, got {}.".format(type(imagefolder_ds)))

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
    r'''
    VOC dataset transform class.

    Inputs:
        img (Union[numpy.ndarray, PIL.Image]): Image to be transformed in VOC-style.

    Outputs:
        numpy.ndarray, transformed image.

    Examples:
        >>> from PIL import Image
        >>> from tinyms.vision import VOCTransform
        >>>
        >>> voc_transform = VOCTransform()
        >>> img = Image.open('object_detection.jpg')
        >>> img = voc_transform(img)
    '''

    def __init__(self, configs=None):
        if configs:
            self.configs = configs
        else:
            self.configs = get_specified_config('VOCTransform')
        super().__init__(configs=self.configs)
        self.resize = Resize(self.configs['resize'])
        self.horizontal_flip = PILRandomHorizontalFlip(self.configs['horizontal_flip'])
        self.normalize = Normalize(self.configs['normalize']['mean'], self.configs['normalize']['std'])
        self.random_color_adjust = RandomColorAdjust(brightness=self.configs['random_color_adjust']['brightness'],
                                                     contrast=self.configs['random_color_adjust']['contrast'],
                                                     saturation=self.configs['random_color_adjust']['saturation'])

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
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input type should be numpy.ndarray or PIL.Image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, voc_ds, repeat_size=1, batch_size=32,
                 num_parallel_workers=None, is_training=True):
        r'''
        Apply preprocess operation on VOCDataset instance.

        Args:
            voc_ds (data.VOCDataset): VOCDataset instance.
            repeat_size (int): The repeat size of dataset. Default: 1.
            batch_size (int): Batch size. Default: 32.
            num_parallel_workers (int): The number of concurrent workers. Default: None.
            is_training (bool): Specifies if is in training step. Default: True.

        Returns:
            data.VOCDataset, the preprocessed VOCDataset instance.

        Examples:
            >>> from tinyms.vision import VOCTransform
            >>>
            >>> VOC_transform = VOCTransform()
            >>> voc_ds = voc_transform.apply_ds(voc_ds)
        '''
        if not isinstance(voc_ds, VOCDataset):
            raise TypeError("Input type should be VOCDataset, got {}.".format(type(voc_ds)))

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
        r'''
        Apply postprocess operation for prediction result.

        Args:
            input (numpy.ndarray): Prediction result.
            image_shape (tuple): Image shape.
            strategy (str): Specifies the postprocess strategy. Default: TOP1_CLASS.

        Returns:
            dict, the postprocess result.
        '''
        if not isinstance(input, np.ndarray):
            raise TypeError("Input type should be numpy.ndarray, got {}.".format(type(input)))
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


class ShanshuiTransform(VOCTransform):
    r'''
    Shanshui dataset transform class.

    Inputs:
        img (Union[numpy.ndarray, PIL.Image]): Image to be transformed in VOC-style.

    Outputs:
        numpy.ndarray, transformed image.

    Examples:
        >>> from PIL import Image
        >>> from tinyms.vision import ShanshuiTransform
        >>>
        >>> shanshui_transform = ShanshuiTransform()
        >>> img = Image.open('object_detection.jpg')
        >>> img = shanshui_transform(img)
    '''

    def __init__(self, configs=None):
        if configs:
            self.configs = configs
        else:
            self.configs = get_specified_config('ShanshuiTransform')
        super().__init__(configs=self.configs)

    def __call__(self, img):
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input type should be numpy.ndarray or PIL.Image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self.normalize(img)
        img = hwc2chw(img)
        return img


class CycleGanDatasetTransform():
    r'''
    CycleGan dataset transform class.

    Inputs:
        img (Union[numpy.ndarray, PIL.Image]): Image to be transformed in city_scape.

    Outputs:
        numpy.ndarray, transformed image.

    Examples:
        >>> from PIL import Image
        >>> from tinyms.vision import CycleGanDatasetTransform
        >>>
        >>> cyclegan_transform = CycleGanDatasetTransform()
        >>> img = Image.open('object_detection.jpg')
        >>> img = cyclegan_transform(img)
    '''

    def __init__(self, configs=None):
        if configs:
            self.configs = configs
        else:
            self.configs = get_specified_config('CycleGanDatasetTransform')
        self.random_resized_crop = RandomResizedCrop(self.configs['random_resized_crop']['size'],
                                                     scale=tuple(self.configs['random_resized_crop']['scale']),
                                                     ratio=tuple(self.configs['random_resized_crop']['ratio']))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=self.configs['random_horizontal_flip']['prob'])
        self.resize = Resize(self.configs['resize'])
        self.normalize = Normalize(mean=self.configs['normalize']['mean'], std=self.configs['normalize']['std'])

    def __call__(self, img):
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input type should be numpy.ndarray or PIL.Image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, gan_generator_ds, repeat_size=1, batch_size=1,
                 num_parallel_workers=1, shuffle=True, phase='train'):
        r'''
        Apply preprocess operation on GeneratorDataset instance.

        Args:
            gan_generator_ds (data.GeneratorDataset): GeneratorDataset instance.
            repeat_size (int): The repeat size of dataset. Default: 1.
            batch_size (int): Batch size. Default: 32.
            num_parallel_workers (int): The number of concurrent workers. Default: 1.
            shuffle (bool): Specifies if applying shuffle operation. Default: True.
            phase (str): Specifies the current phase. Default: train.

        Returns:
            data.GeneratorDataset, the preprocessed GeneratorDataset instance.

        Examples:
            >>> from tinyms.vision import CycleGanDatasetTransform
            >>>
            >>> cyclegan_transform = CycleGanDatasetTransform()
            >>> gan_generator_ds = cyclegan_transform.apply_ds(gan_generator_ds)

        Raises:
            TypeError: If `gan_generator_ds` is not instance of GeneratorDataset.
        '''
        if not isinstance(gan_generator_ds, GeneratorDataset):
            raise TypeError("Input type should be GeneratorDataset, got {}.".format(type(gan_generator_ds)))

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
shanshui_tranform = ShanshuiTransform()
cyclegan_transform = CycleGanDatasetTransform()