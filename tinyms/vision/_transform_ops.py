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

from mindspore.dataset.vision.py_transforms import Grayscale, RandomHorizontalFlip as PILRandomHorizontalFlip
from mindspore.dataset.vision.c_transforms import *
from mindspore.dataset.transforms.c_transforms import *

vision_trans = [
    'AutoContrast',
    'BoundingBoxAugment',
    'CenterCrop',
    'CutMixBatch',
    'CutOut',
    'Decode',
    'Equalize',
    'Grayscale',
    'HWC2CHW',
    'Invert',
    'MixUpBatch',
    'Normalize',
    'Pad',
    'PILRandomHorizontalFlip',
    'RandomAffine',
    'RandomColor',
    'RandomColorAdjust',
    'RandomCrop',
    'RandomCropDecodeResize',
    'RandomCropWithBBox',
    'RandomHorizontalFlip',
    'RandomHorizontalFlipWithBBox',
    'RandomPosterize',
    'RandomResize',
    'RandomResizedCrop',
    'RandomResizedCropWithBBox',
    'RandomResizeWithBBox',
    'RandomRotation',
    'RandomSelectSubpolicy',
    'RandomSharpness',
    'RandomSolarize',
    'RandomVerticalFlip',
    'RandomVerticalFlipWithBBox',
    'Rescale',
    'Resize',
    'ResizeWithBBox',
    'SoftDvppDecodeRandomCropResizeJpeg',
    'SoftDvppDecodeResizeJpeg',
    'UniformAugment',
]

common_trans = [
    'Compose',
    'Concatenate',
    'Duplicate',
    'Fill',
    'Mask',
    'OneHot',
    'PadEnd',
    'RandomApply',
    'RandomChoice',
    'Slice',
    'TypeCast',
    'Unique',
]

__all__ = vision_trans + common_trans

decode = Decode()
hwc2chw = HWC2CHW()

__all__.extend([
    'decode',
    'hwc2chw',
])
