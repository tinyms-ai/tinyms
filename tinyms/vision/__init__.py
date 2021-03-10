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
"""
This module is to support vision augmentations. transforms is a high performance
image augmentation module which is developed with C++ OpenCV.
"""
from mindspore.dataset.vision.utils import Inter, Border, ImageBatchFormat
from . import transforms
from .transforms import *
from .view import ImageViewer
from .utils import ssd_bboxes_encode, ssd_bboxes_filter, coco_eval

vision_utils = ['Inter', 'Border', 'ImageBatchFormat']
bbox_utils = [
    'ssd_bboxes_encode',
    'ssd_bboxes_filter',
    'coco_eval',
]

__all__ = vision_utils + bbox_utils
__all__.extend(transforms.__all__)
