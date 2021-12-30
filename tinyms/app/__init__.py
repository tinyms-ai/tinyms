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
This module is to support vision visualization with opencv, which can help
developers use pre-trained models to predict and show the reasoning image fast.
Current it only supports object detection model.
"""
from . import object_detection
from .object_detection.object_detector import object_detection_predict, ObjectDetector
from .object_detection.utils.view_util import visualize_boxes_on_image, draw_boxes_on_image, save_image
from .object_detection.utils.config_util import load_and_parse_config


object_detection_utils = ['visualize_boxes_on_image', 'draw_boxes_on_image', 'save_image', 'load_and_parse_config']

__all__ = ['ObjectDetector', 'object_detection_predict']
__all__.extend(object_detection_utils)
__all__.extend(object_detection.__all__)
