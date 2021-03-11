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

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


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

    def __init__(self, image, label=None):
        if not isinstance(image, (np.ndarray, Image.Image)):
            raise TypeError("Input should be NumPy or PIL image, got {}.".format(type(image)))
        if isinstance(image, Image.Image):
            image = np.array(image)
        self._image = image
        self._label = label

    @property
    def image(self):
        return self._image

    @property
    def label(self):
        return self._label

    def show(self):
        plt.imshow(np.squeeze(self._image))
        plt.title("label: %s" % self._label)
        plt.show()

    def draw(self, pred_res, labels):
        colors = plt.cm.hsv(np.linspace(0, 1, len(labels)+1)).tolist()
        plt.figure(figsize=(20, 12))
        plt.imshow(np.squeeze(self._image))

        current_axis = plt.gca()

        for sample in pred_res:
            xmin = sample['bbox'][0]
            ymin = sample['bbox'][1]
            width = sample['bbox'][2]
            height = sample['bbox'][3]
            category_id = sample['category_id']
            cls = labels.index(category_id)
            color = colors[cls]
            
            label = '{}: {:.2f}'.format(category_id, sample['score'])
            current_axis.add_patch(plt.Rectangle((xmin, ymin), width, height,
                                                 color=color, fill=False, linewidth=2))
            current_axis.text(xmin, ymin, label, size='x-large', color='white',
                              bbox={'facecolor': color, 'alpha': 1.0})
