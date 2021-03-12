# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from tinyms.vision import ImageViewer


def test_imageviewer():
    fake_img = np.random.uniform(0.0, 1.0, size=[3, 224, 224])
    image_viewer = ImageViewer(fake_img, 'cat')

    assert np.all(image_viewer.image == fake_img)
    assert image_viewer.label == 'cat'
