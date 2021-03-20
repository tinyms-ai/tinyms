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

import os
import numpy as np
import tinyms as ts
from mindspore.train.serialization import load_checkpoint, load_param_into_net
from tinyms import Tensor


def cyclegan_predict(G_generator, input_data, ckpt_path):
    G_generator.set_train(True)
    # load checkpoint
    if not os.path.isfile(ckpt_path):
        err_msg = "The model path " + ckpt_path + " not exist!"
        raise ValueError(err_msg)
    param_G = load_checkpoint(ckpt_path)
    load_param_into_net(G_generator, param_G)
    fake_img = G_generator(ts.expand_dims(input_data, 0))
    if isinstance(fake_img, Tensor):
        # Decode a [1, C, H, W] Tensor to image numpy array.
        mean = 0.5 * 255
        std = 0.5 * 255
        fake_img = (fake_img.asnumpy()[0] * std + mean).astype(np.uint8).transpose((1, 2, 0))
    elif not isinstance(fake_img, np.ndarray):
        raise ValueError("img should be Tensor or numpy array, but get {}".format(type(fake_img)))
    return fake_img
