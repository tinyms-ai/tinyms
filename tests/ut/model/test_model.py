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
import mindspore.nn as nn
import mindspore.context as context
from mindspore import Tensor

from tinyms import Model
from tinyms.layers import Sequential


def test_model_predict():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = Sequential([
        nn.Conv2d(1, 6, 5, pad_mode='valid', weight_init="ones"),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2)
    ])
    model = Model(net)
    model.compile()
    z = model.predict(Tensor(np.ones([1, 1, 28, 28]).astype(np.float32)))
    print(z.asnumpy())
