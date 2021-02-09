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

import tinyms as ts
from tinyms import context, layers
from tinyms.model import Model, lenet5, resnet50, mobilenet_v2


def test_sequential():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    net = layers.SequentialLayer([
        layers.Conv2d(1, 6, 5, pad_mode='valid', weight_init="ones"),
        layers.ReLU(),
        layers.MaxPool2d(kernel_size=2, stride=2)
    ])
    model = Model(net)
    model.compile()
    z = model.predict(ts.ones((1, 1, 32, 32)))
    print(z.asnumpy())


def test_lenet5():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    model = Model(lenet5())
    model.compile()
    z = model.predict(ts.ones((1, 1, 32, 32)))
    print(z.asnumpy())


def test_resnet50():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    model = Model(resnet50())
    model.compile()
    z = model.predict(ts.ones((1, 3, 224, 224)))
    print(z.asnumpy())


def test_mobilenet_v2():
    context.set_context(mode=context.GRAPH_MODE, device_target="CPU")

    model = Model(mobilenet_v2())
    model.compile()
    z = model.predict(ts.ones((1, 3, 224, 224)))
    print(z.asnumpy())
