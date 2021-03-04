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
Layer.

The high-level components(Cells) used to construct the neural network.
"""
from mindspore.nn import Cell
from mindspore.nn.layer.container import SequentialCell, CellList
from mindspore.nn.layer import activation, normalization, conv, lstm, basic, \
    embedding, pooling, math as nn_math, combined
from mindspore.nn.layer.activation import *
from mindspore.nn.layer.normalization import *
from mindspore.nn.layer.conv import *
from mindspore.nn.layer.lstm import *
from mindspore.nn.layer.basic import *
from mindspore.nn.layer.embedding import *
from mindspore.nn.layer.pooling import *
from mindspore.nn.layer.math import *
from mindspore.nn.layer.combined import *

__all__ = ['Layer', 'SequentialLayer', 'LayerList']
__all__.extend(activation.__all__)
__all__.extend(normalization.__all__)
__all__.extend(conv.__all__)
__all__.extend(lstm.__all__)
__all__.extend(basic.__all__)
__all__.extend(embedding.__all__)
__all__.extend(pooling.__all__)
__all__.extend(nn_math.__all__)
__all__.extend(combined.__all__)


class Layer(Cell):
    pass


class SequentialLayer(SequentialCell):
    pass


class LayerList(CellList):
    pass
