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
Layer module contains pre-defined building blocks or computing units to construct neural networks.

The high-level components (Layers) used to construct the neural network.
"""

from mindspore.nn import Cell, GraphCell
from mindspore.nn.layer.container import SequentialCell, CellList
from mindspore.nn import wrap
from mindspore.nn.wrap import *
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
__all__.extend(wrap.__all__)


class Layer(Cell):
    """
    Base class for all neural networks.

    A 'Layer' could be a single neural network layer, such as conv2d, relu, batch_norm, etc. or a composition of cells to constructing a network.

    Note:
        In general, the autograd algorithm will automatically generate the implementation of the gradient function, but if back-propagation(bprop) method is implemented, the gradient function will be replaced by the bprop.
        The bprop implementation will receive a Tensor `dout` containing the gradient of the loss w.r.t. the output, and a Tensor `out` containing the forward result.
        The bprop needs to compute the gradient of the loss w.r.t. the inputs, gradient of the loss w.r.t. Parameter variables are not supported currently.
        The bprop method must contain the self parameter.

    Args:
        auto_prefix (bool): Recursively generate namespaces. Default: True.

    Examples:
        >>> from tinyms import layers, primitives as P
        >>>
        >>> class MyNet(layers.Layer):
        ...    def __init__(self):
        ...        super(MyNet, self).__init__()
        ...        self.relu = P.ReLU()
        ...
        ...    def construct(self, x):
        ...        return self.relu(x)
    """
    pass


class GraphLayer(GraphCell):
    """
    Base class for running the graph loaded from MindIR.

    This feature is still under development. Currently `GraphLayer` do not support modifying the structure of the
    diagram, and can only use data that shape and type are the same as the input when exporting the MindIR.

    Args:
        graph (object): A compiled graph loaded from MindIR.

    Supported Platforms:
        ``Ascend`` ``GPU`` ``CPU``
    """
    pass


class SequentialLayer(SequentialCell):
    """
    Sequential layer container.

    A list of Layers will be added to it in the order they are passed in the constructor.
    Alternatively, an ordered dict of cells can also be passed in.

    Args:
        args (Union[list, OrderedDict]): List of subclass of Layer.

    Raises:
        TypeError: If the type of the argument is not list or OrderedDict.

    Inputs:
        - **input** (Tensor) - Tensor with shape according to the first Cell in the sequence.

    Outputs:
        Tensor, the output Tensor with shape depending on the input and defined sequence of Layers.

    Examples:
        >>> import tinyms as ts
        >>> from tinyms.layers import SequentialLayer, Conv2d, ReLU
        >>>
        >>> seq_layer = SequentialLayer([Conv2d(3, 2, 3, pad_mode='valid', weight_init="ones"), ReLU()])
        >>> x = ts.ones([1, 3, 4, 4])
        >>> print(seq_layer(x))
        [[[[27. 27.]
           [27. 27.]]
          [[27. 27.]
           [27. 27.]]]]
    """
    pass


class LayerList(CellList):
    """
    Holds Layers in a list.

    LayerList can be used like a regular Python list, support
    '__getitem__', '__setitem__', '__delitem__', '__len__', '__iter__' and '__iadd__',
    but layers it contains are properly registered, and will be visible by all Layer methods.

    Args:
        args (list, optional): List of subclass of Layer.

    Examples:
        >>> from tinyms.layers import LayerList, Conv2d, BatchNorm2d, ReLU
        >>>
        >>> conv = nn.Conv2d(100, 20, 3)
        >>> layers = LayerList([BatchNorm2d(20)])
        >>> layers.insert(0, Conv2d(100, 20, 3))
        >>> layers.append(ReLU())
        >>> layers
        LayerList<
          (0): Conv2d<input_channels=100, ..., bias_init=None>
          (1): BatchNorm2d<num_features=20, ..., moving_variance=Parameter (name=variance)>
          (2): ReLU<>
          >
    """
    pass
