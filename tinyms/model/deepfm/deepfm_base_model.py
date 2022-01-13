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

from tinyms import layers, Tensor
from tinyms import Parameter

from tinyms.common import dtype
from tinyms.layers import Dropout
from tinyms.primitives import BiasAdd, Cast, MatMul, ReLU, Sigmoid, Tanh
from tinyms.primitives import Concat, Gather, Mul, ReduceSum, Reshape, Square, Tile


def _init_activation(activation):
    activation = str(activation).lower()
    if activation == "relu":
        activation_func = ReLU()
    elif activation == "sigmoid":
        activation_func = Sigmoid()
    elif activation == "tanh":
        activation_func = Tanh()
    else:
        raise ValueError("activation type: {} not supported!".format(activation))

    return activation_func


class DenseLayer(layers.Layer):
    """
    DeepFM denselayer with convert dtypes definition.

    Args:
        in_channels (int): Input channels.
        out_channels (int): Output channels.
        keep_prob (float): Keep prob. Default: 1.0.
        convert_dtype (bool): Whether to convert data type. Defalut: True.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> DenseLayer(3, 256)
    """
    def __init__(self, in_channels, out_channels, keep_prob=0.9, convert_dtype=True):
        super(DenseLayer, self).__init__()
        self.weight = Parameter(
            Tensor(np.random.normal(loc=0.0, scale=0.01, size=[in_channels, out_channels]).astype(dtype=np.float32)),
            name="weight")
        self.bias = Parameter(
            Tensor(np.random.normal(loc=0.0, scale=0.01, size=[out_channels]).astype(dtype=np.float32)),
            name="bias")
        self.convert_dtype = convert_dtype
        self.dropout = Dropout(keep_prob=keep_prob)
        self.cast = Cast()
        self.matmul = MatMul(transpose_b=False)
        self.bias_add = BiasAdd()
        self.activation = ReLU()

    def construct(self, x):
        """
        Construct function
        """
        x = self.dropout(x)
        if self.convert_dtype:
            x = self.cast(x, dtype.float16)
            weight = self.cast(self.weight, dtype.float16)
            bias = self.cast(self.bias, dtype.float16)
            wx = self.matmul(x, weight)
            wx = self.bias_add(wx, bias)
            if self.activation is not None:
                wx = self.activation(wx)
            wx = self.cast(wx, dtype.float32)
        else:
            wx = self.matmul(x, self.weight)
            wx = self.bias_add(wx, self.bias)
            if self.activation:
                wx = self.activation(wx)
        return wx


class DeepFM(layers.Layer):
    """
    DeepFM architecture.

    Args:
        field_size (int): The field size.
        vocab_size (int): The vocabulary size.
        embed_size (int): The embeding size.
        keep_prob (float): The keep prob value. Default: 0.9.
        convert_dtype (bool): Whether to convert data type.
        CPU can only set to False. Defalut: False.

    Returns:
        Tensor, output tensor.

    Examples:
        >>> from tinyms.model import DeepFM
        >>>
        >>> DeepFM(field_size=39, vocab_size=184965, embed_size=80)

    """
    def __init__(self, field_size, vocab_size, embed_size, keep_prob=0.9, convert_dtype=False):
        super(DeepFM, self).__init__()
        self.field_size = field_size
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.embedding = Parameter(
            Tensor(np.random.normal(loc=0.0, scale=0.01, size=[vocab_size, embed_size]).astype(dtype=np.float32)),
            name="embedding")
        self.fm_weight = Parameter(
            Tensor(np.random.normal(loc=0.0, scale=0.01, size=[vocab_size, 1]).astype(dtype=np.float32)),
            name="fm_weight")
        self.dense_layer_1 = DenseLayer(field_size * embed_size, 1024, keep_prob=keep_prob, convert_dtype=convert_dtype)
        self.dense_layer_2 = DenseLayer(1024, 512, keep_prob=keep_prob, convert_dtype=convert_dtype)
        self.dense_layer_3 = DenseLayer(512, 256, keep_prob=keep_prob, convert_dtype=convert_dtype)
        self.dense_layer_4 = DenseLayer(256, 128, keep_prob=keep_prob, convert_dtype=convert_dtype)
        self.dense_layer_5 = DenseLayer(128, 1, keep_prob=keep_prob, convert_dtype=convert_dtype)
        self.gather = Gather()
        self.mul = Mul()
        self.reduce_sum = ReduceSum(keep_dims=False)
        self.reshape = Reshape()
        self.square = Square()
        self.tile = Tile()
        self.concat = Concat(axis=1)
        self.cast = Cast()

    def construct(self, ids, wts):
        mask = self.reshape(wts, (-1, self.field_size, 1))
        # Linear layer
        fm_id_weight = self.gather(self.fm_weight, ids, 0)
        wx = self.mul(fm_id_weight, mask)
        linear_out = self.reduce_sum(wx, 1)
        # FM layer
        fm_id_embeds = self.gather(self.embedding, ids, 0)
        vx = self.mul(fm_id_embeds, mask)
        v1 = self.reduce_sum(vx, 1)
        v1 = self.square(v1)
        v2 = self.square(vx)
        v2 = self.reduce_sum(v2, 1)
        fm_out = 0.5 * self.reduce_sum(v1 - v2, 1)
        fm_out = self.reshape(fm_out, (-1, 1))
        # Deep layer
        deep_in = self.reshape(vx, (-1, self.field_size * self.embed_size))
        deep_in = self.dense_layer_1(deep_in)
        deep_in = self.dense_layer_2(deep_in)
        deep_in = self.dense_layer_3(deep_in)
        deep_in = self.dense_layer_4(deep_in)
        deep_out = self.dense_layer_5(deep_in)
        out = linear_out + fm_out + deep_out
        return out, self.fm_weight, self.embedding


def deepfm(**kwargs):
    """
    Get DeepFM neural network.

    Args:
        field_size (int): The field size. Default: 39.
        vocab_size (int): The vocabulary size. Default: 184965.
        embed_size (int): The embeding size. Default: 80.
        keep_prob (float): The keep prob value. Default: 0.9.
        convert_dtype (bool): Whether to convert data type. Defalut: False.

    Returns:
        layers.Layer, layer instance of DeepFM neural network.

    Examples:
        >>> from tinyms.model import deepfm
        >>>
        >>> net = deepfm(39, 184965, 80)
    """
    return DeepFM(field_size=kwargs.get('field_size', 39),
                  vocab_size=kwargs.get('vocab_size', 184965),
                  embed_size=kwargs.get('embed_size', 80),
                  keep_prob=kwargs.get('keep_prob', 0.9),
                  convert_dtype=kwargs.get('convert_dtype', False))
