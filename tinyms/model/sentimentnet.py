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

import math
import numpy as np

import tinyms as ts
from tinyms import layers, Tensor
from tinyms import primitives as P
from tinyms import ParameterTuple, context, Parameter
from tinyms.initializers import initializer


STACK_LSTM_DEVICE = ["CPU"]


# Initialize short-term memory (h) and long-term memory (c) to 0
def lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 2 if bidirectional else 1
    h = ts.zeros((num_layers * num_directions, batch_size, hidden_size))
    c = ts.zeros((num_layers * num_directions, batch_size, hidden_size))
    return h, c


def stack_lstm_default_state(batch_size, hidden_size, num_layers, bidirectional):
    """init default input."""
    num_directions = 2 if bidirectional else 1

    h_list = c_list = []
    for _ in range(num_layers):
        h_list.append(ts.zeros((num_directions, batch_size, hidden_size)))
        c_list.append(ts.zeros((num_directions, batch_size, hidden_size)))
    h, c = tuple(h_list), tuple(c_list)
    return h, c


class StackLSTM(layers.Layer):
    """
    Stack multi-layers LSTM together.
    """

    def __init__(self,
                 input_size,
                 hidden_size,
                 num_layers=1,
                 has_bias=True,
                 batch_first=False,
                 dropout=0.0,
                 bidirectional=False):
        super(StackLSTM, self).__init__()
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.transpose = P.Transpose()

        # direction number
        num_directions = 2 if bidirectional else 1

        # input_size list
        input_size_list = [input_size]
        for i in range(num_layers - 1):
            input_size_list.append(hidden_size * num_directions)

        # layers
        layer_list = []
        for i in range(num_layers):
            layer_list.append(layers.LSTMCell(input_size=input_size_list[i],
                                              hidden_size=hidden_size,
                                              has_bias=has_bias,
                                              batch_first=batch_first,
                                              bidirectional=bidirectional,
                                              dropout=dropout))

        # weights
        weights = []
        for i in range(num_layers):
            # weight size
            weight_size = (input_size_list[i] + hidden_size) * num_directions * hidden_size * 4
            if has_bias:
                bias_size = num_directions * hidden_size * 4
                weight_size = weight_size + bias_size

            # numpy weight
            stdv = 1 / math.sqrt(hidden_size)
            w_np = np.random.uniform(-stdv, stdv, (weight_size, 1, 1)).astype(np.float32)

            # lstm weight
            weights.append(Parameter(initializer(Tensor(w_np), w_np.shape), name="weight" + str(i)))

        #
        self.lstm = layer_list
        self.weight = ParameterTuple(tuple(weights))

    def construct(self, x, hx):
        """construct"""
        h, c = hx
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        # stack lstm
        hn = cn = None
        for i in range(self.num_layers):
            x, hn, cn, _, _ = self.lstm[i](x, h[i], c[i], self.weight[i])
        if self.batch_first:
            x = self.transpose(x, (1, 0, 2))
        return x, (hn, cn)


class SentimentNet(layers.Layer):
    """Sentiment network structure."""

    def __init__(self,
                 vocab_size,
                 embed_size,
                 num_hiddens,
                 num_layers,
                 bidirectional,
                 num_classes,
                 weight,
                 batch_size):
        super(SentimentNet, self).__init__()
        # Map words to vectors
        self.embedding = layers.Embedding(vocab_size,
                                          embed_size,
                                          embedding_table=weight)
        self.embedding.embedding_table.requires_grad = False
        self.trans = P.Transpose()
        self.perm = (1, 0, 2)

        if context.get_context("device_target") in STACK_LSTM_DEVICE:
            # stack lstm by user
            self.encoder = StackLSTM(input_size=embed_size,
                                     hidden_size=num_hiddens,
                                     num_layers=num_layers,
                                     has_bias=True,
                                     bidirectional=bidirectional,
                                     dropout=0.0)
            self.h, self.c = stack_lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)
        else:
            # standard lstm
            self.encoder = layers.LSTM(input_size=embed_size,
                                       hidden_size=num_hiddens,
                                       num_layers=num_layers,
                                       has_bias=True,
                                       bidirectional=bidirectional,
                                       dropout=0.0)
            self.h, self.c = lstm_default_state(batch_size, num_hiddens, num_layers, bidirectional)

        self.concat = P.Concat(1)
        if bidirectional:
            self.decoder = layers.Dense(num_hiddens * 4, num_classes)
        else:
            self.decoder = layers.Dense(num_hiddens * 2, num_classes)

    def construct(self, inputs):
        # input：(64,500,300)
        embeddings = self.embedding(inputs)
        embeddings = self.trans(embeddings, self.perm)
        output, _ = self.encoder(embeddings, (self.h, self.c))
        # states[i] size(64,200)  -> encoding.size(64,400)
        encoding = self.concat((output[0], output[499]))
        outputs = self.decoder(encoding)
        return outputs


def sentimentnet(
    vocab_size,
    embed_size,
    num_hiddens,
    num_layers,
    bidirectional,
    num_classes,
    weight,
    batch_size
):
    """Sentiment network structure."""

    return SentimentNet(
        vocab_size,
        embed_size,
        num_hiddens,
        num_layers,
        bidirectional,
        num_classes,
        weight,
        batch_size
    )
