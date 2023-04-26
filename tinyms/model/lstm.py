# Copyright 2023 Huawei Technologies Co., Ltd
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
import mindspore.numpy as mnp
import mindspore.dataset as ds
from mindspore import nn, ops, Tensor
from mindspore.common.initializer import Uniform, HeUniform
import numpy as np


class RNN(nn.Cell):
    def __init__(self, embeddings, hidden_dim, output_dim, n_layers,
                 bidirectional, dropout, pad_idx):
        super().__init__()
        vocab_size, embedding_dim = embeddings.shape
        self.embedding = nn.Embedding(vocab_size,
                                      embedding_dim,
                                      embedding_table=Tensor(embeddings),
                                      padding_idx=pad_idx)
        self.rnn = nn.LSTM(embedding_dim,
                           hidden_dim,
                           num_layers=n_layers,
                           bidirectional=bidirectional,
                           dropout=dropout,
                           batch_first=True)
        weight_init = HeUniform(math.sqrt(5))
        bias_init = Uniform(1 / math.sqrt(hidden_dim * 2))
        self.predict = nn.Dense(hidden_dim * 2,
                                output_dim,
                                weight_init=weight_init,
                                bias_init=bias_init)
        self.dropout = nn.Dropout(1 - dropout)
        self.softmax = ops.Softmax()

    def construct(self, inputs):
        """
        :param inputs:
        :return: output vector
        """
        embedded = self.dropout(self.embedding(inputs))
        _, (hidden, _) = self.rnn(embedded)
        hidden = self.dropout(mnp.concatenate((hidden[-2, :, :], hidden[-1, :, :]), axis=1))
        output = self.predict(hidden)
        return self.softmax(output)


def load_glove(glove_path):
    embeddings = []
    tokens = []
    with open(glove_path, encoding='utf-8') as file:
        for glove in file:
            word, embedding = glove.split(maxsplit=1)
            tokens.append(word)
            embeddings.append(np.fromstring(embedding, dtype=np.float32, sep=' '))

    embeddings.append(np.random.rand(100))
    embeddings.append(np.zeros((100,), np.float32))

    vocab = ds.text.Vocab.from_list(tokens, special_tokens=["<unk>", "<pad>"], special_first=False)
    embeddings = np.array(embeddings).astype(np.float32)
    return vocab, embeddings


def lstm(glove_path, class_num=10):
    """
    Get ResNet50 neural network.
    Args:
        glove_path: path of glove file
        class_num (int): Class number. Default: 10.
    Returns:
        layers.Layer, layer instance of LST< neural network.
    Examples:
        >>> net = lstm(glove_path, 5)
    """
    embeddings, vocab = load_glove(glove_path)
    return RNN(embeddings,
               hidden_dim=256,
               output_dim=class_num,
               n_layers=3,
               bidirectional=True,
               dropout=0.5,
               pad_idx=vocab.tokens_to_ids('<pad>'))
