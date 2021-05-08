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

from . import _transform_ops
from ._transform_ops import *
from ..data import BertDataset


__all__ = [
    'bert_transform', 'BertDatasetTransform',
]
__all__.extend(_transform_ops.__all__)


class BertDatasetTransform(object):
    r'''
    Apply preprocess operation on GeneratorDataset instance.
    '''
    def __init__(self):
        pass

    def apply_ds(self, data_set, batch_size):

        assert isinstance(data_set, BertDataset), "For BertDatasetTransform, BertDataset is needed"

        type_cast_op = TypeCast(ts.int32)
        data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_ids")
        data_set = data_set.map(operations=type_cast_op, input_columns="masked_lm_positions")
        data_set = data_set.map(operations=type_cast_op, input_columns="next_sentence_labels")
        data_set = data_set.map(operations=type_cast_op, input_columns="segment_ids")
        data_set = data_set.map(operations=type_cast_op, input_columns="input_mask")
        data_set = data_set.map(operations=type_cast_op, input_columns="input_ids")
        # apply batch operations
        data_set = data_set.batch(batch_size, drop_remainder=True)

        return data_set


bert_transform = BertDatasetTransform()