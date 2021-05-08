# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Bert Init."""

from .bert_for_pretraining import BertNetworkWithLoss, BertPreTraining, \
    BertPretrainingLoss, GetMaskedLMOutput, GetNextSentenceOutput, \
    BertTrainOneStepCell, BertTrainOneStepWithLossScaleCell, \
    BertTrainAccumulationAllReduceEachWithLossScaleLayer, \
    BertTrainAccumulationAllReducePostWithLossScaleLayer, \
    BertTrainOneStepWithLossScaleCellForAdam
from .bert import BertAttention, BertEncoderLayer, Bert, bert, \
    BertOutput, BertSelfAttention, BertTransformer, EmbeddingLookup, \
    EmbeddingPostprocessor, RelaPosEmbeddingsGenerator, RelaPosMatrixGenerator, \
    SaturateCast, CreateAttentionMaskFromInputMask
from .bert_for_finetune import BertFinetuneLayer, BertCLS

__all__ = [
    "BertNetworkWithLoss", "BertPreTraining", "BertPretrainingLoss",
    "GetMaskedLMOutput", "GetNextSentenceOutput", "BertTrainOneStepCell",
    "BertTrainOneStepWithLossScaleCell", "BertTrainAccumulationAllReduceEachWithLossScaleLayer",
    "BertTrainAccumulationAllReducePostWithLossScaleLayer",
    "BertAttention", "BertEncoderLayer", "Bert", "bert", "BertOutput",
    "BertSelfAttention", "BertTransformer", "EmbeddingLookup",
    "EmbeddingPostprocessor", "RelaPosEmbeddingsGenerator",
    "RelaPosMatrixGenerator", "SaturateCast", "CreateAttentionMaskFromInputMask",
    "BertTrainOneStepWithLossScaleCellForAdam", "BertFinetuneLayer", "BertCLS"
]
