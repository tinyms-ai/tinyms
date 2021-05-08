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
# ===========================================================================
'''
Bert for finetune script.
'''

import tinyms as ts
from tinyms import Tensor
from tinyms import Parameter
from tinyms import context
from tinyms import layers
from tinyms import primitives as P
from tinyms.initializers import  initializer

from .bert_for_pretraining import clip_grad
from .finetune_eval_model import BertCLSModel, BertNERModel, BertSquadModel


GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0
grad_scale = P.MultitypeFuncGraph("grad_scale")
reciprocal = P.Reciprocal()
@grad_scale.register("Tensor", "Tensor")
def tensor_grad_scale(scale, grad):
    return grad * reciprocal(scale)

_grad_overflow = P.MultitypeFuncGraph("_grad_overflow")
grad_overflow = P.FloatStatus()
@_grad_overflow.register("Tensor")
def _tensor_grad_overflow(grad):
    return grad_overflow(grad)

class CrossEntropyCalculation(layers.Layer):
    """
    Cross Entropy loss
    """

    def __init__(self, is_training=True):
        super(CrossEntropyCalculation, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, ts.float32)
        self.off_value = Tensor(0.0, ts.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()
        self.is_training = is_training

    def construct(self, logits, label_ids, num_labels):
        if self.is_training:
            label_ids = self.reshape(label_ids, self.last_idx)
            one_hot_labels = self.onehot(label_ids, num_labels, self.on_value, self.off_value)
            per_example_loss = self.neg(self.reduce_sum(one_hot_labels * logits, self.last_idx))
            loss = self.reduce_mean(per_example_loss, self.last_idx)
            return_value = self.cast(loss, ts.float32)
        else:
            return_value = logits * 1.0
        return return_value


class BertFinetuneLayer(layers.Layer):
    """
    Especifically defined for finetuning where only four inputs tensor are needed.
    """

    def __init__(self, network, optimizer, scale_update_layer=None):

        super(BertFinetuneLayer, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.optimizer.global_step = Parameter(initializer(0., [1,]), name='global_step')
        self.grad = P.GradOperation(get_by_list=True,
                                    sens_param=True)
        self.allreduce = P.AllReduce()
        self.grad_reducer = None
        self.cast = P.Cast()
        self.gpu_target = False
        if context.get_context("device_target") == "GPU":
            self.gpu_target = True
            self.float_status = P.FloatStatus()
            self.addn = P.AddN()
            self.reshape = P.Reshape()
        else:
            self.alloc_status = P.NPUAllocFloatStatus()
            self.get_status = P.NPUGetFloatStatus()
            self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.Depend()
        self.base = Tensor(1, ts.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = P.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_layer
        if scale_update_layer:
            self.loss_scale = Parameter(Tensor(scale_update_layer.get_loss_scale(), dtype=ts.float32),
                                        name="loss_scale")

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  label_ids,
                  sens=None):
        """Bert Finetune"""

        weights = self.weights
        init = False
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            label_ids)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        if not self.gpu_target:
            init = self.alloc_status()
            clear_before_grad = self.clear_before_grad(init)
            loss = P.depend(loss, init)
            self.depend_parameter_use(clear_before_grad, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 label_ids,
                                                 self.cast(scaling_sens,
                                                           ts.float32))
        grads = self.hyper_map(P.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(P.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        if not self.gpu_target:
            flag = self.get_status(init)
            flag_sum = self.reduce_sum(init, (0,))
            grads = P.depend(grads, flag)
            flag_sum = P.depend(flag_sum, flag)
        else:
            flag_sum = self.hyper_map(P.partial(_grad_overflow), grads)
            flag_sum = self.addn(flag_sum)
            flag_sum = self.reshape(flag_sum, (()))

        cond = self.less_equal(self.base, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond)
        return P.depend(ret, succ)


class BertSquadLayer(layers.Layer):
    """
    specifically defined for finetuning where only four inputs tensor are needed.
    """

    def __init__(self, network, optimizer, scale_update_layer=None):
        super(BertSquadLayer, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.allreduce = P.AllReduce()
        self.grad_reducer = None
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.depend_parameter_use = P.Depend()
        self.base = Tensor(1, ts.float32)
        self.less_equal = P.LessEqual()
        self.hyper_map = P.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_layer
        if scale_update_layer:
            self.loss_scale = Parameter(Tensor(scale_update_layer.get_loss_scale(), dtype=ts.float32),
                                        name="loss_scale")
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  start_position,
                  end_position,
                  unique_id,
                  is_impossible,
                  sens=None):
        """BertSquad"""
        weights = self.weights
        init = self.alloc_status()
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            start_position,
                            end_position,
                            unique_id,
                            is_impossible)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 start_position,
                                                 end_position,
                                                 unique_id,
                                                 is_impossible,
                                                 self.cast(scaling_sens,
                                                           ts.float32))
        clear_before_grad = self.clear_before_grad(init)
        loss = P.depend(loss, init)
        self.depend_parameter_use(clear_before_grad, scaling_sens)
        grads = self.hyper_map(P.partial(grad_scale, scaling_sens), grads)
        grads = self.hyper_map(P.partial(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        flag = self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        cond = self.less_equal(self.base, flag_sum)
        P.depend(grads, flag)
        P.depend(flag, flag_sum)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond)
        return P.depend(ret, succ)


class BertCLS(layers.Layer):
    """
    Train interface for classification finetuning task.
    """

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False,
                 assessment_method=""):
        super(BertCLS, self).__init__()
        self.bert = BertCLSModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings,
                                 assessment_method)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.assessment_method = assessment_method
        self.is_training = is_training
    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.assessment_method == "spearman_correlation":
            if self.is_training:
                loss = self.loss(logits, label_ids)
            else:
                loss = logits
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss


class BertNER(layers.Layer):
    """
    Train interface for sequence labeling finetuning task.
    """

    def __init__(self, config, batch_size, is_training, num_labels=11, use_crf=False,
                 tag_to_index=None, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertNER, self).__init__()
        self.bert = BertNERModel(config, is_training, num_labels, use_crf, dropout_prob, use_one_hot_embeddings)
        if use_crf:
            if not tag_to_index:
                raise Exception("The dict for tag-index mapping should be provided for CRF.")
            from src.CRF import CRF
            self.loss = CRF(tag_to_index, batch_size, config.seq_length, is_training)
        else:
            self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.use_crf = use_crf
    def construct(self, input_ids, input_mask, token_type_id, label_ids):
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.use_crf:
            loss = self.loss(logits, label_ids)
        else:
            loss = self.loss(logits, label_ids, self.num_labels)
        return loss


class BertSquad(layers.Layer):
    '''
    Train interface for SQuAD finetuning task.
    '''

    def __init__(self, config, is_training, num_labels=2, dropout_prob=0.0, use_one_hot_embeddings=False):
        super(BertSquad, self).__init__()
        self.bert = BertSquadModel(config, is_training, num_labels, dropout_prob, use_one_hot_embeddings)
        self.loss = CrossEntropyCalculation(is_training)
        self.num_labels = num_labels
        self.seq_length = config.seq_length
        self.is_training = is_training
        self.total_num = Parameter(Tensor([0], ts.float32), name='total_num')
        self.start_num = Parameter(Tensor([0], ts.float32), name='start_num')
        self.end_num = Parameter(Tensor([0], ts.float32), name='end_num')
        self.sum = P.ReduceSum()
        self.equal = P.Equal()
        self.argmax = P.ArgMaxWithValue(axis=1)
        self.squeeze = P.Squeeze(axis=-1)

    def construct(self, input_ids, input_mask, token_type_id, start_position, end_position, unique_id, is_impossible):
        """interface for SQuAD finetuning task"""
        logits = self.bert(input_ids, input_mask, token_type_id)
        if self.is_training:
            unstacked_logits_0 = self.squeeze(logits[:, :, 0:1])
            unstacked_logits_1 = self.squeeze(logits[:, :, 1:2])
            start_loss = self.loss(unstacked_logits_0, start_position, self.seq_length)
            end_loss = self.loss(unstacked_logits_1, end_position, self.seq_length)
            total_loss = (start_loss + end_loss) / 2.0
        else:
            start_logits = self.squeeze(logits[:, :, 0:1])
            end_logits = self.squeeze(logits[:, :, 1:2])
            total_loss = (unique_id, start_logits, end_logits)
        return total_loss
