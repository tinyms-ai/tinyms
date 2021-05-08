# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""Bert for pretraining."""
import numpy as np

import tinyms as ts
from tinyms import layers
from tinyms import primitives as P
from tinyms import Tensor
from tinyms import Parameter
from tinyms.initializers import TruncatedNormal, initializer

from .bert import Bert

GRADIENT_CLIP_TYPE = 1
GRADIENT_CLIP_VALUE = 1.0

clip_grad = P.MultitypeFuncGraph("clip_grad")

@clip_grad.register("Number", "Number", "Tensor")
def _clip_grad(clip_type, clip_value, grad):
    """
    Clip gradients.

    Inputs:
        clip_type (int): The way to clip, 0 for 'value', 1 for 'norm'.
        clip_value (float): Specifies how much to clip.
        grad (tuple[Tensor]): Gradients.

    Outputs:
        tuple[Tensor], clipped gradients.
    """

    if clip_type not in (0, 1):
        return grad
    dt = P.DType()(grad)

    if clip_type == 0:
        new_grad = P.clip_by_value(grad, P.Cast()(ts.array((-clip_value,)), dt),
                                   P.Cast()(ts.array((clip_value,)), dt))
    else:
        new_grad = layers.ClipByNorm()(grad, P.Cast()(ts.array((clip_value,)), dt))
    return new_grad


class GetMaskedLMOutput(layers.Layer):
    """
    Get masked lm output.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, masked lm output.
    """

    def __init__(self, config):
        super(GetMaskedLMOutput, self).__init__()
        self.width = config.hidden_size
        self.reshape = P.Reshape()
        self.gather = P.Gather()

        weight_init = TruncatedNormal(config.initializer_range)
        self.dense = layers.Dense(self.width,
                              config.hidden_size,
                              weight_init=weight_init,
                              activation=config.hidden_act).to_float(config.compute_type)
        self.layernorm = layers.LayerNorm((config.hidden_size,)).to_float(config.compute_type)
        self.output_bias = Parameter(
            initializer(
                'zero',
                config.vocab_size))
        self.matmul = P.MatMul(transpose_b=True)
        self.log_softmax = layers.LogSoftmax(axis=-1)
        self.shape_flat_offsets = (-1, 1)
        self.last_idx = (-1,)
        self.shape_flat_sequence_tensor = (-1, self.width)
        self.seq_length_tensor = Tensor(np.array((config.seq_length,)).astype(np.int32))
        self.cast = P.Cast()
        self.compute_type = config.compute_type
        self.dtype = config.dtype

    def construct(self,
                  input_tensor,
                  output_weights,
                  positions):
        """Get output log_probs"""
        rng = ts.array(ts.arange(P.Shape()(input_tensor)[0]))
        flat_offsets = self.reshape(rng * self.seq_length_tensor, self.shape_flat_offsets)
        flat_position = self.reshape(positions + flat_offsets, self.last_idx)
        flat_sequence_tensor = self.reshape(input_tensor, self.shape_flat_sequence_tensor)
        input_tensor = self.gather(flat_sequence_tensor, flat_position, 0)
        input_tensor = self.cast(input_tensor, self.compute_type)
        output_weights = self.cast(output_weights, self.compute_type)
        input_tensor = self.dense(input_tensor)
        input_tensor = self.layernorm(input_tensor)
        logits = self.matmul(input_tensor, output_weights)
        logits = self.cast(logits, self.dtype)
        logits = logits + self.output_bias
        log_probs = self.log_softmax(logits)
        return log_probs


class GetNextSentenceOutput(layers.Layer):
    """
    Get next sentence output.

    Args:
        config (BertConfig): The config of Bert.

    Returns:
        Tensor, next sentence output.
    """

    def __init__(self, config):
        super(GetNextSentenceOutput, self).__init__()
        self.log_softmax = P.LogSoftmax()
        weight_init = TruncatedNormal(config.initializer_range)
        self.dense = layers.Dense(config.hidden_size, 2,
                              weight_init=weight_init, has_bias=True).to_float(config.compute_type)
        self.dtype = config.dtype
        self.cast = P.Cast()

    def construct(self, input_tensor):
        logits = self.dense(input_tensor)
        logits = self.cast(logits, self.dtype)
        log_prob = self.log_softmax(logits)
        return log_prob


class BertPreTraining(layers.Layer):
    """
    Bert pretraining network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings.

    Returns:
        Tensor, prediction_scores, seq_relationship_score.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings):
        super(BertPreTraining, self).__init__()
        self.bert = Bert(config, is_training, use_one_hot_embeddings)
        self.cls1 = GetMaskedLMOutput(config)
        self.cls2 = GetNextSentenceOutput(config)

    def construct(self, input_ids, input_mask, token_type_id,
                  masked_lm_positions):
        sequence_output, pooled_output, embedding_table = \
            self.bert(input_ids, token_type_id, input_mask)
        prediction_scores = self.cls1(sequence_output,
                                      embedding_table,
                                      masked_lm_positions)
        seq_relationship_score = self.cls2(pooled_output)
        return prediction_scores, seq_relationship_score


class BertPretrainingLoss(layers.Layer):
    """
    Provide bert pre-training loss.

    Args:
        config (BertConfig): The config of BertModel.

    Returns:
        Tensor, total loss.
    """

    def __init__(self, config):
        super(BertPretrainingLoss, self).__init__()
        self.vocab_size = config.vocab_size
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, ts.float32)
        self.off_value = Tensor(0.0, ts.float32)
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.reshape = P.Reshape()
        self.last_idx = (-1,)
        self.neg = P.Neg()
        self.cast = P.Cast()

    def construct(self, prediction_scores, seq_relationship_score, masked_lm_ids,
                  masked_lm_weights, next_sentence_labels):
        """Defines the computation performed."""
        label_ids = self.reshape(masked_lm_ids, self.last_idx)
        label_weights = self.cast(self.reshape(masked_lm_weights, self.last_idx), ts.float32)
        one_hot_labels = self.onehot(label_ids, self.vocab_size, self.on_value, self.off_value)

        per_example_loss = self.neg(self.reduce_sum(prediction_scores * one_hot_labels, self.last_idx))
        numerator = self.reduce_sum(label_weights * per_example_loss, ())
        denominator = self.reduce_sum(label_weights, ()) + self.cast(ts.array((1e-5,)), ts.float32)
        masked_lm_loss = numerator / denominator

        # next_sentence_loss
        labels = self.reshape(next_sentence_labels, self.last_idx)
        one_hot_labels = self.onehot(labels, 2, self.on_value, self.off_value)
        per_example_loss = self.neg(self.reduce_sum(
            one_hot_labels * seq_relationship_score, self.last_idx))
        next_sentence_loss = self.reduce_mean(per_example_loss, self.last_idx)

        # total_loss
        total_loss = masked_lm_loss + next_sentence_loss

        return total_loss


class BertNetworkWithLoss(layers.Layer):
    """
    Provide bert pre-training loss through network.

    Args:
        config (BertConfig): The config of BertModel.
        is_training (bool): Specifies whether to use the training mode.
        use_one_hot_embeddings (bool): Specifies whether to use one-hot for embeddings. Default: False.

    Returns:
        Tensor, the loss of the network.
    """

    def __init__(self, config, is_training, use_one_hot_embeddings=False):
        super(BertNetworkWithLoss, self).__init__()
        self.bert = BertPreTraining(config, is_training, use_one_hot_embeddings)
        self.loss = BertPretrainingLoss(config)
        self.cast = P.Cast()

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):
        """Get pre-training loss"""
        prediction_scores, seq_relationship_score = \
            self.bert(input_ids, input_mask, token_type_id, masked_lm_positions)
        total_loss = self.loss(prediction_scores, seq_relationship_score,
                               masked_lm_ids, masked_lm_weights, next_sentence_labels)
        return self.cast(total_loss, ts.float32)


class BertTrainOneStepCell(layers.TrainOneStepCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(BertTrainOneStepCell, self).__init__(network, optimizer, sens)
        self.cast = P.Cast()
        self.hyper_map = P.HyperMap()

    def set_sens(self, value):
        self.sens = value

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights):
        """Defines the computation performed."""
        weights = self.weights

        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(ts.array((self.sens,)),
                                                           ts.float32))
        grads = self.hyper_map(P.Partial()(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        grads = self.grad_reducer(grads)
        succ = self.optimizer(grads)
        return P.Depend()(loss, succ)


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


class BertTrainOneStepWithLossScaleCell(layers.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """

    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainOneStepWithLossScaleCell, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ts.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ts.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(P.Partial()(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(P.Partial()(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if sens is None:
            overflow = self.loss_scaling_manager(self.loss_scale, cond)
        if overflow:
            succ = False
        else:
            succ = self.optimizer(grads)
        ret = (loss, cond, scaling_sens)
        return P.Depend()(ret, succ)


class BertTrainOneStepWithLossScaleCellForAdam(layers.TrainOneStepWithLossScaleCell):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.
    Different from BertTrainOneStepWithLossScaleCell, the optimizer takes the overflow
    condition as input.

    Args:
        network (Cell): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
    """
    def __init__(self, network, optimizer, scale_update_cell=None):
        super(BertTrainOneStepWithLossScaleCellForAdam, self).__init__(network, optimizer, scale_update_cell)
        self.cast = P.Cast()
        self.degree = 1
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ts.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        status, scaling_sens = self.start_overflow_check(loss, scaling_sens)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ts.float32))
        # apply grad reducer on grads
        grads = self.grad_reducer(grads)
        grads = self.hyper_map(P.Partial()(grad_scale, scaling_sens * self.degree), grads)
        grads = self.hyper_map(P.Partial()(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
        cond = self.get_overflow_status(status, grads)
        overflow = cond
        if self.loss_scaling_manager is not None:
            overflow = self.loss_scaling_manager(scaling_sens, cond)
        succ = self.optimizer(grads, overflow)
        ret = (loss, cond, scaling_sens)
        return P.Depend()(ret, succ)

cast = P.Cast()
add_grads = P.MultitypeFuncGraph("add_grads")


@add_grads.register("Tensor", "Tensor")
def _add_grads(accu_grad, grad):
    return accu_grad + cast(grad, ts.float32)

update_accu_grads = P.MultitypeFuncGraph("update_accu_grads")

@update_accu_grads.register("Tensor", "Tensor")
def _update_accu_grads(accu_grad, grad):
    succ = True
    return P.Depend()(succ, P.AssignAdd()(accu_grad, cast(grad, ts.float32)))

accumulate_accu_grads = P.MultitypeFuncGraph("accumulate_accu_grads")

@accumulate_accu_grads.register("Tensor", "Tensor")
def _accumulate_accu_grads(accu_grad, grad):
    succ = True
    return P.Depend()(succ, P.AssignAdd()(accu_grad, cast(grad, ts.float32)))


zeroslike = P.ZerosLike()
reset_accu_grads = P.MultitypeFuncGraph("reset_accu_grads")


@reset_accu_grads.register("Tensor")
def _reset_accu_grads(accu_grad):
    succ = True
    return P.Depend()(succ, P.Assign()(accu_grad, zeroslike(accu_grad)))


class BertTrainAccumulationAllReducePostWithLossScaleLayer(layers.Layer):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    Args:
        network (layers.Layer): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                batch_size * accumulation_steps. Default: 1.
    """

    def __init__(self, network, optimizer, scale_update_cell=None, accumulation_steps=1, enable_global_norm=False):
        super(BertTrainAccumulationAllReducePostWithLossScaleLayer, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enable_global_norm = enable_global_norm
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.local_step = Parameter(initializer(0, [1], ts.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], ts.int32))
        self.accu_loss = Parameter(initializer(0, [1], ts.float32))

        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = P.Identity()
        self.degree = 1
        self.overflow_reducer = P.Identity()
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_status = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, ts.float32)
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = P.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ts.float32))

    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens
        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        init = P.Depend()(init, loss)
        clear_status = self.clear_status(init)
        scaling_sens = P.Depend()(scaling_sens, clear_status)
        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ts.float32))

        accu_succ = self.hyper_map(accumulate_accu_grads, self.accu_grads, grads)
        mean_loss = P.Depend()(mean_loss, accu_succ)

        init = P.Depend()(init, mean_loss)
        get_status = self.get_status(init)
        init = P.Depend()(init, get_status)
        flag_sum = self.reduce_sum(init, (0,))
        overflow = self.less_equal(self.base, flag_sum)
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)

        if is_accu_step:
            succ = False
        else:
            # apply grad reducer on grads
            grads = self.grad_reducer(self.accu_grads)
            scaling = scaling_sens * self.degree * self.accumulation_steps
            grads = self.hyper_map(P.Partial()(grad_scale, scaling), grads)
            if self.enable_global_norm:
                grads = P.clip_by_global_norm(grads, 1.0, None)
            else:
                grads = self.hyper_map(P.Partial()(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)
            accu_overflow = P.Depend()(accu_overflow, grads)
            accu_overflow = self.overflow_reducer(accu_overflow)
            overflow = self.less_equal(self.base, accu_overflow)
            accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
            overflow = P.Depend()(overflow, accu_succ)
            overflow = self.reshape(overflow, (()))
            if sens is None:
                overflow = self.loss_scaling_manager(self.loss_scale, overflow)
            if overflow:
                succ = False
            else:
                succ = self.optimizer(grads)

        ret = (mean_loss, overflow, scaling_sens)
        return P.Depend()(ret, succ)


class BertTrainAccumulationAllReduceEachWithLossScaleLayer(layers.Layer):
    """
    Encapsulation class of bert network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    To mimic higher batch size, gradients are accumulated N times before weight update.

    Args:
        network (layers.Layer): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        scale_update_cell (Cell): Cell to do the loss scale. Default: None.
        accumulation_steps (int): Number of accumulation steps before gradient update. The global batch size =
                                  batch_size * accumulation_steps. Default: 1.
    """
    def __init__(self, network, optimizer, scale_update_cell=None, accumulation_steps=1, enable_global_norm=False):
        super(BertTrainAccumulationAllReduceEachWithLossScaleLayer, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = optimizer.parameters
        self.optimizer = optimizer
        self.accumulation_steps = accumulation_steps
        self.enable_global_norm = enable_global_norm
        self.one = Tensor(np.array([1]).astype(np.int32))
        self.zero = Tensor(np.array([0]).astype(np.int32))
        self.local_step = Parameter(initializer(0, [1], ts.int32))
        self.accu_grads = self.weights.clone(prefix="accu_grads", init='zeros')
        self.accu_overflow = Parameter(initializer(0, [1], ts.int32))
        self.accu_loss = Parameter(initializer(0, [1], ts.float32))

        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.grad_reducer = P.Identity()
        self.degree = 1
        self.overflow_reducer = P.Identity()
        self.cast = P.Cast()
        self.alloc_status = P.NPUAllocFloatStatus()
        self.get_status = P.NPUGetFloatStatus()
        self.clear_before_grad = P.NPUClearFloatStatus()
        self.reduce_sum = P.ReduceSum(keep_dims=False)
        self.base = Tensor(1, ts.float32)
        self.less_equal = P.LessEqual()
        self.logical_or = P.LogicalOr()
        self.not_equal = P.NotEqual()
        self.select = P.Select()
        self.reshape = P.Reshape()
        self.hyper_map = P.HyperMap()
        self.loss_scale = None
        self.loss_scaling_manager = scale_update_cell
        if scale_update_cell:
            self.loss_scale = Parameter(Tensor(scale_update_cell.get_loss_scale(), dtype=ts.float32))

    @P.add_flags(has_effect=True)
    def construct(self,
                  input_ids,
                  input_mask,
                  token_type_id,
                  next_sentence_labels,
                  masked_lm_positions,
                  masked_lm_ids,
                  masked_lm_weights,
                  sens=None):
        """Defines the computation performed."""
        weights = self.weights
        loss = self.network(input_ids,
                            input_mask,
                            token_type_id,
                            next_sentence_labels,
                            masked_lm_positions,
                            masked_lm_ids,
                            masked_lm_weights)
        if sens is None:
            scaling_sens = self.loss_scale
        else:
            scaling_sens = sens

        # update accumulation parameters
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)
        self.local_step = self.select(is_accu_step, self.local_step + self.one, self.one)
        self.accu_loss = self.select(is_accu_step, self.accu_loss + loss, loss)
        mean_loss = self.accu_loss / self.local_step
        is_accu_step = self.not_equal(self.local_step, self.accumulation_steps)

        # alloc status and clear should be right before gradoperation
        init = self.alloc_status()
        self.clear_before_grad(init)
        grads = self.grad(self.network, weights)(input_ids,
                                                 input_mask,
                                                 token_type_id,
                                                 next_sentence_labels,
                                                 masked_lm_positions,
                                                 masked_lm_ids,
                                                 masked_lm_weights,
                                                 self.cast(scaling_sens,
                                                           ts.float32))


        accu_grads = self.hyper_map(add_grads, self.accu_grads, grads)
        scaling = scaling_sens * self.degree * self.accumulation_steps
        grads = self.hyper_map(P.Partial()(grad_scale, scaling), accu_grads)
        grads = self.grad_reducer(grads)

        self.get_status(init)
        flag_sum = self.reduce_sum(init, (0,))
        flag_reduce = self.overflow_reducer(flag_sum)
        overflow = self.less_equal(self.base, flag_reduce)
        overflow = self.logical_or(self.not_equal(self.accu_overflow, self.zero), overflow)
        accu_overflow = self.select(overflow, self.one, self.zero)
        self.accu_overflow = self.select(is_accu_step, accu_overflow, self.zero)
        overflow = self.reshape(overflow, (()))

        if is_accu_step:
            succ = False
            accu_succ = self.hyper_map(update_accu_grads, self.accu_grads, accu_grads)
            succ = P.Depend()(succ, accu_succ)
        else:
            if sens is None:
                overflow = self.loss_scaling_manager(self.loss_scale, overflow)
            if overflow:
                succ = False
            else:
                if self.enable_global_norm:
                    grads = P.clip_by_global_norm(grads, 1.0, None)
                else:
                    grads = self.hyper_map(P.Partial()(clip_grad, GRADIENT_CLIP_TYPE, GRADIENT_CLIP_VALUE), grads)

                succ = self.optimizer(grads)

            accu_succ = self.hyper_map(reset_accu_grads, self.accu_grads)
            succ = P.Depend()(succ, accu_succ)

        ret = (mean_loss, overflow, scaling_sens)
        return P.Depend()(ret, succ)
