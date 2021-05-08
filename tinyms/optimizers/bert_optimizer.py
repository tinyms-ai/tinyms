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
"""AdamWeightDecayForBert, a customized Adam for bert. Input: gradient, overflow flag."""
import numpy as np

from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR
from mindspore._checkparam import Validator as validator
from mindspore._checkparam import Rel


import tinyms as ts
from tinyms import Tensor
from tinyms import primitives as P
from tinyms import context
from . import AdamWeightDecay, Lamb, Momentum, THOR, Optimizer


__all__ = ['AdamWeightDecayForBert', 'AdamWeightDecayOp']



_adam_opt = P.MultitypeFuncGraph("adam_opt")


def _check_param_value(beta1, beta2, eps, prim_name):
    """Check the type of inputs."""
    validator.check_value_type("beta1", beta1, [float], prim_name)
    validator.check_value_type("beta2", beta2, [float], prim_name)
    validator.check_value_type("eps", eps, [float], prim_name)
    validator.check_float_range(beta1, 0.0, 1.0, Rel.INC_NEITHER, "beta1", prim_name)
    validator.check_float_range(beta2, 0.0, 1.0, Rel.INC_NEITHER, "beta2", prim_name)
    validator.check_positive_float(eps, "eps", prim_name)

class AdamWeightDecayForBert(Optimizer):
    """
    Implements the Adam algorithm to fix the weight decay.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" is in the keys, the value of the corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" is in the keys, the value of the corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" is in the keys, the value must be the order of parameters and
              the order will be followed in the optimizer. There are no other keys in the `dict` and the parameters
              which in the 'order_params' must be in one of group parameters.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use the dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: 0.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.
        - **overflow** (tuple[Tensor]) - The overflow flag in dynamiclossscale.

    Outputs:
        tuple[bool], all elements are True.

    Supported Platforms:
        ``Ascend`` ``GPU``

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = AdamWeightDecay(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = AdamWeightDecay(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = layers.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net)
        >>> model.compile(loss_fn=loss, optimizer=optim)
   """
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayForBert, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = ts.array([beta1], dtype=ts.float32)
        self.beta2 = ts.array([beta2], dtype=ts.float32)
        self.eps = ts.array([eps], dtype=ts.float32)
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.hyper_map = P.HyperMap()
        self.op_select = P.Select()
        self.op_cast = P.Cast()
        self.op_reshape = P.Reshape()
        self.op_shape = P.Shape()

    def construct(self, gradients, overflow):
        """AdamWeightDecayForBert"""
        lr = self.get_lr()
        cond = self.op_cast(P.Fill()(ts.int32, self.op_shape(self.beta1), 1) *\
                            self.op_reshape(overflow, (())), ts.bool_)
        beta1 = self.op_select(cond, self.op_cast(ts.array((1.0,)), ts.float32), self.beta1)
        beta2 = self.op_select(cond, self.op_cast(ts.array((1.0,)), ts.float32), self.beta2)
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(P.Partial()(_adam_opt, self.beta1, self.beta2, self.eps),
                                              lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(P.Partial()(_adam_opt, beta1, beta2, self.eps, lr, overflow),
                                              self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(P.Partial()(_adam_opt, self.beta1, self.beta2, self.eps, lr, self.weight_decay),
                                          self.parameters, self.moments1, self.moments2,
                                          gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result

class AdamWeightDecayOp(Optimizer):
    """
    Implements the Adam algorithm to fix the weight decay. It is a complete operator, not a combination of other ops.

    Note:
        When separating parameter groups, the weight decay in each group will be applied on the parameters if the
        weight decay is positive. When not separating parameter groups, the `weight_decay` in the API will be applied
        on the parameters without 'beta' or 'gamma' in their names if `weight_decay` is positive.

        To improve parameter groups performance, the customized order of parameters can be supported.

    Args:
        params (Union[list[Parameter], list[dict]]): When the `params` is a list of `Parameter` which will be updated,
            the element in `params` must be class `Parameter`. When the `params` is a list of `dict`, the "params",
            "lr", "weight_decay" and "order_params" are the keys can be parsed.

            - params: Required. The value must be a list of `Parameter`.

            - lr: Optional. If "lr" is in the keys, the value of the corresponding learning rate will be used.
              If not, the `learning_rate` in the API will be used.

            - weight_decay: Optional. If "weight_decay" is in the keys, the value of the corresponding weight decay
              will be used. If not, the `weight_decay` in the API will be used.

            - order_params: Optional. If "order_params" is in the keys, the value must be the order of parameters and
              the order will be followed in the optimizer. There are no other keys in the `dict` and the parameters
              which in the 'order_params' must be in one of group parameters.

        learning_rate (Union[float, Tensor, Iterable, LearningRateSchedule]): A value or a graph for the learning rate.
            When the learning_rate is an Iterable or a Tensor in a 1D dimension, use the dynamic learning rate, then
            the i-th step will take the i-th value as the learning rate. When the learning_rate is LearningRateSchedule,
            use dynamic learning rate, the i-th learning rate will be calculated during the process of training
            according to the formula of LearningRateSchedule. When the learning_rate is a float or a Tensor in a zero
            dimension, use fixed learning rate. Other cases are not supported. The float learning rate must be
            equal to or greater than 0. If the type of `learning_rate` is int, it will be converted to float.
            Default: 1e-3.
        beta1 (float): The exponential decay rate for the 1st moment estimations. Default: 0.9.
            Should be in range (0.0, 1.0).
        beta2 (float): The exponential decay rate for the 2nd moment estimations. Default: 0.999.
            Should be in range (0.0, 1.0).
        eps (float): Term added to the denominator to improve numerical stability. Default: 1e-6.
            Should be greater than 0.
        weight_decay (float): Weight decay (L2 penalty). It must be equal to or greater than 0. Default: 0.0.

    Inputs:
        - **gradients** (tuple[Tensor]) - The gradients of `params`, the shape is the same as `params`.

    Outputs:
        tuple[bool], all elements are True.

    Supported Platforms:
        ``GPU``

    Examples:
        >>> net = Net()
        >>> #1) All parameters use the same learning rate and weight decay
        >>> optim = AdamWeightDecayOp(params=net.trainable_params())
        >>>
        >>> #2) Use parameter groups and set different values
        >>> conv_params = list(filter(lambda x: 'conv' in x.name, net.trainable_params()))
        >>> no_conv_params = list(filter(lambda x: 'conv' not in x.name, net.trainable_params()))
        >>> group_params = [{'params': conv_params, 'weight_decay': 0.01},
        ...                 {'params': no_conv_params, 'lr': 0.01},
        ...                 {'order_params': net.trainable_params()}]
        >>> optim = AdamWeightDecayOp(group_params, learning_rate=0.1, weight_decay=0.0)
        >>> # The conv_params's parameters will use default learning rate of 0.1 and weight decay of 0.01.
        >>> # The no_conv_params's parameters will use learning rate of 0.01 and default weight decay of 0.0.
        >>> # The final parameters order in which the optimizer will be followed is the value of 'order_params'.
        >>>
        >>> loss = layers.SoftmaxCrossEntropyWithLogits()
        >>> model = Model(net)
        >>> model.compile(loss_fn=loss, optimizer=optim)
   """
    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-6, weight_decay=0.0):
        super(AdamWeightDecayOp, self).__init__(learning_rate, params, weight_decay)
        _check_param_value(beta1, beta2, eps, self.cls_name)
        self.beta1 = ts.array([beta1], dtype=ts.float32)
        self.beta2 = ts.array([beta2], dtype=ts.float32)
        self.eps = ts.array([eps], dtype=ts.float32)
        self.moments1 = self.parameters.clone(prefix="adam_m", init='zeros')
        self.moments2 = self.parameters.clone(prefix="adam_v", init='zeros')
        self.hyper_map = P.HyperMap()

    def construct(self, gradients):
        """AdamWeightDecayOp"""
        lr = self.get_lr()
        if self.is_group:
            if self.is_group_lr:
                optim_result = self.hyper_map(P.Partial()(_adam_opt, self.beta1, self.beta2, self.eps),
                                              lr, self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
            else:
                optim_result = self.hyper_map(P.Partial()(_adam_opt, self.beta1, self.beta2, self.eps, lr),
                                              self.weight_decay, self.parameters, self.moments1, self.moments2,
                                              gradients, self.decay_flags, self.optim_filter)
        else:
            optim_result = self.hyper_map(P.Partial()(_adam_opt, self.beta1, self.beta2, self.eps, lr, self.weight_decay),
                                          self.parameters, self.moments1, self.moments2,
                                          gradients, self.decay_flags, self.optim_filter)
        if self.use_parallel:
            self.broadcast_params(optim_result)
        return optim_result




class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """
    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = ts.array([warmup_steps], dtype=ts.float32)

        self.greater = P.Greater()
        self.one = ts.array([1.0], dtype=ts.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), ts.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr



def _get_poly_lr(global_step, lr_init, lr_end, lr_max, warmup_steps, total_steps, poly_power):
    """
    generate learning rate array

    Args:
       global_step(int): current step
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_steps(int): number of warmup epochs
       total_steps(int): total epoch of training
       poly_power(int): poly learning rate power

    Returns:
       np.array, learning rate array
    """
    lr_each_step = []
    if warmup_steps != 0:
        inc_each_step = (float(lr_max) - float(lr_init)) / float(warmup_steps)
    else:
        inc_each_step = 0
    for i in range(total_steps):
        if i < warmup_steps:
            lr = float(lr_init) + inc_each_step * float(i)
        else:
            base = (1.0 - (float(i) - float(warmup_steps)) / (float(total_steps) - float(warmup_steps)))
            lr = float(lr_max - lr_end) * (base ** poly_power)
            lr = lr + lr_end
            if lr < 0.0:
                lr = 0.0
        lr_each_step.append(lr)

    learning_rate = np.array(lr_each_step).astype(np.float32)
    current_step = global_step
    learning_rate = learning_rate[current_step:]
    return learning_rate



def get_bert_thor_damping(damping_max=5e-2, damping_min=1e-6, damping_power=1.0, damping_total_steps=30000):
    damping = _get_poly_lr(global_step=0, lr_init=0.0, lr_end=damping_min, lr_max=damping_max, warmup_steps=0,
                           total_steps=damping_total_steps, poly_power=damping_power)
    return Tensor(damping)


def get_bert_thor_lr(lr_max=0.0034, lr_min=3.244e-05, lr_power=1.0, lr_total_steps=30000):
    learning_rate = _get_poly_lr(global_step=0, lr_init=0.0, lr_end=lr_min, lr_max=lr_max, warmup_steps=0,
                                 total_steps=lr_total_steps, poly_power=lr_power)
    return Tensor(learning_rate)





def get_optimizer(args_opt, network, cfg, bert_net_cfg):
    """get bert optimizer, support Lamb, Momentum, AdamWeightDecay."""
    if cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=cfg.Lamb.learning_rate,
                                       end_learning_rate=cfg.Lamb.end_learning_rate,
                                       warmup_steps=cfg.Lamb.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.Lamb.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.Lamb.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.Lamb.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.Lamb.weight_decay},
                        {'params': other_params},
                        {'order_params': params}]
        optimizer = Lamb(group_params, learning_rate=lr_schedule, eps=cfg.Lamb.eps)
    elif cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=cfg.Momentum.learning_rate,
                             momentum=cfg.Momentum.momentum)
    elif cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=cfg.AdamWeightDecay.warmup_steps,
                                       decay_steps=args_opt.train_steps,
                                       power=cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0},
                        {'order_params': params}]

        if args_opt.enable_lossscale == "true" and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayForBert(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
        elif context.get_context("mode") == context.PYNATIVE_MODE and args_opt.device_target == 'GPU':
            optimizer = AdamWeightDecayOp(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)
        else:
            optimizer = AdamWeightDecay(group_params, learning_rate=lr_schedule, eps=cfg.AdamWeightDecay.eps)

    elif cfg.optimizer == "Thor":

        lr = get_bert_thor_lr(cfg.Thor.lr_max, cfg.Thor.lr_min, cfg.Thor.lr_power, cfg.Thor.lr_total_steps)
        damping = get_bert_thor_damping(cfg.Thor.damping_max, cfg.Thor.damping_min, cfg.Thor.damping_power,
                                        cfg.Thor.damping_total_steps)
        split_indices = None

        if bert_net_cfg.num_hidden_layers == 12:
            if bert_net_cfg.use_relative_positions:
                split_indices = [29, 58, 87, 116, 145, 174, 203, 217]
            else:
                split_indices = [28, 55, 82, 109, 136, 163, 190, 205]
        elif bert_net_cfg.num_hidden_layers == 24:
            if bert_net_cfg.use_relative_positions:
                split_indices = [30, 90, 150, 210, 270, 330, 390, 421]
            else:
                split_indices = [38, 93, 148, 203, 258, 313, 368, 397]

        optimizer = THOR(network, lr, damping, cfg.Thor.momentum,
                         cfg.Thor.weight_decay, cfg.Thor.loss_scale, cfg.batch_size,
                         decay_filter=lambda x: 'layernorm' not in x.name.lower() and 'bias' not in x.name.lower(),
                         split_indices=split_indices)
    else:
        raise ValueError("Don't support optimizer {}, only support [Lamb, Momentum, AdamWeightDecay, Thor]".
                         format(cfg.optimizer))

    return optimizer