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
from tinyms import layers
from tinyms import ParameterTuple

from tinyms.context import ParallelMode, get_auto_parallel_context
from tinyms.layers import DistributedGradReducer
from tinyms.optimizers import Adam
from tinyms.primitives import HyperMap, GradOperation
from tinyms.primitives import depend
from tinyms.primitives import DType, Fill, ReduceMean, ReduceSum, Shape, Square
from tinyms.primitives import SigmoidCrossEntropyWithLogits
from mindspore.communication.management import get_group_size


class DeepFMWithLoss(layers.Layer):
    """
    Provide DeepFM training loss through network.

    Args:
        network (layers.Layer): The training network.
        l2_coef (float): value for l2 loss. Default: 1e-6.

    Returns:
        Tensor, the loss of the network.

    Examples:
        >>> from tinyms.model import deepfm, DeepFMWithLoss, DeepFMTrainModel
        >>>
        >>> net = deepfm()
        >>> train_net = DeepFMTrainModel(DeepFMWithLoss(net))
    """
    def __init__(self, network, l2_coef=1e-6):
        super(DeepFMWithLoss, self).__init__(auto_prefix=False)
        self.network = network
        self.l2_coef = l2_coef
        self.square = Square()
        self.reduce_mean = ReduceMean(keep_dims=False)
        self.reduce_sum = ReduceSum(keep_dims=False)
        self.loss = SigmoidCrossEntropyWithLogits()

    def construct(self, ids, wts, labels):
        predicts, fm_id_weight, fm_id_embeds = self.network(ids, wts)
        log_loss = self.loss(predicts, labels)
        mean_log_loss = self.reduce_mean(log_loss)
        l2_loss_w = self.reduce_sum(self.square(fm_id_weight))
        l2_loss_v = self.reduce_sum(self.square(fm_id_embeds))
        l2_loss_all = self.l2_coef * (l2_loss_v + l2_loss_w) * 0.5
        loss_df = mean_log_loss + l2_loss_all
        return loss_df


class DeepFMTrainModel(layers.Layer):
    """
    Provide DeepFM training network.

    Args:
        network (layers.Layer): The base network.
        learning_rate (float): A value or a graph for the learning rate. Default: 0.0005.
        eps (float): Term added to the denominator to improve numerical stability.
        Should be greater than 0. Default: 0.00000005.
        loss_scale (float): A floating point value for the loss scale.
        Should be greater than 0. Default: 1024.0.

    Returns:
        Tensor, the value passed by last operator.

    Examples:
        >>> from tinyms.model import deepfm, DeepFMWithLoss, DeepFMTrainModel
        >>>
        >>> net = deepfm()
        >>> train_net = DeepFMTrainModel(DeepFMWithLoss(net))
    """
    def __init__(self, network, learning_rate=0.0005, eps=0.00000005, loss_scale=1024.0):
        super(DeepFMTrainModel, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_train()
        self.weights = ParameterTuple(network.trainable_params())
        self.optimizer = Adam(self.weights, learning_rate=learning_rate, eps=eps, loss_scale=loss_scale)
        self.hyper_map = HyperMap()
        self.grad = GradOperation(get_by_list=True, sens_param=True)
        self.sens = loss_scale

        self.reducer_flag = False
        self.grad_reducer = None
        parallel_mode = get_auto_parallel_context("parallel_mode")
        if parallel_mode in (ParallelMode.DATA_PARALLEL, ParallelMode.HYBRID_PARALLEL):
            self.reducer_flag = True
        if self.reducer_flag:
            mean = get_auto_parallel_context("gradients_mean")
            degree = get_group_size()
            self.grad_reducer = DistributedGradReducer(self.optimizer.parameters, mean, degree)

    def construct(self, ids, wts, labels):
        weights = self.weights
        loss = self.network(ids, wts, labels)
        sens = Fill()(DType()(loss), Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(ids, wts, labels, sens)
        if self.reducer_flag:
            # apply grad reducer on grads
            grads = self.grad_reducer(grads)
        return depend(loss, self.optimizer(grads))
