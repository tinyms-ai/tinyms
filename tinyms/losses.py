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

from mindspore.nn import loss
from mindspore.nn.loss import *
import tinyms as ts
from . import layers, primitives as P, Tensor
from .model import SSD300

__all__ = ['net_with_loss', 'SSD300WithLoss']
__all__.extend(loss.__all__)


class SigmoidFocalClassificationLoss(layers.Layer):
    """"
    Sigmoid focal-loss for classification.

    Args:
        gamma (float): Hyper-parameter to balance the easy and hard examples. Default: 2.0
        alpha (float): Hyper-parameter to balance the positive and negative example. Default: 0.25

    Returns:
        Tensor, the focal loss.
    """

    def __init__(self, gamma=2.0, alpha=0.25):
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.sigmiod_cross_entropy = P.SigmoidCrossEntropyWithLogits()
        self.sigmoid = P.Sigmoid()
        self.pow = P.Pow()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0, ts.float32)
        self.off_value = Tensor(0.0, ts.float32)
        self.gamma = gamma
        self.alpha = alpha

    def construct(self, logits, label):
        label = self.onehot(label, P.shape(logits)[-1], self.on_value, self.off_value)
        sigmiod_cross_entropy = self.sigmiod_cross_entropy(logits, label)
        sigmoid = self.sigmoid(logits)
        label = P.cast(label, ts.float32)
        p_t = label * sigmoid + (1 - label) * (1 - sigmoid)
        modulating_factor = self.pow(1 - p_t, self.gamma)
        alpha_weight_factor = label * self.alpha + (1 - label) * (1 - self.alpha)
        focal_loss = modulating_factor * alpha_weight_factor * sigmiod_cross_entropy
        return focal_loss


class SSD300WithLoss(layers.Layer):
    """"
    Provide SSD300 training loss through network.

    Args:
        network (Layer): The training network.

    Returns:
        Tensor, the loss of the network.

    Examples:
        net = SSD300WithLoss(ssd300())
    """

    def __init__(self, network):
        super(SSD300WithLoss, self).__init__()
        self.network = network
        self.less = P.Less()
        self.tile = P.Tile()
        self.reduce_sum = P.ReduceSum()
        self.reduce_mean = P.ReduceMean()
        self.expand_dims = P.ExpandDims()
        self.class_loss = SigmoidFocalClassificationLoss(2.0, 0.75)
        self.loc_loss = SmoothL1Loss()

    def construct(self, x, gt_loc, gt_label):
        pred_loc, pred_label = self.network(x)
        mask = P.cast(self.less(0, gt_label), ts.float32)
        num_matched_boxes = P.cast(P.count_nonzero(gt_label), ts.float32)

        # Localization Loss
        mask_loc = self.tile(self.expand_dims(mask, -1), (1, 1, 4))
        smooth_l1 = self.loc_loss(pred_loc, gt_loc) * mask_loc
        loss_loc = self.reduce_sum(self.reduce_mean(smooth_l1, -1), -1)

        # Classification Loss
        loss_cls = self.class_loss(pred_label, gt_label)
        loss_cls = self.reduce_sum(loss_cls, (1, 2))

        return self.reduce_sum((loss_cls + loss_loc) / num_matched_boxes)


def net_with_loss(net):
    if not isinstance(net, layers.Layer):
        raise TypeError("Input should be inheritted from layers.Layer!")

    if isinstance(net, SSD300):
        return SSD300WithLoss(net)
    else:
        raise TypeError("Input should be in [SSD300], got {}.".format(type(net)))
