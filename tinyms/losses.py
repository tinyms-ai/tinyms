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
from mindspore.nn.loss.loss import _Loss
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

    def construct(self, x, gt_loc, gt_label, num_matched_boxes):
        pred_loc, pred_label = self.network(x)
        mask = P.cast(self.less(0, gt_label), ts.float32)
        num_matched_boxes = self.reduce_sum(P.cast(num_matched_boxes, ts.float32))

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


class CrossEntropyWithLabelSmooth(_Loss):
    """
    CrossEntropyWith LabelSmooth.

    Args:
        smooth_factor (float): smooth factor. Default is 0.
        num_classes (int): number of classes. Default is 1000.

    Returns:
        None.

    Examples:
        >>> CrossEntropyWithLabelSmooth(smooth_factor=0., num_classes=1000)
    """

    def __init__(self, smooth_factor=0., num_classes=1000):
        super(CrossEntropyWithLabelSmooth, self).__init__()
        self.onehot = P.OneHot()
        self.on_value = Tensor(1.0 - smooth_factor, ts.float32)
        self.off_value = Tensor(1.0 * smooth_factor /
                                (num_classes - 1), ts.float32)
        self.ce = SoftmaxCrossEntropyWithLogits()
        self.mean = P.ReduceMean(False)
        self.cast = P.Cast()

    def construct(self, logit, label):
        one_hot_label = self.onehot(self.cast(label, ts.int32), P.shape(logit)[1],
                                    self.on_value, self.off_value)
        out_loss = self.ce(logit, one_hot_label)
        out_loss = self.mean(out_loss, 0)
        return out_loss


class GANLoss(_Loss):
    """
    Cycle GAN loss factory.

    Args:
        mode (str): The type of GAN objective. It currently supports 'vanilla', 'lsgan'. Default: 'lsgan'.
        reduction (str): Specifies the reduction to be applied to the output.
            Its value must be one of 'none', 'mean', 'sum'. Default: 'none'.

    Outputs:
        Tensor or Scalar, if `reduction` is 'none', then output is a tensor and has the same shape as `inputs`.
        Otherwise, the output is a scalar.
    """
    def __init__(self, mode="lsgan", reduction='mean'):
        super(GANLoss, self).__init__()
        self.loss = None
        self.ones = P.OnesLike()
        if mode == "lsgan":
            self.loss = loss.MSELoss(reduction)
        elif mode == "vanilla":
            self.loss = BCEWithLogits(reduction)
        else:
            raise NotImplementedError(f'GANLoss {mode} not recognized, we support lsgan and vanilla.')

    def construct(self, predict, target):
        target = P.cast(target, P.dtype(predict))
        target = self.ones(predict) * target
        loss = self.loss(predict, target)
        return loss


class GeneratorLoss(_Loss):
    """
    Cycle GAN generator loss.

    Args:
        args (class): Option class.
        generator (Cell): Generator of CycleGAN.
        D_A (Cell): The discriminator network of domain A to domain B.
        D_B (Cell): The discriminator network of domain B to domain A.

    Outputs:
        Tuple Tensor, the losses of generator.
    """
    def __init__(self, generator, D_A, D_B):
        super(GeneratorLoss, self).__init__()
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_idt = 0.5
        self.use_identity = True
        self.dis_loss = GANLoss("lsgan")
        self.rec_loss = loss.L1Loss("mean")
        self.generator = generator
        self.D_A = D_A
        self.D_B = D_B
        self.true = Tensor(True, ts.bool_)

    def construct(self, img_A, img_B):
        """If use_identity, identity loss will be used."""
        fake_A, fake_B, rec_A, rec_B, identity_A, identity_B = self.generator(img_A, img_B)
        loss_G_A = self.dis_loss(self.D_B(fake_B), self.true)
        loss_G_B = self.dis_loss(self.D_A(fake_A), self.true)
        loss_C_A = self.rec_loss(rec_A, img_A) * self.lambda_A
        loss_C_B = self.rec_loss(rec_B, img_B) * self.lambda_B
        if self.use_identity:
            loss_idt_A = self.rec_loss(identity_A, img_A) * self.lambda_A * self.lambda_idt
            loss_idt_B = self.rec_loss(identity_B, img_B) * self.lambda_B * self.lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0
        loss_G = loss_G_A + loss_G_B + loss_C_A + loss_C_B + loss_idt_A + loss_idt_B
        return (fake_A, fake_B, loss_G, loss_G_A, loss_G_B, loss_C_A, loss_C_B, loss_idt_A, loss_idt_B)


class DiscriminatorLoss(_Loss):
    """
    Cycle GAN discriminator loss.

    Args:
        args (class): option class.
        D_A (Cell): The discriminator network of domain A to domain B.
        D_B (Cell): The discriminator network of domain B to domain A.

    Outputs:
        Tuple Tensor, the loss of discriminator.
    """
    def __init__(self, D_A, D_B, reduction='none'):
        super(DiscriminatorLoss, self).__init__()
        self.D_A = D_A
        self.D_B = D_B
        self.false = Tensor(False, ts.bool_)
        self.true = Tensor(True, ts.bool_)
        self.dis_loss = GANLoss("lsgan")
        self.rec_loss = loss.L1Loss("mean")
        self.reduction = reduction

    def construct(self, img_A, img_B, fake_A, fake_B):
        D_fake_A = self.D_A(fake_A)
        D_img_A = self.D_A(img_A)
        D_fake_B = self.D_B(fake_B)
        D_img_B = self.D_B(img_B)
        loss_D_A = self.dis_loss(D_fake_A, self.false) + self.dis_loss(D_img_A, self.true)
        loss_D_B = self.dis_loss(D_fake_B, self.false) + self.dis_loss(D_img_B, self.true)
        loss_D = (loss_D_A + loss_D_B) * 0.5
        return loss_D

