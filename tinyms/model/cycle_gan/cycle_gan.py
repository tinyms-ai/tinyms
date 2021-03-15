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

"""Cycle GAN network."""

import tinyms as ts
from tinyms import layers
from tinyms.primitives import OnesLike, GradOperation, Fill, DType, Shape, depend
from .resnet import ResNetGenerator
from .unet import UnetGenerator
from .common_net import ConvNormReLU, init_weights


def get_generator(model):
    """Return generator by model."""
    if model == "resnet":
        net = ResNetGenerator(in_planes=3, ngf=64, n_layers=9, alpha=0.2,
                              norm_mode='instance', dropout=True, pad_mode='CONSTANT')
        init_weights(net, init_type='normal', init_gain=0.02)
    elif model == "unet":
        net = UnetGenerator(in_planes=3, out_planes=3, ngf=64, n_layers=9,
                            alpha=0.2, norm_mode='instance', dropout=True)
        init_weights(net, init_type='normal', init_gain=0.02)
    else:
        raise NotImplementedError(f'Model {model} not recognized.')
    return net


def get_discriminator():
    """Return discriminator."""
    net = Discriminator(in_planes=3, ndf=64, n_layers=3, alpha=0.2, norm_mode='instance')
    init_weights(net, init_type='normal', init_gain=0.02)
    return net


class Discriminator(layers.Layer):
    """
    Discriminator of GAN.

    Args:
        in_planes (int): Input channel.
        ndf (int): Output channel.
        n_layers (int): The number of ConvNormReLU blocks.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".

    Returns:
        Tensor, output tensor.

    Examples:
        >>> Discriminator(3, 64, 3)
    """
    def __init__(self, in_planes=3, ndf=64, n_layers=3, alpha=0.2, norm_mode='batch'):
        super(Discriminator, self).__init__()
        kernel_size = 4
        layer_list = [
            layers.Conv2d(in_planes, ndf, kernel_size, 2, pad_mode='pad', padding=1),
            layers.LeakyReLU(alpha)
        ]
        nf_mult = ndf
        for i in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** i, 8) * ndf
            layer_list.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 2, alpha, norm_mode, padding=1))
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8) * ndf
        layer_list.append(ConvNormReLU(nf_mult_prev, nf_mult, kernel_size, 1, alpha, norm_mode, padding=1))
        layer_list.append(layers.Conv2d(nf_mult, 1, kernel_size, 1, pad_mode='pad', padding=1))
        self.features = layers.SequentialLayer(layer_list)

    def construct(self, x):
        output = self.features(x)
        return output


class Generator(layers.Layer):
    """
    Generator of CycleGAN, return fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.

    Args:
        G_A (Cell): The generator network of domain A to domain B.
        G_B (Cell): The generator network of domain B to domain A.
        use_identity (bool): Use identity loss or not. Default: True.

    Returns:
        Tensors, fake_A, fake_B, rec_A, rec_B, identity_A and identity_B.

    Examples:
        >>> Generator(G_A, G_B)
    """

    def __init__(self, G_A, G_B, use_identity=True):
        super(Generator, self).__init__()
        self.G_A = G_A
        self.G_B = G_B
        self.ones = OnesLike()
        self.use_identity = use_identity

    def construct(self, img_A, img_B):
        """If use_identity, identity loss will be used."""
        fake_A = self.G_B(img_B)
        fake_B = self.G_A(img_A)
        rec_A = self.G_B(fake_B)
        rec_B = self.G_A(fake_A)
        if self.use_identity:
            identity_A = self.G_B(img_A)
            identity_B = self.G_A(img_B)
        else:
            identity_A = self.ones(img_A)
            identity_B = self.ones(img_B)
        return fake_A, fake_B, rec_A, rec_B, identity_A, identity_B


class WithLossCell(layers.Layer):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Cell): The target network to wrap.
    """
    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_A, img_B):
        _, _, lg, _, _, _, _, _, _ = self.network(img_A, img_B)
        return lg


class TrainOneStepG(layers.Layer):
    """
    Encapsulation class of Cycle GAN generator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        G (Cell): Generator with loss Cell. Note that loss function should have been added.
        generator (Cell): Generator of CycleGAN.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, G, generator, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.G = G
        self.G.set_grad()
        self.G.set_train()
        self.G.D_A.set_grad(False)
        self.G.D_A.set_train(False)
        self.G.D_B.set_grad(False)
        self.G.D_B.set_train(False)
        self.grad = GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ts.ParameterTuple(generator.trainable_params())
        self.net = WithLossCell(G)

    def construct(self, img_A, img_B):
        weights = self.weights
        fake_A, fake_B, lg, lga, lgb, lca, lcb, lia, lib = self.G(img_A, img_B)
        sens = Fill()(DType()(lg), Shape()(lg), self.sens)
        grads_g = self.grad(self.net, weights)(img_A, img_B, sens)
        return fake_A, fake_B, depend(lg, self.optimizer(grads_g)), lga, lgb, lca, lcb, lia, lib


class TrainOneStepD(layers.Layer):
    """
    Encapsulation class of Cycle GAN discriminator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        G (Cell): Generator with loss Cell. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepD, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ts.ParameterTuple(D.trainable_params())

    def construct(self, img_A, img_B, fake_A, fake_B):
        weights = self.weights
        ld = self.D(img_A, img_B, fake_A, fake_B)
        sens_d = Fill()(DType()(ld), Shape()(ld), self.sens)
        grads_d = self.grad(self.D, weights)(img_A, img_B, fake_A, fake_B, sens_d)
        return depend(ld, self.optimizer(grads_d))


def get_generator_discriminator(model='resnet'):
    """
    Get G_A, G_B generator network and  D_A, D_B discriminator network.

    Args:
        model: generator model, should be in [resnet, unet].

    Returns:
        G_A, G_B, D_A, D_B network.

    Examples:
        >>> G_A, G_B, D_A, D_B = cycle_gan('resnet')
    """
    if model not in ['resnet', 'unet']:
        raise NotImplementedError(f'Model {model} not recognized.')

    G_A = get_generator(model)
    G_B = get_generator(model)
    D_A = get_discriminator()
    D_B = get_discriminator()

    return G_A, G_B, D_A, D_B


def cycle_gan(G_A, G_B):
    """
        Get Cycle GAN network.

        Args:
            G_A: generator net, should be in [resnet, unet].
            G_B: generator net, should be in [resnet, unet].

        Returns:
            Cycle GAN network.

        Examples:
            >>> gan_net = cycle_gan(G_A, G_B)
        """
    if not isinstance(G_A, layers.Layer) or not isinstance(G_B, layers.Layer):
        raise NotImplementedError(f'G_A and G_B are not the instance of layers.Layer')
    return Generator(G_A, G_B)


def cycle_gan_infer(g_model='resnet'):
    if g_model not in ['resnet', 'unet']:
        raise NotImplementedError(f'Model {g_model} not recognized.')

    G_A = get_generator(g_model)
    G_B = get_generator(g_model)
    return G_A, G_B