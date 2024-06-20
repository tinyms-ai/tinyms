import tinyms as ts
from tinyms import layers
from tinyms.primitives import GradOperation, Fill, DType, Shape, depend


class WithLossCell(layers.Layer):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Layer): The target network to wrap.

    Returns:
       Generator Loss lg
    """

    def __init__(self, network):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_A, img_B):
        _, _, lg, _, _, _, _, _, _, _ = self.network(img_A, img_B)
        return lg


class WithLossCell2(layers.Layer):
    def __init__(self, network):
        super(WithLossCell2, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_A, img_B, img_S=None):
        _, _, lg, _, _, _, _, _, _, _ = self.network(img_A, img_B, img_S)
        return lg


class TrainOneStepG(layers.Layer):
    """
    Encapsulation class of Cycle GAN generator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        loss_G (layers.Layer): Generator with loss Layer. Note that loss function should have been added.
        combined_G (layers.Layer): Generator of CycleGAN.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (Number): The adjust parameter. Default: 1.0.
    """

    def __init__(self, args, loss_G, combined_G, optimizer, sens=1.0):
        super(TrainOneStepG, self).__init__(auto_prefix=False)
        self.args = args
        self.lambda_sem = args.lambda_sem
        self.optimizer = optimizer
        self.loss_G = loss_G
        self.loss_G.set_grad()
        self.loss_G.set_train()
        self.loss_G.D_A.set_grad(False)
        self.loss_G.D_A.set_train(False)
        self.loss_G.D_B.set_grad(False)
        self.loss_G.D_B.set_train(False)
        if args.lambda_sem != 0:
            self.loss_G.G_S.set_grad(False)
            self.loss_G.G_S.set_train(False)
            self.net = WithLossCell2(loss_G)
        else:
            self.net = WithLossCell(loss_G)
        self.grad = GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ts.ParameterTuple(combined_G.trainable_params())

    def construct(self, img_A, img_B, img_S=None):
        weights = self.weights
        fake_A, fake_B, loss_G, loss_G_A, loss_G_B, loss_C_A, loss_C_B, loss_idt_A, loss_idt_B, loss_sem = \
            self.loss_G(img_A, img_B, img_S)
        sens = Fill()(DType()(loss_G), Shape()(loss_G), self.sens)
        if self.lambda_sem != 0:
            grads_g = self.grad(self.net, weights)(img_A, img_B, img_S, sens)
        else:
            grads_g = self.grad(self.net, weights)(img_A, img_B, sens)
        depend_loss_G = depend(loss_G, self.optimizer(grads_g))

        return fake_A, fake_B, depend_loss_G, loss_G_A, loss_G_B, loss_C_A, loss_C_B, loss_idt_A, loss_idt_B, loss_sem


class TrainOneStepD(layers.Layer):
    """
    Encapsulation class of Cycle GAN discriminator network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        D (layers.Layer): Generator with loss Layer. Note that loss function should have been added.
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
        loss_D = self.D(img_A, img_B, fake_A, fake_B)
        sens_d = Fill()(DType()(loss_D), Shape()(loss_D), self.sens)
        grads_d = self.grad(self.D, weights)(img_A, img_B, fake_A, fake_B, sens_d)
        depend_loss_D = depend(loss_D, self.optimizer(grads_d))
        return depend_loss_D


class WithLossCellGS(layers.Layer):
    """
    Wrap the network with loss function to return generator loss.

    Args:
        network (Layer): The target network to wrap.

    Returns:
       Generator Loss lg
    """

    def __init__(self, network):
        super(WithLossCellGS, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, img_A, img_B):
        _, _, lg, _, _ = self.network(img_A, img_B)
        return lg


class TrainOneStepGS(layers.Layer):
    def __init__(self, G, D, optimizer, sens=1.0):
        super(TrainOneStepGS, self).__init__(auto_prefix=False)
        self.G = G
        self.D = D
        self.optimizer = optimizer
        self.G.set_grad()
        self.G.set_train()
        self.D.set_grad(False)
        self.D.set_train(False)
        self.grad = GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ts.ParameterTuple(G.trainable_params())
        self.net = WithLossCellGS(G)

    def construct(self, img_A, img_B):
        weights = self.weights
        fake_S_A, fake_S_B, loss_G_S, loss_S_A, loss_S_B = self.G(img_A, img_B)
        sens = Fill()(DType()(loss_G_S), Shape()(loss_G_S), self.sens)
        grads_g = self.grad(self.net, weights)(img_A, img_B, sens)
        depend_loss_G_S = depend(loss_G_S, self.optimizer(grads_g))
        return fake_S_A, fake_S_B, depend_loss_G_S, loss_S_A, loss_S_B


class TrainOneStepDS(layers.Layer):
    def __init__(self, D, optimizer, sens=1.0):
        super(TrainOneStepDS, self).__init__(auto_prefix=False)
        self.optimizer = optimizer
        self.D = D
        self.D.set_grad()
        self.D.set_train()
        self.grad = GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.weights = ts.ParameterTuple(D.trainable_params())

    def construct(self, img_S, fake_S_A, fake_S_B):
        weights = self.weights
        loss_D = self.D(img_S, fake_S_A, fake_S_B)
        sens_d = Fill()(DType()(loss_D), Shape()(loss_D), self.sens)
        grads_d = self.grad(self.D, weights)(img_S, fake_S_A, fake_S_B, sens_d)
        depend_loss_D = depend(loss_D, self.optimizer(grads_d))
        return depend_loss_D
