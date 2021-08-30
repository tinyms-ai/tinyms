import tinyms as ts
from mindspore.nn.loss.loss import _Loss
from tinyms import Tensor
from tinyms import layers
from tinyms import losses


class CycleGANDiscriminatorLoss(_Loss):
    """
    Cycle GAN discriminator loss.

    Args:
        D_A (layers.Layer): The discriminator network of domain A to domain B.
        D_B (layers.Layer): The discriminator network of domain B to domain A.
        reduction (str): The discriminator network of reduction. Default: none.

    Outputs:
        the loss of discriminator.
    """

    def __init__(self, D_A, D_B, reduction='none'):
        super(CycleGANDiscriminatorLoss, self).__init__()
        self.D_A = D_A
        self.D_B = D_B
        self.false = Tensor(False, ts.bool_)
        self.true = Tensor(True, ts.bool_)
        self.dis_loss = losses.GANLoss("lsgan")
        self.rec_loss = losses.L1Loss("mean")
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


class CycleGANGeneratorLoss(_Loss):
    """
    Cycle GAN generator loss.

    Args:
        generator (layers.Layer): Generator of CycleGAN.
        D_A (layers.Layer): The discriminator network of domain A to domain B.
        D_B (layers.Layer): The discriminator network of domain B to domain A.

    Outputs:
        Tuple Tensor, the losses of generator.
    """

    def __init__(self, args, nets):
        super(CycleGANGeneratorLoss, self).__init__()
        self.args = args
        self.lambda_sem = args.lambda_sem
        self.lambda_A = 10.0
        self.lambda_B = 10.0
        self.lambda_idt = 0.5
        self.use_identity = True
        self.dis_loss = losses.GANLoss("lsgan")
        self.rec_loss = losses.L1Loss("mean")
        if args.lambda_sem != 0:
            self.sem_loss = losses.L1Loss("mean")
            self.G_S = nets.G_S
        self.generator = nets.generator
        self.D_A = nets.D_A
        self.D_B = nets.D_B
        self.true = Tensor(True, ts.bool_)

    def construct(self, img_A, img_B, img_S=None):
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
        if self.lambda_sem != 0:
            loss_sem = self.sem_loss(self.G_S(fake_A), self.G_S(fake_B)) * self.lambda_sem
        else:
            loss_sem = 0
        loss_G = loss_G_A + loss_G_B + loss_C_A + loss_C_B + loss_idt_A + loss_idt_B + loss_sem
        return fake_A, fake_B, loss_G, loss_G_A, loss_G_B, loss_C_A, loss_C_B, loss_idt_A, loss_idt_B, loss_sem


class SemanticDiscriminatorLoss(_Loss):
    def __init__(self, D_S, reduction='none'):
        super(SemanticDiscriminatorLoss, self).__init__()
        self.D_S = D_S
        self.false = Tensor(False, ts.bool_)
        self.true = Tensor(True, ts.bool_)
        self.dis_loss = losses.GANLoss("lsgan")
        self.reduction = reduction

    def construct(self, img_S, fake_S_A, fake_S_B):
        D_fake_S_A = self.D_S(fake_S_A)
        D_fake_S_B = self.D_S(fake_S_B)
        D_img_S = self.D_S(img_S)
        loss_D_S = self.dis_loss(D_fake_S_A, self.false) + self.dis_loss(D_fake_S_B, self.false) + self.dis_loss(
            D_img_S, self.true)
        return loss_D_S


class SemanticGeneratorLoss(_Loss):
    def __init__(self, G_S, D_S):
        super(SemanticGeneratorLoss, self).__init__()
        self.G_S = G_S
        self.D_S = D_S
        self.true = Tensor(True, ts.bool_)
        self.dis_loss = losses.GANLoss("lsgan")

    def construct(self, img_A, img_B):
        fake_S_A = self.G_S(img_A)
        fake_S_B = self.G_S(img_B)
        loss_S_A = self.dis_loss(self.D_S(fake_S_A), self.true)
        loss_S_B = self.dis_loss(self.D_S(fake_S_B), self.true)
        loss_G_S = loss_S_A + loss_S_B
        return fake_S_A, fake_S_B, loss_G_S, loss_S_A, loss_S_B
