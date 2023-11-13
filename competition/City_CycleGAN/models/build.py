from munch import Munch
from tinyms.model.cycle_gan.common_net import init_weights

from models.combined_generator import CombinedGenerator
from models.discriminator import Discriminator
from models.resnet_generator import ResNetGenerator
from models.unet_generator import UnetGenerator


def build_model(args):
    G_A = get_generator(args.g_arch)
    G_B = get_generator(args.g_arch)
    D_A = get_discriminator()
    D_B = get_discriminator()
    combined_G = CombinedGenerator(G_A, G_B)

    nets = Munch(G_A=G_A, G_B=G_B, D_A=D_A, D_B=D_B)

    # Semantic related
    if args.lambda_sem != 0:
        G_S = get_generator(args.sem_g_arch)
        D_S = get_discriminator()
        nets.G_S = G_S
        nets.D_S = D_S

    return nets, combined_G


def get_generator(model):
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
    net = Discriminator(in_planes=3, ndf=64, n_layers=3, alpha=0.2, norm_mode='instance')
    init_weights(net, init_type='normal', init_gain=0.02)
    return net
