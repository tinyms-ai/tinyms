from tinyms import layers
from tinyms.model.cycle_gan.common_net import ConvNormReLU


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
