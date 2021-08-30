from tinyms import layers
from models.layers import UnetSkipConnectionBlock


class UnetGenerator(layers.Layer):
    """
    Unet-based generator.

    Args:
        in_planes (int): the number of channels in input images.
        out_planes (int): the number of channels in output images.
        ngf (int): the number of filters in the last conv layer.
        n_layers (int): the number of downsamplings in UNet.
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        dropout (bool): Use dropout or not. Default: False.

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, in_planes, out_planes, ngf=64, n_layers=7, alpha=0.2, norm_mode='bn', dropout=False):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, in_planes=None, submodule=None,
                                             norm_mode=norm_mode, innermost=True)
        for _ in range(n_layers - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, in_planes=None, submodule=unet_block,
                                                 norm_mode=norm_mode, dropout=dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, in_planes=None, submodule=unet_block,
                                             norm_mode=norm_mode)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, in_planes=None, submodule=unet_block,
                                             norm_mode=norm_mode)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, in_planes=None, submodule=unet_block, norm_mode=norm_mode)
        self.model = UnetSkipConnectionBlock(out_planes, ngf, in_planes=in_planes, submodule=unet_block,
                                             outermost=True, norm_mode=norm_mode)

    def construct(self, x):
        return self.model(x)
