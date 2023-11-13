from tinyms import layers
from tinyms.model.cycle_gan.common_net import ConvNormReLU
from tinyms.primitives import Concat


class ResidualBlock(layers.Layer):
    """
    ResNet residual block definition.

    Args:
        dim (int): Input and output channel.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".
        dropout (bool): Use dropout or not. Default: False.
        pad_mode (str): Specifies padding mode. The optional values are "CONSTANT", "REFLECT", "SYMMETRIC".
            Default: "CONSTANT".

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, dim, norm_mode='batch', dropout=True, pad_mode="CONSTANT"):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode)
        self.conv2 = ConvNormReLU(dim, dim, 3, 1, 0, norm_mode, pad_mode, use_relu=False)
        self.dropout = dropout
        if dropout:
            self.dropout = layers.Dropout(0.5)

    def construct(self, x):
        out = self.conv1(x)
        if self.dropout:
            out = self.dropout(out)
        out = self.conv2(out)
        return x + out


class UnetSkipConnectionBlock(layers.Layer):
    """Unet submodule with skip connection.

    Args:
        outer_nc (int): The number of filters in the outer conv layer
        inner_nc (int): The number of filters in the inner conv layer
        in_planes (int): The number of channels in input images/features
        dropout (bool): Use dropout or not. Default: False.
        submodule (Layer): Previously defined submodules
        outermost (bool): If this module is the outermost module
        innermost (bool): If this module is the innermost module
        alpha (float): LeakyRelu slope. Default: 0.2.
        norm_mode (str): Specifies norm method. The optional values are "batch", "instance".

    Returns:
        Tensor, output tensor.
    """

    def __init__(self, outer_nc, inner_nc, in_planes=None, dropout=False,
                 submodule=None, outermost=False, innermost=False, alpha=0.2, norm_mode='batch'):
        super(UnetSkipConnectionBlock, self).__init__()
        downnorm = layers.BatchNorm2d(inner_nc)
        upnorm = layers.BatchNorm2d(outer_nc)
        use_bias = False
        if norm_mode == 'instance':
            downnorm = layers.BatchNorm2d(inner_nc, affine=False)
            upnorm = layers.BatchNorm2d(outer_nc, affine=False)
            use_bias = True
        if in_planes is None:
            in_planes = outer_nc
        downconv = layers.Conv2d(in_planes, inner_nc, kernel_size=4,
                                 stride=2, padding=1, has_bias=use_bias, pad_mode='pad')
        downrelu = layers.LeakyReLU(alpha)
        uprelu = layers.ReLU()

        if outermost:
            upconv = layers.Conv2dTranspose(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, pad_mode='pad')
            down = [downconv]
            up = [uprelu, upconv, layers.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = layers.Conv2dTranspose(inner_nc, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, has_bias=use_bias, pad_mode='pad')
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = layers.Conv2dTranspose(inner_nc * 2, outer_nc,
                                            kernel_size=4, stride=2,
                                            padding=1, has_bias=use_bias, pad_mode='pad')
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up
            if dropout:
                model.append(layers.Dropout(0.5))

        self.model = layers.SequentialLayer(model)
        self.skip_connections = not outermost
        self.concat = Concat(axis=1)

    def construct(self, x):
        out = self.model(x)
        if self.skip_connections:
            out = self.concat((out, x))
        return out
