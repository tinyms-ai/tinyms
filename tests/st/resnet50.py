# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""ResNet50 Tutorial
The sample can be run on CPU, GPU and Ascend 910 AI processor.
"""
import random
import argparse
from mindspore import dtype as mstype

import tinyms as ts
from tinyms import context, layers, Model
from tinyms.data import Cifar10Dataset, download_dataset
from tinyms.data.transforms import TypeCast
from tinyms.vision import RandomCrop, RandomHorizontalFlip, Resize, Rescale, Normalize, HWC2CHW
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits

random.seed(1)


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1,
                   training=True):
    """ create Cifar10 dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    cifar_ds = Cifar10Dataset(data_path)

    # define operation parameters
    resize_height = 224
    resize_width = 224
    rescale = 1.0 / 255.0
    shift = 0.0

    # define map operations
    random_crop_op = RandomCrop((32, 32), (4, 4, 4, 4))  # padding_mode default CONSTANT
    random_horizontal_op = RandomHorizontalFlip()
    resize_op = Resize((resize_height, resize_width))  # interpolation default BILINEAR
    rescale_op = Rescale(rescale, shift)
    normalize_op = Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    changeswap_op = HWC2CHW()
    type_cast_op = TypeCast(mstype.int32)

    c_trans = []
    if training:
        c_trans = [random_crop_op, random_horizontal_op]
    c_trans += [resize_op, rescale_op, normalize_op,
                changeswap_op]

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image", num_parallel_workers=num_parallel_workers)
    # apply shuffle operations
    cifar_ds = cifar_ds.shuffle(buffer_size=10)
    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)
    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_size)

    return cifar_ds


def weight_variable_0(shape):
    """weight_variable_0"""
    return ts.zeros(shape)


def weight_variable_1(shape):
    """weight_variable_1"""
    return ts.ones(shape)


def conv3x3(in_channels, out_channels, stride=1, padding=0):
    """3x3 convolution """
    return layers.Conv2d(in_channels, out_channels,
                         kernel_size=3, stride=stride, padding=padding, weight_init='XavierUniform',
                         has_bias=False, pad_mode="same")


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    """1x1 convolution"""
    return layers.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride, padding=padding, weight_init='XavierUniform',
                         has_bias=False, pad_mode="same")


def conv7x7(in_channels, out_channels, stride=1, padding=0):
    """7x7 convolution"""
    return layers.Conv2d(in_channels, out_channels,
                         kernel_size=7, stride=stride, padding=padding, weight_init='XavierUniform',
                         has_bias=False, pad_mode="same")


def bn_with_initialize(out_channels):
    """bn_with_initialize"""
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = layers.BatchNorm2d(out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                            beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def bn_with_initialize_last(out_channels):
    """bn_with_initialize_last"""
    shape = (out_channels)
    mean = weight_variable_0(shape)
    var = weight_variable_1(shape)
    beta = weight_variable_0(shape)
    bn = layers.BatchNorm2d(out_channels, momentum=0.99, eps=0.00001, gamma_init='Uniform',
                            beta_init=beta, moving_mean_init=mean, moving_var_init=var)
    return bn


def fc_with_initialize(input_channels, out_channels):
    """fc_with_initialize"""
    return layers.Dense(input_channels, out_channels, weight_init='XavierUniform', bias_init='Uniform')


class ResidualBlock(layers.Layer):
    """ResidualBlock"""
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1):
        """init block"""
        super(ResidualBlock, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.bn1 = bn_with_initialize(out_chls)
        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.bn2 = bn_with_initialize(out_chls)
        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize_last(out_channels)
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def construct(self, x):
        """construct"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = self.add(out, identity)
        out = self.relu(out)

        return out


class ResidualBlockWithDown(layers.Layer):
    """ResidualBlockWithDown"""
    expansion = 4

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride=1,
                 down_sample=False):
        """init block with down"""
        super(ResidualBlockWithDown, self).__init__()

        out_chls = out_channels // self.expansion
        self.conv1 = conv1x1(in_channels, out_chls, stride=stride, padding=0)
        self.bn1 = bn_with_initialize(out_chls)
        self.conv2 = conv3x3(out_chls, out_chls, stride=1, padding=0)
        self.bn2 = bn_with_initialize(out_chls)
        self.conv3 = conv1x1(out_chls, out_channels, stride=1, padding=0)
        self.bn3 = bn_with_initialize_last(out_channels)
        self.relu = layers.ReLU()
        self.down_sample = down_sample
        self.conv_down_sample = conv1x1(in_channels, out_channels, stride=stride, padding=0)
        self.bn_down_sample = bn_with_initialize(out_channels)
        self.add = layers.Add()

    def construct(self, x):
        """construct"""
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        identity = self.conv_down_sample(identity)
        identity = self.bn_down_sample(identity)
        out = self.add(out, identity)
        out = self.relu(out)

        return out


class MakeLayer0(layers.Layer):
    """MakeLayer0"""

    def __init__(self, block, in_channels, out_channels, stride):
        """init"""
        super(MakeLayer0, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=1, down_sample=True)
        self.b = block(out_channels, out_channels, stride=stride)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        """construct"""
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x


class MakeLayer1(layers.Layer):
    """MakeLayer1"""

    def __init__(self, block, in_channels, out_channels, stride):
        """init"""
        super(MakeLayer1, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)
        self.d = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        """construct"""
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)

        return x


class MakeLayer2(layers.Layer):
    """MakeLayer2"""

    def __init__(self, block, in_channels, out_channels, stride):
        """init"""
        super(MakeLayer2, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)
        self.d = block(out_channels, out_channels, stride=1)
        self.e = block(out_channels, out_channels, stride=1)
        self.f = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        """construct"""
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)
        x = self.d(x)
        x = self.e(x)
        x = self.f(x)

        return x


class MakeLayer3(layers.Layer):
    """MakeLayer3"""

    def __init__(self, block, in_channels, out_channels, stride):
        """init"""
        super(MakeLayer3, self).__init__()
        self.a = ResidualBlockWithDown(in_channels, out_channels, stride=stride, down_sample=True)
        self.b = block(out_channels, out_channels, stride=1)
        self.c = block(out_channels, out_channels, stride=1)

    def construct(self, x):
        """construct"""
        x = self.a(x)
        x = self.b(x)
        x = self.c(x)

        return x


class ResNet(layers.Layer):
    """ResNet"""

    def __init__(self, block, num_classes=100):
        """init"""
        super(ResNet, self).__init__()

        self.conv1 = conv7x7(3, 64, stride=2, padding=0)
        self.bn1 = bn_with_initialize(64)
        self.relu = layers.ReLU()
        self.maxpool = layers.MaxPool2d(kernel_size=3, stride=2, pad_mode="same")
        self.layer1 = MakeLayer0(block, in_channels=64, out_channels=256, stride=1)
        self.layer2 = MakeLayer1(block, in_channels=256, out_channels=512, stride=2)
        self.layer3 = MakeLayer2(block, in_channels=512, out_channels=1024, stride=2)
        self.layer4 = MakeLayer3(block, in_channels=1024, out_channels=2048, stride=2)
        self.pool = layers.ReduceMean(keep_dims=True)
        self.squeeze = layers.Squeeze(axis=(2, 3))
        self.fc = fc_with_initialize(512 * block.expansion, num_classes)

    def construct(self, x):
        """construct"""
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x, (2, 3))
        x = self.squeeze(x)
        x = self.fc(x)

        return x


def resnet50(num_classes):
    """create resnet50"""
    return ResNet(ResidualBlock, num_classes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Image classification')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Cifar10 dataset path.')
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # download cifar10 dataset
    if not args_opt.dataset_path:
        args_opt.dataset_path = download_dataset('cifar10')
    # build the network
    net = resnet50(args_opt.num_classes)
    model = Model(net)
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define the optimizer
    net_opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model.compile(loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": Accuracy()})

    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    cifar10_path = args_opt.dataset_path
    dataset_sink_mode = not args_opt.device_target == "CPU"
    if args_opt.do_eval:  # as for evaluation, users could use model.eval
        if args_opt.checkpoint_path:
            model.load_checkpoint(args_opt.checkpoint_path)
        ds_eval = create_dataset(cifar10_path, batch_size=batch_size, training=False)
        acc = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
    else:  # as for train, users could use model.train
        ds_train = create_dataset(cifar10_path, batch_size=batch_size)
        ckpoint_cb = ModelCheckpoint(prefix="resnet_cifar10", config=CheckpointConfig(
            save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=35))
        model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                    dataset_sink_mode=dataset_sink_mode)
