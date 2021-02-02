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
"""Lenet5 Tutorial
The sample can be run on CPU, GPU and Ascend 910 AI processor.
"""
import os
import argparse
from mindspore import dtype as mstype

from tinyms import context, layers, Model
from tinyms.data import MnistDataset, download_dataset
from tinyms.data.transforms import TypeCast
from tinyms.vision import Inter, Resize, Rescale, HWC2CHW
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits
from tinyms.initializers import Normal


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """ create Mnist dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset
    mnist_ds = MnistDataset(data_path)

    # define operation parameters
    resize_height, resize_width = 32, 32
    rescale = 1.0 / 255.0
    shift = 0.0
    rescale_nml = 1 / 0.3081
    shift_nml = -1 * 0.1307 / 0.3081

    # define map operations
    resize_op = Resize((resize_height, resize_width), interpolation=Inter.LINEAR)  # Resize images to (32, 32)
    rescale_nml_op = Rescale(rescale_nml, shift_nml)  # normalize images
    rescale_op = Rescale(rescale, shift)  # rescale images
    hwc2chw_op = HWC2CHW()  # change shape from (height, width, channel) to (channel, height, width) to fit network.
    type_cast_op = TypeCast(mstype.int32)  # change data type of label to int32 to fit network

    # apply map operations on images
    mnist_ds = mnist_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=resize_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=rescale_nml_op, input_columns="image", num_parallel_workers=num_parallel_workers)
    mnist_ds = mnist_ds.map(operations=hwc2chw_op, input_columns="image", num_parallel_workers=num_parallel_workers)

    # apply DatasetOps
    buffer_size = 10000
    mnist_ds = mnist_ds.shuffle(buffer_size=buffer_size)  # 10000 as in LeNet train script
    mnist_ds = mnist_ds.batch(batch_size, drop_remainder=True)
    mnist_ds = mnist_ds.repeat(repeat_size)

    return mnist_ds


class LeNet5(layers.Layer):
    """Lenet network structure."""
    # define the operator required

    def __init__(self, num_class=10, num_channel=1):
        super(LeNet5, self).__init__()
        self.conv1 = layers.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = layers.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = layers.Dense(16 * 5 * 5, 120, weight_init=Normal(0.02))
        self.fc2 = layers.Dense(120, 84, weight_init=Normal(0.02))
        self.fc3 = layers.Dense(84, num_class, weight_init=Normal(0.02))
        self.relu = layers.ReLU()
        self.max_pool2d = layers.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = layers.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Mnist dataset path.')
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # download mnist dataset
    if not args_opt.dataset_path:
        args_opt.dataset_path = download_dataset('mnist')
    # build the network
    net = LeNet5()
    model = Model(net)
    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
    # define the optimizer
    net_opt = Momentum(net.trainable_params(), 0.01, 0.9)
    model.compile(net_loss, net_opt, metrics={"Accuracy": Accuracy()})

    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    mnist_path = args_opt.dataset_path
    dataset_sink_mode = not args_opt.device_target == "CPU"

    if args_opt.do_eval:  # as for evaluation, users could use model.eval
        print("============== Starting Evaluating ==============")
        # load the saved model for evaluation
        model.load_checkpoint(args_opt.checkpoint_path)
        # load testing dataset
        ds_eval = create_dataset(os.path.join(mnist_path, "test"))
        acc = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
    else:  # as for train, users could use model.train
        print("============== Starting Training ==============")
        # load training dataset
        ds_train = create_dataset(os.path.join(mnist_path, "train"), batch_size=batch_size)
        # save the network model and parameters for subsequence fine-tuning
        ckpoint_cb = ModelCheckpoint(prefix="checkpoint_lenet", config=CheckpointConfig(
            save_checkpoint_steps=1875, keep_checkpoint_max=10))
        model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                    dataset_sink_mode=dataset_sink_mode)
