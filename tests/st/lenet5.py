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

from tinyms import context
from tinyms.data import MnistDataset, download_dataset
from tinyms.vision import mnist_transform
from tinyms.model import Model, lenet5
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits


def parse_args():
    parser = argparse.ArgumentParser(description='MindSpore LeNet Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Mnist dataset path.')
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
    args_opt = parser.parse_args()

    return args_opt


def create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=1):
    """ create Mnist dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset and apply the transform func
    mnist_ds = MnistDataset(data_path, num_parallel_workers=num_parallel_workers,
                            shuffle=True)
    mnist_ds = mnist_transform.apply_ds(mnist_ds,
                                        repeat_size=repeat_size,
                                        batch_size=batch_size,
                                        num_parallel_workers=num_parallel_workers)

    return mnist_ds


if __name__ == "__main__":
    args_opt = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # download mnist dataset
    if not args_opt.dataset_path:
        args_opt.dataset_path = download_dataset('mnist')
    # build the network
    net = lenet5()
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
        # load testing dataset
        ds_eval = create_dataset(os.path.join(mnist_path, "test"))
        # load the saved model for evaluation
        if args_opt.checkpoint_path:
            model.load_checkpoint(args_opt.checkpoint_path)
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
