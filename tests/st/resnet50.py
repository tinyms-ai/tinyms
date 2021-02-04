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

import tinyms as ts
from tinyms import context
from tinyms.data import Cifar10Dataset, download_dataset
from tinyms.data.transforms import TypeCast
from tinyms.vision import RandomCrop, RandomHorizontalFlip, Resize, Rescale, Normalize, HWC2CHW
from tinyms.model import Model, resnet50
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
    cifar_ds = Cifar10Dataset(data_path, num_parallel_workers=num_parallel_workers,
                              shuffle=True)

    # define map operations
    c_trans = []
    if training:
        c_trans += [
            RandomCrop((32, 32), (4, 4, 4, 4)),
            RandomHorizontalFlip(prob=0.5),
        ]
    c_trans += [
        Resize((224, 224)), Rescale(1.0 / 255.0, 0.0),
        Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),
        HWC2CHW(),
    ]
    type_cast_op = TypeCast(ts.int32)

    # apply map operations on images
    cifar_ds = cifar_ds.map(operations=type_cast_op, input_columns="label", num_parallel_workers=num_parallel_workers)
    cifar_ds = cifar_ds.map(operations=c_trans, input_columns="image", num_parallel_workers=num_parallel_workers)
    # apply batch operations
    cifar_ds = cifar_ds.batch(batch_size=batch_size, drop_remainder=True)
    # apply repeat operations
    cifar_ds = cifar_ds.repeat(repeat_size)

    return cifar_ds


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
        ds_eval = create_dataset(cifar10_path, batch_size=batch_size, training=False)
        if args_opt.checkpoint_path:
            model.load_checkpoint(args_opt.checkpoint_path)
        acc = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
    else:  # as for train, users could use model.train
        ds_train = create_dataset(cifar10_path, batch_size=batch_size)
        ckpoint_cb = ModelCheckpoint(prefix="resnet_cifar10", config=CheckpointConfig(
            save_checkpoint_steps=ds_train.get_dataset_size(), keep_checkpoint_max=35))
        model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                    dataset_sink_mode=dataset_sink_mode)
