# Copyright 2021 Huawei Technologies Co., Ltd
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

import math
import argparse
import tinyms as ts
from tinyms import context, layers, primitives as P
from tinyms.data import VOCDataset
from tinyms.vision import voc_transform
from tinyms.model import Model, ssd300_mobilenet_v2
from tinyms.losses import net_with_loss
from tinyms.optimizers import Momentum
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from tinyms.initializers import initializer, TruncatedNormal


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1,
                   is_training=True):
    """ create V0C2007 dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset and apply the transform func
    voc_ds = VOCDataset(data_path, task='Detection', num_parallel_workers=num_parallel_workers,
                        shuffle=True, decode=True)
    voc_ds = voc_transform.apply_ds(voc_ds,
                                    repeat_size=repeat_size,
                                    batch_size=batch_size,
                                    num_parallel_workers=num_parallel_workers,
                                    is_training=is_training)

    return voc_ds


def init_net_param(network, initialize_mode='TruncatedNormal'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            if initialize_mode == 'TruncatedNormal':
                p.set_data(initializer(TruncatedNormal(0.02), p.data.shape, p.data.dtype))
            else:
                p.set_data(initialize_mode, p.data.shape, p.data.dtype)


def get_lr(global_step, lr_init, lr_end, lr_max, warmup_epochs, total_epochs, steps_per_epoch):
    """
    generate learning rate array

    Args:
       global_step(int): total steps of the training
       lr_init(float): init learning rate
       lr_end(float): end learning rate
       lr_max(float): max learning rate
       warmup_epochs(float): number of warmup epochs
       total_epochs(int): total epoch of training
       steps_per_epoch(int): steps of one epoch

    Returns:
       Tensor, learning rate array
    """
    lr_each_step = []
    total_steps = steps_per_epoch * total_epochs
    warmup_steps = steps_per_epoch * warmup_epochs
    for i in range(total_steps):
        if i < warmup_steps:
            lr = lr_init + (lr_max - lr_init) * i / warmup_steps
        else:
            lr = lr_end + \
                (lr_max - lr_end) * \
                (1. + math.cos(math.pi * (i - warmup_steps) / (total_steps - warmup_steps))) / 2.
        if lr < 0.0:
            lr = 0.0
        lr_each_step.append(lr)

    current_step = global_step
    lr_each_step = ts.array(lr_each_step, dtype=ts.float32)
    learning_rate = lr_each_step[current_step:]

    return learning_rate


class TrainingWrapper(layers.Layer):
    """
    Encapsulation class of SSD network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Layer): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (float): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ts.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.hyper_map = P.HyperMap()

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        return P.depend(loss, self.optimizer(grads))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSD object detection")
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='VOC2007 dataset path.')
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate, default is 0.05.")
    parser.add_argument("--epoch_size", type=int, default=500, help="Epoch size, default is 500.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--checkpoint_path", type=str, default=None, help="Pretrained Checkpoint file path.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument('--save_checkpoint_epochs', type=int, default=10,
                        help='Specify epochs interval to save each checkpoints.')
    parser.add_argument("--loss_scale", type=int, default=1024, help="Loss scale, default is 1024.")
    args_opt = parser.parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # build the network
    net = ssd300_mobilenet_v2(class_num=21)
    # define the loss function
    net = net_with_loss(net)
    init_net_param(net)

    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    voc2007_path = args_opt.dataset_path
    ds_train = create_dataset(voc2007_path, batch_size=batch_size)
    dataset_size = ds_train.get_dataset_size()

    # define the optimizer
    lr = get_lr(global_step=args_opt.pre_trained_epoch_size * dataset_size,
                lr_init=0.001, lr_end=0.001 * args_opt.lr, lr_max=args_opt.lr,
                warmup_epochs=2, total_epochs=args_opt.epoch_size,
                steps_per_epoch=dataset_size)
    loss_scale = float(args_opt.loss_scale)
    if args_opt.device_target == "CPU":
        loss_scale = 1.0
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                   0.9, 1.5e-4, loss_scale)
    model = Model(TrainingWrapper(net, opt, loss_scale))
    model.compile()

    save_checkpoint_epochs = args_opt.save_checkpoint_epochs
    dataset_sink_mode = not args_opt.device_target == "CPU"
    if not args_opt.do_eval:  # as for train, users could use model.train
        ckpoint_cb = ModelCheckpoint(prefix="ssd300", config=CheckpointConfig(
            save_checkpoint_steps=save_checkpoint_epochs * dataset_size,
            keep_checkpoint_max=10))
        model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), TimeMonitor(data_size=dataset_size)],
                    dataset_sink_mode=dataset_sink_mode)
    else:  # as for evaluation, users could use model.eval
        ds_eval = create_dataset(voc2007_path, batch_size=batch_size, is_training=False)
        if args_opt.checkpoint_path:
            model.load_checkpoint(args_opt.checkpoint_path)
        acc = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
