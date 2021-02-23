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

"""MobileNetV2 Tutorial
The sample can be run on GPU and Ascend 910 AI processors
"""


import argparse

from tinyms import context, Tensor
from tinyms.data import Cifar10Dataset, download_dataset
from tinyms.vision import cifar10_transform
from tinyms.model import Model, MobileNetV2
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits, CrossEntropyWithLabelSmooth
from tests.st.train.loss_manager import FixedLossScaleManager
from tests.st.train.lr_generator import mobilenetv2_lr
from tests.st.train.cb_config import mobilenetv2_cb


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=1, training=True):
    """create Cifar10 dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define cifar_10 dataset and apply the transform func
    cifar10_ds = Cifar10Dataset(data_path,
                                num_parallel_workers=num_parallel_workers,
                                shuffle=True)
    cifar10_ds = cifar10_transform.apply_ds(cifar10_ds,
                                            repeat_size=repeat_size,
                                            batch_size=batch_size,
                                            training=training)

    return cifar10_ds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MobileNetV2 Image classification')
    parser.add_argument('--device_target', type=str, default="GPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Cifar10 dataset path.')
    parser.add_argument('--num_classes', type=int, default=10, help='Num classes.')
    parser.add_argument('--label_smooth', type=int, default=0.1, help='label smooth')
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument('--epoch_size', type=int, default=100, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=150, help='Batch size.')
    parser.add_argument('--is_saving_checkpoint', type=bool, default=True, help='Whether to save checkpoint.')
    parser.add_argument('--save_checkpoint_epochs', type=int, default=1,
                        help='Specify epochs interval to save each checkpoints.')
    parser.add_argument('--checkpoint_path', type=str, default="", help='Checkpoint file path.')
    args_opt = parser.parse_args()

    # Declare common variables and assign the args_opt value to them
    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    cifar10_path = args_opt.dataset_path

    # set runtime environment
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    dataset_sink_mode = not args_opt.device_target == "CPU"

    # download cifar10 dataset
    if not args_opt.dataset_path:
        args_opt.dataset_path = download_dataset('cifar10')

    # build the network
    net = MobileNetV2(args_opt.num_classes)
    model = Model(net)

    # create cifar10 dataset for training
    ds_train = create_dataset(cifar10_path, batch_size=batch_size)

    # define the loss function
    if args_opt.label_smooth > 0:
        loss = CrossEntropyWithLabelSmooth(smooth_factor=args_opt.label_smooth,
                                           num_classes=args_opt.num_classes)
    else:
        loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')

    # get learning rate
    step_size = ds_train.get_dataset_size()
    lr = Tensor(mobilenetv2_lr(global_step=0, lr_init=.0, lr_end=.0, lr_max=0.8, warmup_epochs=0,
                               total_epochs=epoch_size,
                               steps_per_epoch=step_size))

    # define the optimizer
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, 0.9, 4e-5, 1024)
    model.compile(loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()}, loss_scale_manager=loss_scale)

    if args_opt.do_eval:  # as for evaluation, users could use model.eval
        # create cifar10 dataset for eval
        ds_eval = create_dataset(cifar10_path, batch_size=batch_size, training=False)
        if args_opt.checkpoint_path:
            model.load_checkpoint(args_opt.checkpoint_path)
        acc = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
    else:  # as for train, users could use model.train
        # configure checkpoint to save weights and do training job
        save_checkpoint_epochs = args_opt.save_checkpoint_epochs
        ckpoint_cb = mobilenetv2_cb(device_target=args_opt.device_target,
                                    lr=lr,
                                    is_saving_checkpoint=args_opt.is_saving_checkpoint,
                                    save_checkpoint_epochs=args_opt.save_checkpoint_epochs,
                                    step_size=step_size)
        model.train(epoch_size, ds_train, callbacks=ckpoint_cb, dataset_sink_mode=dataset_sink_mode)
