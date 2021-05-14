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
The sample can be run on CPU, GPU and Ascend 910 AI processors
"""
import argparse

from tinyms import context
from tinyms.data import Cifar10Dataset, download_dataset
from tinyms.vision import cifar10_transform
from tinyms.model import Model, mobilenetv2
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import CrossEntropyWithLabelSmooth
from tinyms.utils.train.loss_manager import FixedLossScaleManager
from tinyms.utils.train.lr_generator import mobilenetv2_lr
from tinyms.utils.train.cb_config import mobilenetv2_cb


def parse_args():
    parser = argparse.ArgumentParser(description='MobileNetV2 Image classification')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented. (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='Cifar10 dataset path.')
    parser.add_argument('--num_classes', type=int, default=10, help='Num classes. (default: 10)')
    parser.add_argument('--label_smooth', type=float, default=0.1, help='label smooth. (default: 0.1)')
    parser.add_argument('--epoch_size', type=int, default=60, help='Epoch size. (default: 60)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size. (default: 32)')
    parser.add_argument('--is_saving_checkpoint', type=bool, default=True, help='Whether to save checkpoint.')
    parser.add_argument('--save_checkpoint_epochs', type=int, default=10,
                        help='Specify epochs interval to save each checkpoints. (default: 10)')
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not.')
    parser.add_argument('--load_pretrained', type=str, choices=['hub', 'local'], default='local',
                        help='Specify where to load pretrained model, only valid in do_eval mode. (default: local)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint file path. Only valid when load_pretrained is `local`.')
    parser.add_argument('--hub_uid', type=str, default=None,
                        help='Model asset uid. Only valid when load_pretrained is `hub`.')
    args_opt = parser.parse_args()

    return args_opt


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=4,
                   is_training=True):
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
                                            is_training=is_training)

    return cifar10_ds


if __name__ == '__main__':
    args_opt = parse_args()

    # download cifar10 dataset
    if not args_opt.dataset_path:
        args_opt.dataset_path = download_dataset('cifar10')

    # Declare common variables and assign the args_opt value to them
    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    cifar10_path = args_opt.dataset_path

    # set runtime environment
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    dataset_sink_mode = not args_opt.device_target == "CPU"

    # create cifar10 dataset for training
    ds_train = create_dataset(cifar10_path, batch_size=batch_size)
    step_size = ds_train.get_dataset_size()

    # build the network
    if args_opt.do_eval and args_opt.load_pretrained == 'hub':
        from tinyms import hub
        net = hub.load(args_opt.hub_uid, class_num=args_opt.num_classes, is_training=not args_opt.do_eval)
    else:
        net = mobilenetv2(class_num=args_opt.num_classes, is_training=not args_opt.do_eval)
    model = Model(net)
    # define the loss function
    loss = CrossEntropyWithLabelSmooth(smooth_factor=args_opt.label_smooth,
                                       num_classes=args_opt.num_classes)
    # get learning rate
    lr_max = 0.001
    lr_init_scale = 0.01
    lr_end_scale = 0.01
    lr = mobilenetv2_lr(global_step=0,
                        lr_init=lr_max*lr_init_scale,
                        lr_end=lr_max*lr_end_scale,
                        lr_max=lr_max,
                        warmup_epochs=2,
                        total_epochs=epoch_size,
                        steps_per_epoch=step_size)
    # define the optimizer
    loss_scale = FixedLossScaleManager(1024, drop_overflow_update=False)
    opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()),
                   lr, 0.9, 4e-5, 1024)
    model.compile(loss_fn=loss, optimizer=opt, metrics={"Accuracy": Accuracy()},
                  loss_scale_manager=loss_scale)

    if not args_opt.do_eval:  # as for train, users could use model.train
        # configure checkpoint to save weights and do training job
        save_checkpoint_epochs = args_opt.save_checkpoint_epochs
        ckpoint_cb = mobilenetv2_cb(device_target=args_opt.device_target,
                                    lr=lr,
                                    is_saving_checkpoint=args_opt.is_saving_checkpoint,
                                    save_checkpoint_epochs=args_opt.save_checkpoint_epochs,
                                    step_size=step_size)
        model.train(epoch_size, ds_train, callbacks=ckpoint_cb, dataset_sink_mode=dataset_sink_mode)
    else:  # as for evaluation, users could use model.eval
        # create cifar10 dataset for eval
        ds_eval = create_dataset(cifar10_path, batch_size=batch_size, is_training=False)
        if args_opt.load_pretrained == 'local':
            if args_opt.checkpoint_path:
                model.load_checkpoint(args_opt.checkpoint_path)
        acc = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
