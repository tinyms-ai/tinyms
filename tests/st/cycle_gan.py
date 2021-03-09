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
"""Cycle GAN Tutorial
The sample can be run on CPU, GPU and Ascend 910 AI processor.
"""
import os
import argparse

from tinyms import context
from tinyms.data import GeneratorDataset, UnalignedDataset, GanImageFolderDataset, DistributedSampler
from tinyms.vision import cyclegan_transform
from tinyms.model.cycle_gan.cycle_gan import get_generator_discriminator, cycle_gan, TrainOneStepG, TrainOneStepD
from tinyms.utils.utils import gan_load_ckpt, ImagePool
from tinyms.losses import DiscriminatorLoss, GeneratorLoss
from tinyms.optimizers import Adam
from tinyms.utils.train.lr_generator import cyclegan_lr
from tinyms.utils.gan_reporter import GanReporter


def create_dataset(dataset_path, batch_size=1, repeat_size=1, max_dataset_size=None,
                   shuffle=True, num_parallel_workers=1, phase='train', data_dir='testA'):
    """ create Mnist dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset and apply the transform func
    if phase == 'train':
        ds = UnalignedDataset(dataset_path, phase, max_dataset_size=max_dataset_size, shuffle=True)

        device_num = 1
        distributed_sampler = DistributedSampler(len(ds), num_replicas=device_num, rank=0, shuffle=shuffle)
        gan_generator_ds = GeneratorDataset(ds, column_names=["image_A", "image_B"], sampler=distributed_sampler,
                                            num_parallel_workers=num_parallel_workers)
    else:
        datadir = os.path.join(dataset_path, data_dir)
        ds = GanImageFolderDataset(datadir, max_dataset_size=max_dataset_size)
        gan_generator_ds = GeneratorDataset(ds, column_names=["image", "image_name"],
                                            num_parallel_workers=num_parallel_workers)

    gan_generator_ds = cyclegan_transform.apply_ds(gan_generator_ds,
                                                   repeat_size=repeat_size,
                                                   batch_size=batch_size,
                                                   num_parallel_workers=num_parallel_workers,
                                                   shuffle=shuffle,
                                                   phase=phase)
    data_size = len(ds)
    return gan_generator_ds, data_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Cycle GAN Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='cityscape dataset path.')
    parser.add_argument('--phase', type=str, default="train", help='train, eval or predict.')
    parser.add_argument('--model', type=str, default="resnet", choices=("resnet", "unet"),
                        help='generator model, should be in [resnet, unet].')
    parser.add_argument('--max_epoch_size', type=int, default=200, help='epoch size for training, default is 200.')
    parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='CheckPoint file path.')
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--G_A_ckpt", type=str, default=None, help="pretrained checkpoint file path of G_A.")
    parser.add_argument("--G_B_ckpt", type=str, default=None, help="pretrained checkpoint file path of G_B.")
    parser.add_argument("--D_A_ckpt", type=str, default=None, help="pretrained checkpoint file path of D_A.")
    parser.add_argument("--D_B_ckpt", type=str, default=None, help="pretrained checkpoint file path of D_B.")
    parser.add_argument('--outputs_dir', type=str, default='./outputs',
                        help='models are saved here, default is ./outputs.')
    parser.add_argument('--save_imgs', type=bool, default=True,
                        help='whether save imgs when epoch end, default is True.')
    args_opt = parser.parse_args()

    context.set_context(mode=context.PYNATIVE_MODE, device_target=args_opt.device_target)

    dataset_path = args_opt.dataset_path
    batch_size = args_opt.batch_size
    phase = args_opt.phase
    model = args_opt.model
    max_epoch_size = args_opt.max_epoch_size
    epoch_size = args_opt.epoch_size

    if phase != "train" and (args_opt.G_A_ckpt is None or args_opt.G_B_ckpt is None):
        raise ValueError('Must set G_A_ckpt and G_B_ckpt in predict phase!')

    if dataset_path is None and (phase in ["train", "predict"]):
        raise ValueError('Must set dataset_path!')

    max_dataset_size = float("inf")

    dataset, dataset_size = create_dataset(dataset_path, batch_size=batch_size, repeat_size=1,
                                           max_dataset_size=max_dataset_size, shuffle=True,
                                           num_parallel_workers=1,
                                           phase=phase,
                                           data_dir=None)
    G_A, G_B, D_A, D_B = get_generator_discriminator(model)

    gan_load_ckpt(args_opt, G_A, G_B, D_A, D_B)
    generator_net = cycle_gan(G_A, G_B)

    loss_D = DiscriminatorLoss(D_A, D_B)
    loss_G = GeneratorLoss(generator_net, D_A, D_B)
    lr = cyclegan_lr(max_epoch_size, epoch_size, dataset_size)

    optimizer_G = Adam(generator_net.trainable_params(),
                       cyclegan_lr(max_epoch_size, epoch_size, dataset_size), beta1=0.5)
    optimizer_D = Adam(loss_D.trainable_params(),
                       cyclegan_lr(max_epoch_size, epoch_size, dataset_size), beta1=0.5)

    net_G = TrainOneStepG(loss_G, generator_net, optimizer_G)
    net_D = TrainOneStepD(loss_D, optimizer_D)

    imgae_pool_A = ImagePool(pool_size=50)
    imgae_pool_B = ImagePool(pool_size=50)

    data_loader = dataset.create_dict_iterator()
    reporter = GanReporter(args_opt, dataset_size)
    reporter.info('==========start training===============')
    for _ in range(max_epoch_size):
        reporter.epoch_start()
        for data in data_loader:
            img_A = data["image_A"]
            img_B = data["image_B"]
            res_G = net_G(img_A, img_B)
            fake_A = res_G[0]
            fake_B = res_G[1]
            res_D = net_D(img_A, img_B, imgae_pool_A.query(fake_A), imgae_pool_B.query(fake_B))
            reporter.step_end(res_G, res_D)
            reporter.visualizer(img_A, img_B, fake_A, fake_B)
        reporter.epoch_end(net_G)

    reporter.info('==========end training===============')