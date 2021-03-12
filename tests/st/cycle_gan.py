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

import tinyms as ts
from tinyms import context, Tensor
from tinyms.data import GeneratorDataset, UnalignedDataset, GanImageFolderDataset, DistributedSampler
from tinyms.vision import cyclegan_transform
from tinyms.model.cycle_gan.cycle_gan import get_generator_discriminator, cycle_gan, TrainOneStepG, TrainOneStepD
from tinyms.utils.utils import gan_load_ckpt, ImagePool
from tinyms.losses import DiscriminatorLoss, GeneratorLoss
from tinyms.optimizers import Adam
from tinyms.utils.train.lr_generator import cyclegan_lr
from tinyms.utils.gan_reporter import GanReporter
from tinyms.data.utils import save_image, generate_image_list
from tinyms.utils.eval import CityScapes, fast_hist, get_scores


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
    dataset_size = len(ds)
    return gan_generator_ds, dataset_size


def train_process(reporter, data_loader, net_G, net_D, imgae_pool_A, imgae_pool_B):
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


def predict_process(reporter, data_loader, G_generator, predict_name='A_to_B', fake_name='fake_B'):
    reporter.start_predict(predict_name)
    for data in data_loader:
        img = Tensor(data["image"])
        path = str(data["image_name"][0], encoding="utf-8")
        fake = G_generator(img)
        save_image(fake, os.path.join(imgs_out, fake_name, path))
    reporter.info('save %s at %s', fake_name, os.path.join(imgs_out, fake_name, path))
    reporter.end_predict()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='MindSpore Cycle GAN Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='cityscape dataset path.')
    parser.add_argument('--phase', type=str, default="train", help='train, eval or predict.')
    parser.add_argument('--model', type=str, default="resnet", choices=("resnet", "unet"),
                        help='generator model, should be in [resnet, unet].')
    parser.add_argument('--max_epoch_size', type=int, default=2, help='epoch size for training, default is 200.')
    parser.add_argument('--epoch_size', type=int, default=1, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')
    parser.add_argument("--save_checkpoint_epochs", type=int, default=1, help="Save checkpoint epochs, default is 10.")
    parser.add_argument("--G_A_ckpt", type=str, default=None, help="pretrained checkpoint file path of G_A.")
    parser.add_argument("--G_B_ckpt", type=str, default=None, help="pretrained checkpoint file path of G_B.")
    parser.add_argument("--D_A_ckpt", type=str, default=None, help="pretrained checkpoint file path of D_A.")
    parser.add_argument("--D_B_ckpt", type=str, default=None, help="pretrained checkpoint file path of D_B.")
    parser.add_argument('--outputs_dir', type=str, default='./outputs',
                        help='models are saved here, default is ./outputs.')
    parser.add_argument('--save_imgs', type=bool, default=True,
                        help='whether save imgs when epoch end, default is True.')
    parser.add_argument("--cityscapes_dir", type=str, help="Path to the original cityscapes dataset")
    parser.add_argument("--result_dir", type=str, help="Path to the generated images to be evaluated")
    args_opt = parser.parse_args()

    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    dataset_path = args_opt.dataset_path
    phase = args_opt.phase

    if phase == "predict" and (args_opt.G_A_ckpt is None or args_opt.G_B_ckpt is None):
        raise ValueError('Must set G_A_ckpt and G_B_ckpt in predict phase!')

    if dataset_path is None and (phase in ["train", "predict"]):
        raise ValueError('Must set dataset_path!')

    if phase == "eval" and (args_opt.cityscapes_dir is None or args_opt.result_dir is None):
        raise ValueError('Must set cityscapes_dir and result_dir in eval phase!')

    model = args_opt.model

    batch_size = args_opt.batch_size
    max_epoch_size = args_opt.max_epoch_size
    epoch_size = args_opt.epoch_size
    max_dataset_size = float("inf")
    outputs_dir = args_opt.outputs_dir

    if phase == "train":
        # create dataset
        dataset, args_opt.dataset_size = create_dataset(dataset_path, batch_size=batch_size, repeat_size=1,
                                                        max_dataset_size=max_dataset_size, shuffle=True,
                                                        num_parallel_workers=1,
                                                        phase=phase, data_dir=None)
        # build cycle gan generator
        G_A, G_B, D_A, D_B = get_generator_discriminator(model)
        gan_load_ckpt(args_opt, G_A, G_B, D_A, D_B)
        generator_net = cycle_gan(G_A, G_B)

        # define loss function and optimizer
        loss_D = DiscriminatorLoss(D_A, D_B)
        loss_G = GeneratorLoss(generator_net, D_A, D_B)
        lr = cyclegan_lr(max_epoch_size, epoch_size, args_opt.dataset_size)

        optimizer_G = Adam(generator_net.trainable_params(),
                           cyclegan_lr(max_epoch_size, epoch_size, args_opt.dataset_size), beta1=0.5)
        optimizer_D = Adam(loss_D.trainable_params(),
                           cyclegan_lr(max_epoch_size, epoch_size, args_opt.dataset_size), beta1=0.5)

        # build two net: generator net and descrinator net
        net_G = TrainOneStepG(loss_G, generator_net, optimizer_G)
        net_D = TrainOneStepD(loss_D, optimizer_D)

        # train process
        imgae_pool_A = ImagePool(pool_size=50)
        imgae_pool_B = ImagePool(pool_size=50)

        data_loader = dataset.create_dict_iterator()
        reporter = GanReporter(args_opt)
        train_process(reporter, data_loader, net_G, net_D, imgae_pool_A, imgae_pool_B)

    elif phase == 'predict':
        # build cycle gan generator
        G_A, G_B, _, _ = get_generator_discriminator(model)
        G_A.set_train(True)
        G_B.set_train(True)
        gan_load_ckpt(args_opt, G_A, G_B)

        imgs_out = os.path.join(outputs_dir, "predict")
        if not os.path.exists(imgs_out):
            os.makedirs(imgs_out)
        if not os.path.exists(os.path.join(imgs_out, "fake_A")):
            os.makedirs(os.path.join(imgs_out, "fake_A"))
        if not os.path.exists(os.path.join(imgs_out, "fake_B")):
            os.makedirs(os.path.join(imgs_out, "fake_B"))

        # create test dataset A
        dataset, args_opt.dataset_size = create_dataset(dataset_path, batch_size=batch_size, repeat_size=1,
                                                        max_dataset_size=max_dataset_size, shuffle=True,
                                                        num_parallel_workers=1, phase=phase,
                                                        data_dir='testA')
        # predict first time, G_A to testA dataset, then generate fake image into fake_B dir
        data_loader = dataset.create_dict_iterator(output_numpy=True)
        reporter = GanReporter(args_opt)
        predict_process(reporter, data_loader, G_generator=G_A, predict_name='A_to_B', fake_name='fake_B')

        # create test dataset B
        dataset, args_opt.dataset_size = create_dataset(dataset_path, batch_size=batch_size, repeat_size=1,
                                                        max_dataset_size=max_dataset_size, shuffle=True,
                                                        num_parallel_workers=1, phase=phase,
                                                        data_dir='testB')

        # predict second time, G_B to testB dataset, then generate fake image into fake_A dir
        data_loader = dataset.create_dict_iterator(output_numpy=True)
        predict_process(reporter, data_loader, G_generator=G_B, predict_name='B_to_A', fake_name='fake_A')
    else:
        cityscapes_dir = args_opt.cityscapes_dir
        result_dir = args_opt.result_dir
        CS = CityScapes()
        cityscapes = generate_image_list(cityscapes_dir)
        hist_perframe = ts.zeros((CS.class_num, CS.class_num)).asnumpy()
        for i, img_path in enumerate(cityscapes):
            if i % 100 == 0:
                print('Evaluating: %d/%d' % (i, len(cityscapes)))
            img_name = os.path.split(img_path)[1]
            ids1 = CS.get_id(os.path.join(cityscapes_dir, img_name))
            ids2 = CS.get_id(os.path.join(result_dir, img_name))
            hist_perframe += fast_hist(ids1.flatten(), ids2.flatten(), CS.class_num)

        mean_pixel_acc, mean_class_acc, mean_class_iou, per_class_acc, per_class_iou = get_scores(hist_perframe)
        print(f"mean_pixel_acc: {mean_pixel_acc}, mean_class_acc: {mean_class_acc}, mean_class_iou: {mean_class_iou}")
        with open(os.path.join(outputs_dir, 'evaluation_results.txt'), 'w') as f:
            f.write('Mean pixel accuracy: %f\n' % mean_pixel_acc)
            f.write('Mean class accuracy: %f\n' % mean_class_acc)
            f.write('Mean class IoU: %f\n' % mean_class_iou)
            f.write('************ Per class numbers below ************\n')
            for i, cl in enumerate(CS.classes):
                while len(cl) < 15:
                    cl = cl + ' '
                f.write('%s: acc = %f, iou = %f\n' % (cl, per_class_acc[i], per_class_iou[i]))

