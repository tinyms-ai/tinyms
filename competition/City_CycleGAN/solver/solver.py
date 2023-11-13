import datetime
import os
import time

from mindspore.nn.optim import Adam
from munch import Munch
from tinyms.utils.common_utils import GanImagePool
from tinyms.utils.train import cyclegan_lr

from data.dataset import create_dataset
from models.build import build_model
from solver.loss import CycleGANDiscriminatorLoss, CycleGANGeneratorLoss, SemanticDiscriminatorLoss, \
    SemanticGeneratorLoss
from solver.trainer import TrainOneStepG, TrainOneStepD, TrainOneStepGS, TrainOneStepDS
from utils.checkpoint import CheckpointIO
from utils.file import write_record
from utils.image import concatenate_images, save_image
from utils.misc import send_message
from utils.model import print_network


class Solver:
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dataset, args.dataset_size = create_dataset(args.dataset_path, batch_size=args.batch_size, repeat_size=1,
                                                         max_dataset_size=float("inf"), shuffle=True,
                                                         num_parallel_workers=1,
                                                         phase="train", data_dir='', use_S=args.lambda_sem != 0)
        self.nets, self.combined_G = build_model(args)

        for name, module in self.nets.items():
            print_network(module, name)

        if args.mode == 'train':
            self.ckptios = [CheckpointIO(args.model_dir + '/{:06d}_{}.ckpt', **self.nets)]
        else:
            self.ckptios = [CheckpointIO(args.model_dir + '/{:06d}_{}.ckpt', **self.nets)]

        self.nets.generator = self.combined_G

        self.use_tensorboard = args.use_tensorboard
        if self.use_tensorboard:
            from utils.logger import Logger
            self.logger = Logger(args.log_dir)

    def initialize_parameters(self):
        if self.args.parameter_init == 'default':
            # Do nothing because the parameters has been initialized in this manner.
            pass
        else:
            raise NotImplementedError

    def save_model(self, step):
        for ckptio in self.ckptios:
            ckptio.save(step)

    def load_model(self, step):
        for ckptio in self.ckptios:
            ckptio.load(step)

    def train(self):
        args = self.args
        dataset = self.dataset
        nets = self.nets
        use_semantic = args.lambda_sem != 0
        if args.start_epoch > 0:
            self.load_model(args.start_epoch)
        else:
            self.initialize_parameters()

        # Get the models.
        combined_G, D_A, D_B = self.combined_G, self.nets.D_A, self.nets.D_B

        # Define the loss functions.
        loss_G = CycleGANGeneratorLoss(args, nets)
        loss_D = CycleGANDiscriminatorLoss(D_A, D_B)

        # Setup the optimizers.
        lr = cyclegan_lr(args.end_epoch, args.initial_epoch, args.dataset_size)
        optimizer_G = Adam(combined_G.trainable_params(), lr, beta1=args.beta1, beta2=args.beta2)
        optimizer_D = Adam(loss_D.trainable_params(), lr, beta1=args.beta1, beta2=args.beta2)

        # Setup the trainers.
        train_g = TrainOneStepG(args, loss_G, combined_G, optimizer_G)
        train_d = TrainOneStepD(loss_D, optimizer_D)

        if use_semantic:
            # Models
            G_S, D_S = self.nets.G_S, self.nets.D_S
            # Losses
            loss_G_S = SemanticGeneratorLoss(G_S, D_S)
            loss_D_S = SemanticDiscriminatorLoss(D_S)
            # Optimizers
            optimizer_G_S = Adam(loss_G_S.trainable_params(), lr, beta1=args.beta1, beta2=args.beta2)
            optimizer_D_S = Adam(loss_D_S.trainable_params(), lr, beta1=args.beta1, beta2=args.beta2)
            # Trainers
            train_g_s = TrainOneStepGS(loss_G_S, loss_D_S, optimizer_G_S)
            train_d_s = TrainOneStepDS(loss_D_S, optimizer_D_S)

        image_pool_A = GanImagePool(pool_size=50)
        image_pool_B = GanImagePool(pool_size=50)

        data_loader = dataset.create_dict_iterator()

        print('Start training...')
        start_time = time.time()
        step = 0
        for epoch in range(args.start_epoch + 1, args.end_epoch + 1):
            for i, data in enumerate(data_loader):
                step += 1
                img_A = data["image_A"]
                img_B = data["image_B"]
                if use_semantic:
                    img_S = data['image_S']
                else:
                    # img_S is dummy image, won't be used
                    img_S = data['image_A']
                fake_A, fake_B, loss_G, loss_G_A, loss_G_B, loss_C_A, loss_C_B, loss_idt_A, loss_idt_B, loss_sem = \
                    train_g(img_A, img_B, img_S)
                loss_D = train_d(img_A, img_B, image_pool_A.query(fake_A), image_pool_B.query(fake_B))
                g_loss_ref = Munch(total=loss_G, adv_A=loss_G_A, adv_B=loss_G_B, rec_A=loss_C_A, rec_B=loss_C_B,
                                   idt_A=loss_idt_A, idt_B=loss_idt_B, sem=loss_sem)
                d_loss_ref = Munch(adv=loss_D)
                loss_ref_list = [d_loss_ref, g_loss_ref]
                loss_prefix_list = ['D/', 'G/']

                if use_semantic:
                    # Train G_S
                    fake_S_A, fake_S_B, loss_G_S, loss_S_A, loss_S_B = train_g_s(img_A, img_B)
                    # Train D_S
                    loss_D_S = train_d_s(img_S, fake_S_A, fake_S_B)
                    # Deal with loss items.
                    gs_loss_ref = Munch(total=loss_G_S, adv_A=loss_S_A, adv_B=loss_S_B)
                    ds_loss_ref = Munch(adv=loss_D_S)
                    # Append loss related items.
                    loss_ref_list.extend([ds_loss_ref, gs_loss_ref])
                    loss_prefix_list.extend(['D_S/', 'G_S/'])

                if step % args.log_every == 0:
                    elapsed = time.time() - start_time
                    elapsed = str(datetime.timedelta(seconds=elapsed))[:-7]
                    log = "[%s]-[%i/%i]-[%i]: " % (elapsed, epoch, args.end_epoch, step)
                    all_losses = dict()
                    for loss, prefix in zip(loss_ref_list, loss_prefix_list):
                        for key, value in loss.items():
                            all_losses[prefix + key] = float(value.asnumpy())
                    log += ' '.join(['%s: [%.4f]' % (key, value) for key, value in all_losses.items()])
                    print(log)
                    if args.save_loss:
                        if step == args.log_every:
                            header = ','.join(['iter'] + [str(loss) for loss in all_losses.keys()])
                            write_record(header, args.loss_file, False)
                        log = ','.join([str(step)] + [str(loss) for loss in all_losses.values()])
                        write_record(log, args.loss_file, False)
                    if self.use_tensorboard:
                        for tag, value in all_losses.items():
                            self.logger.scalar_summary(tag, value, step)

                if step % args.sample_every == 0:
                    if use_semantic:
                        sample = concatenate_images([img_A, img_B, fake_A, fake_B, fake_S_A, fake_S_B])
                    else:
                        sample = concatenate_images([img_A, img_B, fake_A, fake_B])
                    save_image(sample, os.path.join(args.sample_dir, f"sample_{step}.jpg"))
            self.save_model(epoch)
            # send_message(f"Epoch {epoch} is done.")

        send_message("Model training completed.")
