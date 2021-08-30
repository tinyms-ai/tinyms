import argparse
import json
import os
import random
import sys

import numpy as np
from munch import Munch
from tinyms import context

from utils.file import save_json, prepare_dirs
from utils.misc import get_datetime, str2bool, get_commit_hash


def setup_cfg(args):
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if args.graph_mode == "static":
        graph_mode = context.PYNATIVE_MODE
    else:
        graph_mode = context.GRAPH_MODE

    context.set_context(mode=graph_mode, device_target=args.device)

    args.log_dir = os.path.join(args.exp_dir, args.exp_id, "logs")
    args.sample_dir = os.path.join(args.exp_dir, args.exp_id, "samples")
    args.model_dir = os.path.join(args.exp_dir, args.exp_id, "models")
    args.eval_dir = os.path.join(args.exp_dir, args.exp_id, "eval")
    prepare_dirs([args.log_dir, args.sample_dir, args.model_dir, args.eval_dir])
    args.record_file = os.path.join(args.exp_dir, args.exp_id, "records.txt")
    args.loss_file = os.path.join(args.exp_dir, args.exp_id, "losses.csv")


def validate_cfg(args):
    pass


def load_cfg():
    # There are two ways to load config, use a json file or command line arguments.
    if len(sys.argv) >= 2 and sys.argv[1].endswith('.json'):
        with open(sys.argv[1], 'r') as f:
            cfg = json.load(f)
            cfg = Munch(cfg)
            if len(sys.argv) >= 3:
                cfg.exp_id = sys.argv[2]
            else:
                print("Warning: using existing experiment dir.")
            if not cfg.about:
                cfg.about = f"Copied from: {sys.argv[1]}"
    else:
        cfg = parse_args()
        cfg = Munch(cfg.__dict__)
        if not cfg.hash:
            cfg.hash = get_commit_hash()
    current_hash = get_commit_hash()
    if current_hash != cfg.hash:
        print(f"Warning: unmatched git commit hash: `{current_hash}` & `{cfg.hash}`.")
    return cfg


def save_cfg(cfg):
    exp_path = os.path.join(cfg.exp_dir, cfg.exp_id)
    os.makedirs(exp_path, exist_ok=True)
    filename = cfg.mode
    if cfg.mode == 'train' and cfg.start_epoch != 0:
        filename = f"resume_{cfg.start_epoch}"
    save_json(exp_path, cfg, filename)


def print_cfg(cfg):
    print(json.dumps(cfg, indent=4))


def parse_args():
    parser = argparse.ArgumentParser()

    # About this experiment.
    parser.add_argument('--about', type=str, default="")
    parser.add_argument('--hash', type=str, required=False, help="Git commit hash for this experiment.")
    parser.add_argument('--exp_id', type=str, default=get_datetime(), help='Folder name and id for this experiment.')
    parser.add_argument('--exp_dir', type=str, default='expr')

    # Meta arguments.
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'sample'])
    parser.add_argument('--device', type=str, default='GPU', choices=['GPU', 'CPU', 'Ascend'])
    parser.add_argument('--graph_mode', type=str, default='dynamic', choices=['dynamic', 'static'])

    # Model related arguments.
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--g_arch', type=str, default='resnet', choices=["resnet", "unet"])

    # Dataset related arguments.
    parser.add_argument('--dataset', type=str, required=False)
    parser.add_argument('--dataset_path', type=str, required=True)

    # Training related arguments
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--end_epoch', type=int, default=200, help='epoch size for training, default is 200.')
    parser.add_argument('--initial_epoch', type=int, default=100,
                        help='number of epochs with the initial learning rate, default is 100')
    parser.add_argument('--parameter_init', type=str, default='default', choices=['he', 'default'])

    # Optimizing related arguments.
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate for generator.")
    parser.add_argument('--d_lr', type=float, default=1e-4, help="Learning rate for discriminator.")
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ema_beta', type=float, default=0.999)

    # Semantic loss related arguments.
    parser.add_argument('--lambda_sem', type=float, default=0)
    parser.add_argument('--sem_g_arch', type=str, default='resnet', choices=['resnet', 'unet', 'fpn'])

    # Loss hyper arguments.
    parser.add_argument('--lambda_adv', type=float, default=1)

    # Step related arguments.
    parser.add_argument('--log_every', type=int, default=10)
    parser.add_argument('--sample_every', type=int, default=1000)
    parser.add_argument('--save_every', type=int, default=5000)
    parser.add_argument('--eval_every', type=int, default=5000)

    # Log related arguments.
    parser.add_argument('--use_tensorboard', type=str2bool, default=False)
    parser.add_argument('--save_loss', type=str2bool, default=True)

    # Others
    parser.add_argument('--seed', type=int, default=0, help='Seed for random number generator.')
    parser.add_argument('--keep_all_models', type=str2bool, default=True)

    return parser.parse_args()
