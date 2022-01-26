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
"""DeepFM Tutorial
The sample can be run on CPU, GPU. Not test on Ascend AI processor.
For download speed, you can download the kaggle display advertising dataset by tool
and put it into `${dataset_path}/kaggle_display_advertising`, and uncompress it.
"""
import os
import argparse

from tinyms import context
from tinyms.data import download_dataset, KaggleDisplayAdvertisingDataset
from tinyms.model import Model, DeepFM, DeepFMEvalModel, DeepFMWithLoss, DeepFMTrainModel
from tinyms.callbacks import CheckpointConfig, LossTimeMonitorV2, ModelCheckpoint
from tinyms.metrics import AUCMetric


def parse_args():
    parser = argparse.ArgumentParser(description='TinyMS DeepFM Example')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented. (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='DeepFM dataset path.')
    parser.add_argument('--epoch_size', type=int, default=5, help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=16000, help='Batch size.')
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='Checkpoint directory to save checkpoint file.')
    args_opt = parser.parse_args()

    return args_opt


def create_dataset(data_path, batch_size=16000):
    """ create DeepFM dataset for train and eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
    """
    # define dataset and apply the transform func
    kaggle_display_advertising_ds = KaggleDisplayAdvertisingDataset(data_path)
    kaggle_display_advertising_ds.stats_data()
    kaggle_display_advertising_ds.convert_to_mindrecord()
    train_ds = kaggle_display_advertising_ds.load_mindreocrd_dataset(usage='train', batch_size=batch_size)
    eval_ds = kaggle_display_advertising_ds.load_mindreocrd_dataset(usage='test', batch_size=batch_size)

    return train_ds, eval_ds


if __name__ == "__main__":
    args_opt = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # download kaggle display advertising dataset
    if not args_opt.dataset_path:
        args_opt.dataset_path = download_dataset('kaggle_display_advertising')
    else:
        args_opt.dataset_path = os.path.join(args_opt.dataset_path, "kaggle_display_advertising")

    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    dataset_path = args_opt.dataset_path
    dataset_sink_mode = not args_opt.device_target == "CPU"
    convert_dtype = not args_opt.device_target == "CPU"
    checkpoint_dir = args_opt.checkpoint_dir if args_opt.checkpoint_dir is not None else "."

    # create train and eval dataset
    train_ds, eval_ds = create_dataset(data_path=dataset_path, batch_size=batch_size)
    # build base network
    data_size = train_ds.get_dataset_size()
    net = DeepFM(field_size=39, vocab_size=184965, embed_size=80, convert_dtype=convert_dtype)
    # build train network
    train_net = DeepFMTrainModel(DeepFMWithLoss(net))
    # build eval network
    eval_net = DeepFMEvalModel(net)
    # build model
    model = Model(train_net)
    # loss/ckpt/metric callbacks
    loss_tm = LossTimeMonitorV2()
    config_ckpt = CheckpointConfig(save_checkpoint_steps=data_size // 100, keep_checkpoint_max=10)
    model_ckpt = ModelCheckpoint(prefix='deepfm', directory=checkpoint_dir, config=config_ckpt)
    auc_metric = AUCMetric()

    model.compile(eval_network=eval_net, metrics={"auc_metric": auc_metric}, amp_level='O0')
    print("====== start train model ======", flush=True)
    model.train(
        epoch=epoch_size, train_dataset=train_ds, callbacks=[loss_tm, model_ckpt], dataset_sink_mode=dataset_sink_mode)
    print("====== start eval model ======", flush=True)
    acc = model.eval(eval_ds)
    print("====== eval acc: {} ======".format(acc), flush=True)
