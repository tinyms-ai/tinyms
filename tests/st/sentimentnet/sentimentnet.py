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
"""SentimentNet Tutorial
The sample can be run on CPU, GPU and Ascend 910 AI processor.
"""
import random
import argparse
import numpy as np
import os


from tinyms import Tensor
from tinyms import context
from tinyms.model import Model, SentimentNet
from tinyms.data import ImdbDataset
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits
from tinyms.data import MindDataset




random.seed(1)


def parse_args():

    parser = argparse.ArgumentParser(description='TinyMs LSTM Example')
    parser.add_argument('--preprocess', type=str, default='false', choices=['true', 'false'],
                        help='whether to preprocess data.')
    parser.add_argument('--aclimdb_path', type=str, default="./aclImdb",
                        help='path where the dataset is stored.')
    parser.add_argument('--glove_path', type=str, default="./glove",
                        help='path where the GloVe is stored.')
    parser.add_argument('--preprocess_path', type=str, default="./preprocess",
                        help='path where the pre-process data is stored.')
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--do_eval', type=bool, default=False,
                        help='Do eval or not.')
    parser.add_argument('--epoch_size', type=int, default=20,
                        help='Epoch size.')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size.')
    parser.add_argument('--num_classes', type=int, default=2,
                        help='Num classes.')
    parser.add_argument('--embed_size', type=int, default=300,
                        help='Embed_size.')
    parser.add_argument('--num_hiddens', type=int, default=100,
                        help='Num_hidden.')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Num_layers.')
    parser.add_argument('--bidirectional', type=bool, default=True,
                        help='bidirectional or not.')
    parser.add_argument('--save_checkpoint_epochs', type=int, default=5,
                        help='Specify epochs interval to save each checkpoints.')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint file path.')

    args_opt = parser.parse_args()

    return args_opt


def lstm_create_dataset(data_path, batch_size=32, repeat_size=1,
                   num_parallel_workers=10):
    """ create aclimdb dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """

    data_set = MindDataset(data_path, columns_list=["feature", "label"], num_parallel_workers=num_parallel_workers)

    # apply map operations on aclimdb
    data_set = data_set.shuffle(buffer_size=data_set.get_dataset_size())
    data_set = data_set.batch(batch_size=batch_size, drop_remainder=True)
    data_set = data_set.repeat(count=repeat_size)

    return data_set





if __name__ == '__main__':
    args_opt = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)
    if args_opt.preprocess == "true":
        train_data_path = os.path.join(args_opt.preprocess_path, "aclImdb_train.mindrecord")
        val_data_path = os.path.join(args_opt.preprocess_path, "aclImdb_test.mindrecord")
        if os.path.exists(train_data_path) and os.path.exists(val_data_path):
            pass
        else:
            os.removedirs(args_opt.preprocess_path)
            print("============== Starting Data Pre-processing ==============")
            imdbdata = ImdbDataset(args_opt.aclimdb_path, args_opt.glove_path, args_opt.embed_size)
            imdbdata.convert_to_mindrecord(
                args_opt.preprocess_path
            )

    embedding_table = np.loadtxt(os.path.join(args_opt.preprocess_path, "weight.txt")).astype(np.float32)

    # build the network
    net = SentimentNet(
        vocab_size=embedding_table.shape[0],
        embed_size=args_opt.embed_size,
        num_hiddens=args_opt.num_hiddens,
        num_layers=args_opt.num_layers,
        bidirectional=args_opt.bidirectional,
        num_classes=args_opt.num_classes,
        weight=Tensor(embedding_table),
        batch_size=args_opt.batch_size
    )
    net.update_parameters_name(prefix='huawei')
    model = Model(net)

    # define the loss function
    net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")
    # define the optimizer
    net_opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
    model.compile(loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": Accuracy()})

    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    save_checkpoint_epochs = args_opt.save_checkpoint_epochs
    dataset_sink_mode = not args_opt.device_target == "CPU"
    if args_opt.do_eval:
        # as for evaluation, users could use model.eval
        ds_eval = lstm_create_dataset(os.path.join(args_opt.preprocess_path, "aclImdb_test.mindrecord"), args_opt.batch_size)
        if args_opt.checkpoint_path:
            model.load_checkpoint(args_opt.checkpoint_path)
        acc = model.eval(ds_eval, dataset_sink_mode=dataset_sink_mode)
        print("============== Accuracy:{} ==============".format(acc))
    else:
        # as for train, users could use model.train
        ds_train = lstm_create_dataset(os.path.join(args_opt.preprocess_path, "aclImdb_train.mindrecord"), args_opt.batch_size)
        ckpoint_cb = ModelCheckpoint(prefix="SentimentNet_imdb", config=CheckpointConfig(
            save_checkpoint_steps=save_checkpoint_epochs * ds_train.get_dataset_size(),
            keep_checkpoint_max=10))
        model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor()],
                    dataset_sink_mode=dataset_sink_mode)