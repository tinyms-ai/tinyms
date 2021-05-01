# Copyright 2020 Huawei Technologies Co., Ltd
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
'''
Bert finetune and evaluation script.
'''

import os
import argparse
import logging

from finetune_eval_config import optimizer_cfg, bert_net_cfg
from assessment_method import Accuracy, F1, MCC, SpearmanCorrelation

import tinyms as ts
from tinyms import context
from tinyms import vision
from tinyms.model import Model
from tinyms import primitives as P
from tinyms.layers import DynamicLossScaleUpdateCell
from tinyms.data import TFRecordDataset
from tinyms.model.bert import BertFinetuneLayer, BertCLS
from tinyms.optimizers import AdamWeightDecay, Lamb, Momentum
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, TimeMonitor, BertLossCallBack
from mindspore.nn.learning_rate_schedule import LearningRateSchedule, PolynomialDecayLR, WarmUpLR


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
_cur_dir = os.getcwd()


def LoadNewestCkpt(load_finetune_checkpoint_dir, steps_per_epoch, epoch_num, prefix):
    """
    Find the ckpt finetune generated and load it into eval network.
    """

    files = os.listdir(load_finetune_checkpoint_dir)
    pre_len = len(prefix)
    max_num = 0
    for filename in files:
        name_ext = os.path.splitext(filename)
        if name_ext[-1] != ".ckpt":
            continue
        if filename.find(prefix) == 0 and not filename[pre_len].isalpha():
            index = filename[pre_len:].find("-")
            if index == 0 and max_num == 0:
                load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_dir, filename)
            elif index not in (0, -1):
                name_split = name_ext[-2].split('_')
                if (steps_per_epoch != int(name_split[len(name_split)-1])) \
                        or (epoch_num != int(filename[pre_len + index + 1:pre_len + index + 2])):
                    continue
                num = filename[pre_len + 1:pre_len + index]
                if int(num) > max_num:
                    max_num = int(num)
                    load_finetune_checkpoint_path = os.path.join(load_finetune_checkpoint_dir, filename)
    return load_finetune_checkpoint_path


class BertLearningRate(LearningRateSchedule):
    """
    Warmup-decay learning rate for Bert network.
    """

    def __init__(self, learning_rate, end_learning_rate, warmup_steps, decay_steps, power):
        super(BertLearningRate, self).__init__()
        self.warmup_flag = False
        if warmup_steps > 0:
            self.warmup_flag = True
            self.warmup_lr = WarmUpLR(learning_rate, warmup_steps)
        self.decay_lr = PolynomialDecayLR(learning_rate, end_learning_rate, decay_steps, power)
        self.warmup_steps = ts.array([warmup_steps], dtype=ts.float32)

        self.greater = P.Greater()
        self.one = ts.array([1.0], dtype=ts.float32)
        self.cast = P.Cast()

    def construct(self, global_step):
        decay_lr = self.decay_lr(global_step)
        if self.warmup_flag:
            is_warmup = self.cast(self.greater(self.warmup_steps, global_step), ts.float32)
            warmup_lr = self.warmup_lr(global_step)
            lr = (self.one - is_warmup) * decay_lr + is_warmup * warmup_lr
        else:
            lr = decay_lr
        return lr


def make_directory(path: str):
    """Make directory."""

    if path is None or not isinstance(path, str) or path.strip() == "":
        logger.error("The path(%r) is invalid type.", path)
        raise TypeError("Input path is invaild type")

    # convert the relative paths
    path = os.path.realpath(path)
    logger.debug("The abs path is %r", path)

    # check the path is exist and write permissions?
    if os.path.exists(path):
        real_path = path
    else:
        # All exceptions need to be caught because create directory maybe have some limit(permissions)
        logger.debug("The directory(%s) doesn't exist, will create it", path)
        try:
            os.makedirs(path, exist_ok=True)
            real_path = path
        except PermissionError as e:
            logger.error("No write permission on the directory(%r), error = %r", path, e)
            raise TypeError("No write permission on the directory.")
    return real_path


def create_classification_dataset(batch_size=1, repeat_count=1, assessment_method="accuracy",
                                  data_file_path=None, schema_file_path=None, do_shuffle=True):
    """create finetune or evaluation dataset"""

    type_cast_op = vision.TypeCast(ts.int32)
    ds = TFRecordDataset([data_file_path], schema_file_path if schema_file_path != "" else None,
                            columns_list=["input_ids", "input_mask", "segment_ids", "label_ids"], shuffle=do_shuffle)
    if assessment_method == "Spearmancorrelation":
        type_cast_op_float = vision.TypeCast(ts.float32)
        ds = ds.map(operations=type_cast_op_float, input_columns="label_ids")
    else:
        ds = ds.map(operations=type_cast_op, input_columns="label_ids")
    ds = ds.map(operations=type_cast_op, input_columns="segment_ids")
    ds = ds.map(operations=type_cast_op, input_columns="input_mask")
    ds = ds.map(operations=type_cast_op, input_columns="input_ids")
    ds = ds.repeat(repeat_count)
    # apply batch operations
    ds = ds.batch(batch_size, drop_remainder=True)
    return ds


def do_train(dataset=None, network=None, load_checkpoint_path="", save_checkpoint_path="", epoch_num=1):
    """ do train """

    if load_checkpoint_path == "":
        raise ValueError("Pretrain model missed, finetune task must load pretrain model!")
    steps_per_epoch = dataset.get_dataset_size()
    # optimizer
    if optimizer_cfg.optimizer == 'AdamWeightDecay':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.AdamWeightDecay.learning_rate,
                                       end_learning_rate=optimizer_cfg.AdamWeightDecay.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.AdamWeightDecay.power)
        params = network.trainable_params()
        decay_params = list(filter(optimizer_cfg.AdamWeightDecay.decay_filter, params))
        other_params = list(filter(lambda x: not optimizer_cfg.AdamWeightDecay.decay_filter(x), params))
        group_params = [{'params': decay_params, 'weight_decay': optimizer_cfg.AdamWeightDecay.weight_decay},
                        {'params': other_params, 'weight_decay': 0.0}]

        optimizer = AdamWeightDecay(group_params, lr_schedule, eps=optimizer_cfg.AdamWeightDecay.eps)
    elif optimizer_cfg.optimizer == 'Lamb':
        lr_schedule = BertLearningRate(learning_rate=optimizer_cfg.Lamb.learning_rate,
                                       end_learning_rate=optimizer_cfg.Lamb.end_learning_rate,
                                       warmup_steps=int(steps_per_epoch * epoch_num * 0.1),
                                       decay_steps=steps_per_epoch * epoch_num,
                                       power=optimizer_cfg.Lamb.power)
        optimizer = Lamb(network.trainable_params(), learning_rate=lr_schedule)
    elif optimizer_cfg.optimizer == 'Momentum':
        optimizer = Momentum(network.trainable_params(), learning_rate=optimizer_cfg.Momentum.learning_rate,
                             momentum=optimizer_cfg.Momentum.momentum)
    else:
        raise Exception("Optimizer not supported. support: [AdamWeightDecay, Lamb, Momentum]")

    # load checkpoint into network
    ckpt_config = CheckpointConfig(save_checkpoint_steps=steps_per_epoch, keep_checkpoint_max=1)
    ckpoint_cb = ModelCheckpoint(prefix="classifier",
                                 directory=None if save_checkpoint_path == "" else save_checkpoint_path,
                                 config=ckpt_config)

    update_layer = DynamicLossScaleUpdateCell(loss_scale_value=2**32, scale_factor=2, scale_window=1000)
    netwithgrads = BertFinetuneLayer(network, optimizer=optimizer, scale_update_layer=update_layer)
    model = Model(netwithgrads)
    model.load_checkpoint(load_checkpoint_path)
    callbacks = [TimeMonitor(dataset.get_dataset_size()), BertLossCallBack(dataset.get_dataset_size()), ckpoint_cb]
    model.train(epoch_num, dataset, callbacks=callbacks)


def eval_result_print(assessment_method="accuracy", callback=None):
    """ print eval result """

    if assessment_method == "accuracy":
        print("acc_num {} , total_num {}, accuracy {:.6f}".format(callback.acc_num, callback.total_num,
                                                                  callback.acc_num / callback.total_num))
    elif assessment_method == "f1":
        print("Precision {:.6f} ".format(callback.TP / (callback.TP + callback.FP)))
        print("Recall {:.6f} ".format(callback.TP / (callback.TP + callback.FN)))
        print("F1 {:.6f} ".format(2 * callback.TP / (2 * callback.TP + callback.FP + callback.FN)))
    elif assessment_method == "mcc":
        print("MCC {:.6f} ".format(callback.cal()))
    elif assessment_method == "spearman_correlation":
        print("Spearman Correlation is {:.6f} ".format(callback.cal()[0]))
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")


def do_eval(dataset=None, network=None, num_class=2, assessment_method="accuracy", load_checkpoint_path=""):
    """ do eval """

    if load_checkpoint_path == "":
        raise ValueError("Finetune model missed, evaluation task must load finetune model!")
    net_for_pretraining = network(bert_net_cfg, False, num_class)
    net_for_pretraining.set_train(False)
    model = Model(net_for_pretraining)
    model.load_checkpoint((load_checkpoint_path))

    if assessment_method == "accuracy":
        callback = Accuracy()
    elif assessment_method == "f1":
        callback = F1(False, num_class)
    elif assessment_method == "mcc":
        callback = MCC()
    elif assessment_method == "spearman_correlation":
        callback = SpearmanCorrelation()
    else:
        raise ValueError("Assessment method not supported, support: [accuracy, f1, mcc, spearman_correlation]")

    columns_list = ["input_ids", "input_mask", "segment_ids", "label_ids"]
    for data in dataset.create_dict_iterator():
        input_data = []
        for i in columns_list:
            input_data.append(data[i])
        input_ids, input_mask, token_type_id, label_ids = input_data
        logits = model.predict(input_ids, input_mask, token_type_id, label_ids)
        callback.update(logits, label_ids)
    print("==============================================================")
    eval_result_print(assessment_method, callback)
    print("==============================================================")


def run_classifier():
    """run classifier task"""
    parser = argparse.ArgumentParser(description="run classifier")
    parser.add_argument("--device_target", type=str, default="Ascend", choices=["Ascend", "GPU"],
                        help="Device type, default is Ascend")
    parser.add_argument("--assessment_method", type=str, default="Accuracy",
                        choices=["Mcc", "Spearman_correlation", "Accuracy", "F1"],
                        help="assessment_method including [Mcc, Spearman_correlation, Accuracy, F1],\
                             default is Accuracy")
    parser.add_argument("--do_train", action="store_true",
                        help="Enable train, default is false")
    parser.add_argument("--do_eval", action="store_true",
                        help="Enable eval, default is false")
    parser.add_argument("--device_id", type=int, default=0, help="Device id, default is 0.")
    parser.add_argument("--epoch_num", type=int, default="1", help="Epoch number, default is 1.")
    parser.add_argument("--num_class", type=int, default="2", help="The number of class, default is 2.")
    parser.add_argument("--train_data_shuffle", type=str, default="true", choices=["true", "false"],
                        help="Enable train data shuffle, default is true")
    parser.add_argument("--eval_data_shuffle", type=str, default="false", choices=["true", "false"],
                        help="Enable eval data shuffle, default is false")
    parser.add_argument("--save_finetune_checkpoint_path", type=str, default="", help="Save checkpoint path")
    parser.add_argument("--load_pretrain_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--load_finetune_checkpoint_path", type=str, default="", help="Load checkpoint file path")
    parser.add_argument("--train_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--eval_data_file_path", type=str, default="",
                        help="Data path, it is better to use absolute path")
    parser.add_argument("--schema_file_path", type=str, default="",
                        help="Schema path, it is better to use absolute path")
    args_opt = parser.parse_args()
    epoch_num = args_opt.epoch_num
    assessment_method = args_opt.assessment_method.lower()
    load_pretrain_checkpoint_path = args_opt.load_pretrain_checkpoint_path
    save_finetune_checkpoint_path = args_opt.save_finetune_checkpoint_path
    load_finetune_checkpoint_path = args_opt.load_finetune_checkpoint_path

    if args_opt.do_train and args_opt.do_eval:
        raise ValueError("At least one of 'do_train' or 'do_eval' must be true")
    if args_opt.do_train and args_opt.train_data_file_path == "":
        raise ValueError("'train_data_file_path' must be set when do finetune task")
    if args_opt.do_eval and args_opt.eval_data_file_path == "":
        raise ValueError("'eval_data_file_path' must be set when do evaluation task")

    if args_opt.device_target == "Ascend":
        context.set_context(mode=context.GRAPH_MODE, device_target="Ascend", device_id=args_opt.device_id)
    elif args_opt.device_target == "GPU":
        context.set_context(mode=context.GRAPH_MODE, device_target="GPU", device_id=args_opt.device_id)
        if bert_net_cfg.compute_type != ts.float32:
            logger.warning('GPU only support fp32 temporarily, run with fp32.')
            bert_net_cfg.compute_type = ts.float32
    else:
        raise Exception("Target error, GPU or Ascend is supported.")

    netwithloss = BertCLS(bert_net_cfg, True, num_labels=args_opt.num_class, dropout_prob=0.1,
                          assessment_method=assessment_method)

    if args_opt.do_train:
        ds = create_classification_dataset(batch_size=optimizer_cfg.batch_size, repeat_count=1,
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.train_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.train_data_shuffle.lower() == "true"))
        do_train(ds, netwithloss, load_pretrain_checkpoint_path, save_finetune_checkpoint_path, epoch_num)

        if args_opt.do_eval:
            if save_finetune_checkpoint_path == "":
                load_finetune_checkpoint_dir = _cur_dir
            else:
                load_finetune_checkpoint_dir = make_directory(save_finetune_checkpoint_path)
            load_finetune_checkpoint_path = LoadNewestCkpt(load_finetune_checkpoint_dir,
                                                           ds.get_dataset_size(), epoch_num, "classifier")

    if args_opt.do_eval:
        ds = create_classification_dataset(batch_size=optimizer_cfg.batch_size, repeat_count=1,
                                           assessment_method=assessment_method,
                                           data_file_path=args_opt.eval_data_file_path,
                                           schema_file_path=args_opt.schema_file_path,
                                           do_shuffle=(args_opt.eval_data_shuffle.lower() == "true"))
        do_eval(ds, BertCLS, args_opt.num_class, assessment_method, load_finetune_checkpoint_path)

if __name__ == "__main__":
    run_classifier()