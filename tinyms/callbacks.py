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
"""Callback related classes and functions in model training phase."""
import time
import numpy as np
import math
from mindspore.train import callback
from mindspore.train.callback import *
from . import Tensor

__all__ = ['LossTimeMonitor', 'BertLossCallBack']
__all__.extend(callback.__all__)


class LossTimeMonitor(Callback):
    """
    Monitor loss and time.

    Args:
        lr_init (numpy.ndarray): Train learning rate. Default: None.

    Returns:
        None

    Examples:
        >>> from tinyms import Tensor
        >>> from tinyms.callbacks import LossTimeMonitor
        >>>
        >>> LossTimeMonitor(lr_init=Tensor([0.05] * 100).asnumpy())
    """

    def __init__(self, lr_init=None):
        super(LossTimeMonitor, self).__init__()
        self.lr_init = lr_init
        self.lr_init_len = len(lr_init)

    def epoch_begin(self, run_context):
        self.losses = []
        self.epoch_time = time.time()

    def epoch_end(self, run_context):
        cb_params = run_context.original_args()

        epoch_mseconds = (time.time() - self.epoch_time) * 1000
        per_step_mseconds = epoch_mseconds / cb_params.batch_num
        print("epoch time: {:5.3f}, per step time: {:5.3f}, avg loss: {:5.3f}".format(epoch_mseconds,
                                                                                      per_step_mseconds,
                                                                                      np.mean(self.losses)))

    def step_begin(self, run_context):
        self.step_time = time.time()

    def step_end(self, run_context):
        cb_params = run_context.original_args()
        step_mseconds = (time.time() - self.step_time) * 1000
        step_loss = cb_params.net_outputs

        if isinstance(step_loss, (tuple, list)) and isinstance(step_loss[0], Tensor):
            step_loss = step_loss[0]
        if isinstance(step_loss, Tensor):
            step_loss = np.mean(step_loss.asnumpy())

        self.losses.append(step_loss)
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num

        print("epoch: [{:3d}/{:3d}], step:[{:5d}/{:5d}], loss:[{:5.3f}/{:5.3f}], time:[{:5.3f}], lr:[{:5.3f}]".format(
            cb_params.cur_epoch_num -
            1, cb_params.epoch_num, cur_step_in_epoch, cb_params.batch_num, step_loss,
            np.mean(self.losses), step_mseconds, self.lr_init[cb_params.cur_step_num - 1]))


class BertLossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.

    Args:
        dataset_size (int): Print loss every times. Default: 1.

    Returns:
        None

    Examples:
        >>> from tinyms.callbacks import BertLossCallBack
        >>>
        >>> BertLossCallBack(dataset_size=1)
    """

    def __init__(self, dataset_size=1):
        super(BertLossCallBack, self).__init__()
        self._dataset_size = dataset_size
    def step_end(self, run_context):
        """
        Print loss after each step
        """
        cb_params = run_context.original_args()
        if self._dataset_size > 0:
            percent, epoch_num = math.modf(cb_params.cur_step_num / self._dataset_size)
            if percent == 0:
                percent = 1
                epoch_num -= 1
            print("epoch: {}, current epoch percent: {}, step: {}, outputs are {}"
                  .format(int(epoch_num), "%.3f" % percent, cb_params.cur_step_num, str(cb_params.net_outputs)))
        else:
            print("epoch: {}, step: {}, outputs are {}".format(cb_params.cur_epoch_num, cb_params.cur_step_num,
                                                               str(cb_params.net_outputs)))




