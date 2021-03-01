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

from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossTimeMonitor


def mobilenetv2_cb(device_target, lr, is_saving_checkpoint, save_checkpoint_epochs, step_size):
    cb = None
    if device_target in ("CPU", "GPU"):
        cb = [LossTimeMonitor(lr_init=lr.asnumpy())]

        if is_saving_checkpoint:
            config_ck = CheckpointConfig(save_checkpoint_steps=save_checkpoint_epochs * step_size,
                                         keep_checkpoint_max=10)
            ckpt_save_dir = "./"
            ckpt_cb = ModelCheckpoint(prefix="mobilenetv2_cifar10", directory=ckpt_save_dir, config=config_ck)
            cb += [ckpt_cb]
    return cb

