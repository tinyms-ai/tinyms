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

from mindspore import Model as _Model
from mindspore.train.serialization import load_checkpoint as _load_checkpoint, \
    save_checkpoint as _save_checkpoint, export as _export

from .lenet5 import lenet5, LeNet
from .resnet50 import resnet50, ResNet
from .mobilenetv2 import mobilenetv2, mobilenetv2_infer, MobileNetV2
from .ssd300 import ssd300_mobilenetv2, ssd300_infer, SSD300
from .cycle_gan.cycle_gan import cycle_gan, cycle_gan_infer

__all__ = [
    'Model',
    'lenet5', 'LeNet',
    'resnet50', 'ResNet',
    'mobilenetv2', 'mobilenetv2_infer', 'MobileNetV2',
    'ssd300_mobilenetv2', 'ssd300_infer', 'SSD300',
    'cycle_gan', 'cycle_gan_infer',
]


class Model(_Model):
    """
    High-Level API for Training or Evaluation.

    `Model` groups layers into an object with training and inference features.

    Args:
        network (Layer): A training or testing network.
    """

    def __init__(self, network):
        super(Model, self).__init__(network)

    def compile(self, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                amp_level="O0", **kwargs):
        """
        High-Level API for configure the train or eval network.

        Args:
            loss_fn (Layer): Objective function, if loss_fn is None, the
                                network should contain the logic of loss and grads calculation, and the logic
                                of parallel if needed. Default: None.
            optimizer (Layer): Optimizer for updating the weights. Default: None.
            metrics (Union[dict, set]): A Dictionary or a set of metrics to be evaluated by the model during
                            training and testing. eg: {'accuracy', 'recall'}. Default: None.
            eval_network (Layer): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
                            `eval_network`. Default: None.
            amp_level (str): Option for argument `level` in `mindspore.amp.build_train_network`, level for mixed
                precision training. Supports ["O0", "O2", "O3", "auto"]. Default: "O0".

                - O0: Do not change.
                - O2: Cast network to float16, keep batchnorm run in float32, using dynamic loss scale.
                - O3: Cast network to float16, with additional property 'keep_batchnorm_fp32=False'.
                - auto: Set to level to recommended level in different devices. Set level to O2 on GPU, Set
                level to O3 Ascend. The recommended level is choose by the export experience, cannot
                always generalize. User should specify the level for special network.

                O2 is recommended on `GPU`, O3 is recommended on `Ascend`.
        """
        # configure the train network
        self._loss_fn = loss_fn
        self._optimizer = optimizer
        self._amp_level = amp_level
        self._check_kwargs(kwargs)
        self._process_amp_args(kwargs)
        self._train_network = self._build_train_network()
        # configure the eval network
        self._build_eval_network(metrics, eval_network=eval_network, eval_indexes=None)
        # configure the predict network
        self._build_predict_network()

    def load_checkpoint(self, ckpt_file_name, strict_load=False):
        return _load_checkpoint(ckpt_file_name, net=self._network,
                                strict_load=strict_load)

    def save_checkpoint(self, ckpt_file_name):
        return _save_checkpoint(self._network, ckpt_file_name)

    def export(self, *inputs, file_name, file_format='MINDIR', **kwargs):
        return _export(self._network, inputs, file_name,
                       file_format=file_format, **kwargs)
