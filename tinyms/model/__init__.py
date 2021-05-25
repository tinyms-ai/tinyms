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
    save_checkpoint as _save_checkpoint, export as _export, load as _load

from ..layers import GraphLayer
from .lenet5 import lenet5, LeNet
from .resnet50 import resnet50, ResNet
from .mobilenetv2 import mobilenetv2, MobileNetV2
from .ssd300 import ssd300_mobilenetv2, SSD300
from .cycle_gan.cycle_gan import cycle_gan, cycle_gan_infer
from .densenet100 import densenet100, DenseNet
from .alexnet import alexnet, AlexNet
from .sentimentnet import sentimentnet, SentimentNet
from .bert import bert, Bert
from .vgg import vgg11, vgg13, vgg16, vgg19, VGG

__all__ = [
    'Model',
    'load',
    'lenet5', 'LeNet',
    'resnet50', 'ResNet',
    'mobilenetv2', 'MobileNetV2',
    'ssd300_mobilenetv2', 'SSD300',
    'cycle_gan', 'cycle_gan_infer',
    'densenet100', 'DenseNet',
    'alexnet', 'AlexNet',
    'sentimentnet', 'SentimentNet',
    'bert', 'Bert',
    'vgg11', 'vgg13', 'vgg16', 'vgg19', 'VGG'
]


class Model(_Model):
    """
    High-Level API for Training or Evaluation.

    `Model` groups layers into an object with training and inference features.

    Args:
        network (layers.Layer): A training or testing network.

    Examples:
        >>> from tinyms.model import Model, lenet5
        >>> form tinyms.losses import SoftmaxCrossEntropyWithLogits
        >>> from tinyms.optimizers import Momentum
        >>>
        >>> net = lenet5(class_num=10)
        >>> model = Model(net)
        >>> net_loss = SoftmaxCrossEntropyWithLogits()
        >>> net_opt = Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
        >>> model.compile(loss_fn=net_loss, optimizer=net_opt, metrics=None)
        >>> # For details about how to build the dataset, please refer to the API document on the official website.
        >>> ds_train = create_custom_dataset()
        >>> model.train(2, ds_train)
    """

    def __init__(self, network):
        super(Model, self).__init__(network)

    def compile(self, loss_fn=None, optimizer=None, metrics=None, eval_network=None,
                amp_level="O0", **kwargs):
        """
        High-Level API for configure the train or eval network.

        Args:
            loss_fn (layers.Layer): Objective function, if loss_fn is None, the
                                network should contain the logic of loss and grads calculation, and the logic
                                of parallel if needed. Default: None.
            optimizer (layers.Layer): Optimizer for updating the weights. Default: None.
            metrics (Union[dict, set]): A Dictionary or a set of metrics to be evaluated by the model during
                            training and testing. eg: {'accuracy', 'recall'}. Default: None.
            eval_network (layers.Layer): Network for evaluation. If not defined, `network` and `loss_fn` would be wrapped as
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
        """
        Loads checkpoint info from a specified file.

        Args:
            ckpt_file_name (str): Checkpoint file name.
            strict_load (bool): Whether to strict load the parameter into net. If False, it will load parameter
                            in the param_dict into net with the same suffix. Default: False.

        Returns:
            Dict, key is parameter name, value is a Parameter.

        Raises:
            ValueError: Checkpoint file is incorrect.

        Examples:
            >>> ckpt_file_name = "./checkpoint/LeNet5-1_32.ckpt"
            >>> param_dict = model.load_checkpoint(ckpt_file_name)
        """
        return _load_checkpoint(ckpt_file_name, net=self._network,
                                strict_load=strict_load)

    def save_checkpoint(self, ckpt_file_name):
        """
        Saves checkpoint info to a specified file.

        Args:
            ckpt_file_name (str): Checkpoint file name. If the file name already exists, it will be overwritten.

        Raises:
            TypeError: If the parameter save_obj is not layers.Layer or list type.
        """
        return _save_checkpoint(self._network, ckpt_file_name)

    def export(self, inputs, file_name, file_format='MINDIR', **kwargs):
        """
        Export the TinyMS prediction model to a file in the specified format.

        Args:
            inputs (Tensor): Inputs of the `net`.
            file_name (str): File name of the model to be exported.
            file_format (str): MindSpore currently supports `AIR`, `ONNX` and `MINDIR` format for exported model.
                Default: MINDIR.

                - AIR: Ascend Intermediate Representation. An intermediate representation format of Ascend model.
                Recommended suffix for output file is '.air'.

                - ONNX: Open Neural Network eXchange. An open format built to represent machine learning models.
                Recommended suffix for output file is '.onnx'.

                - MINDIR: MindSpore Native Intermediate Representation for Anf. An intermediate representation format
                for MindSpore models.
                Recommended suffix for output file is '.mindir'.

            kwargs (dict): Configuration options dictionary.

                - quant_mode: The mode of quant.
                - mean: Input data mean. Default: 127.5.
                - std_dev: Input data variance. Default: 127.5.
        """
        return _export(self._network, inputs, file_name=file_name,
                       file_format=file_format, **kwargs)


def load(file_name):
    """
    Load MindIR graph and return the network with parameters.

    The returned object is wrapperred by a `GraphLayer`. However, there are some limitations to
    the current use of `GraphLayer`, see class :class:`tinyms.layers.GraphLayer` for more details.

    Args:
        file_name (str): MindIR file name.

    Returns:
        GraphLayer instance, a compiled graph with parameters.

    Raises:
        ValueError: MindIR file is incorrect.

    Examples:
        >>> import tinyms as ts
        >>> import tinyms.layers as layers
        >>> from tinyms.model import Model, load
        >>>
        >>> net = layers.Conv2d(1, 1, kernel_size=3)
        >>> model = Model(net)
        >>> input = ts.ones([1, 1, 3, 3])
        >>> model.export(input, "net", file_format="MINDIR")
        ...
        >>> net = load("net.mindir")
        >>> print(net(input))
        [[[[ 0.02548009  0.04010789  0.03120251]
           [ 0.00268656  0.02353744  0.03807815]
           [-0.00896441 -0.00303641  0.01502199]]]]
    """
    graph = _load(file_name)
    return GraphLayer(graph)
