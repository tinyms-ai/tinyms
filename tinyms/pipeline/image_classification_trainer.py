from .trainer_configmixin import TrainerConfigMixin, copy_signature, \
    FromConfig, set_from_config, BaseArgsFromConfig
from ..model import Model
from ..losses import SoftmaxCrossEntropyWithLogits
from ..optimizers import Momentum
from .configmixin import SubFolder, save_config, Ignore


class ImageClassificationTrainer(Model, TrainerConfigMixin):
    @save_config
    def __init__(self,
                 model: Ignore = None,
                 optim: dict = {'optimizer': 'Momentum',
                                'params': {'learning_rate': 0.1, 'momentum': 0.9}},
                 loss: dict = {'loss': 'SoftmaxCrossEntropyWithLogits', 'params': {'sparse': True}},
                 metrics=['accuracy'],
                 train_config: SubFolder = None,
                 fit_config: SubFolder = None,
                 build_config: SubFolder = None,
                 eval_config: SubFolder = None,
                 predict_config: SubFolder = None):
        self._model = model
        self.optim = self._make_params(optim)
        self.loss = self._make_params(loss)
        self.metrics = metrics
        self.train_config = train_config
        self.fit_config = fit_config
        self.build_config = build_config
        self.eval_config = eval_config
        self.predict_config = predict_config

    def _make_params(self, config):
        if 'params' not in config:
            config['params'] = {}
        return config

    def _compile(self):
        if self.optim['optimizer'] == 'Momentum':
            optimizer = Momentum(params=self._network.trainable_params(), **self.optim['params'])
        if self.loss['loss'] == 'SoftmaxCrossEntropyWithLogits':
            loss_fn = SoftmaxCrossEntropyWithLogits(**self.loss['params'])
        if isinstance(self.metrics, (tuple, list)):
            self.metrics = set(self.metrics)
        return super().compile(loss_fn, optimizer, self.metrics)

    def init_model(self, model):
        """
        Send the model to trainer.
        """
        if model is None:
            model = self._model
        super().__init__(model)
        self._compile()

    @set_from_config
    def train(self,
              epoch: FromConfig,
              train_dataset: FromConfig,
              callbacks: FromConfig = None,
              dataset_sink_mode: FromConfig = False,
              sink_size: FromConfig = -1,
              initial_epoch: FromConfig = 0,
              **kwargs: FromConfig):
        """
        Training API.

        When setting pynative mode or CPU, the training process will be performed with dataset not sink.

        Note:
            If dataset_sink_mode is True, data will be sent to device. If the device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.

            When dataset_sink_mode is True, the `step_end` method of the instance of Callback will be called at the end
            of epoch.

            If dataset_sink_mode is True, dataset will be bound to this model and cannot be used by other models.

            If sink_size > 0, each epoch of the dataset can be traversed unlimited times until you get sink_size
            elements of the dataset. The next epoch continues to traverse from the end position of the previous
            traversal.

            The interface builds the computational graphs and then executes the computational graphs. However, when
            the `Model.build` is executed first, it only performs the graphs execution.

        Args:
            epoch (int): Total training epochs. Generally, train network will be trained on complete dataset per epoch.
                         If `dataset_sink_mode` is set to True and `sink_size` is greater than 0, each epoch will
                         train `sink_size` steps instead of total steps of dataset.
                         If `epoch` used with `initial_epoch`, it is to be understood as "final epoch".
            train_dataset (Dataset): A training dataset iterator. If `loss_fn` is defined, the data and label will be
                                     passed to the `network` and the `loss_fn` respectively, so a tuple (data, label)
                                     should be returned from dataset. If there is multiple data or labels, set `loss_fn`
                                     to None and implement calculation of loss in `network`,
                                     then a tuple (data1, data2, data3, ...) with all data returned from dataset will be
                                     passed to the `network`.
            callbacks (Optional[list[Callback], Callback]): List of callback objects or callback object,
                                                            which should be executed while training.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink. Default: True.
            sink_size (int): Control the amount of data in each sink. `sink_size` is invalid if `dataset_sink_mode`
                             is False.
                             If sink_size = -1, sink the complete dataset for each epoch.
                             If sink_size > 0, sink sink_size data for each epoch.
                             Default: -1.
            initial_epoch (int): Epoch at which to start train, it used for resuming a previous training run.
                                 Default: 0.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> loss_scale_manager = ms.FixedLossScaleManager()
            >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = ms.Model(net, loss_fn=loss, optimizer=optim, metrics=None,
            ...                  loss_scale_manager=loss_scale_manager)
            >>> model.train(2, dataset)
        """
        train_dataset = train_dataset.run()
        super().train(epoch, train_dataset, callbacks, dataset_sink_mode, sink_size, initial_epoch)

    @set_from_config
    def fit(self,
            epoch: FromConfig,
            train_dataset: FromConfig,
            valid_dataset: FromConfig = None,
            valid_frequency: FromConfig = 1,
            callbacks: FromConfig = None,
            dataset_sink_mode: FromConfig = False,
            valid_dataset_sink_mode: FromConfig = False,
            sink_size: FromConfig = -1,
            initial_epoch: FromConfig = 0):
        """
        Fit API.

        Evaluation process will be performed during training process if `valid_dataset` is provided.

        More details please refer to `mindspore.Model.train` and `mindspore.Model.eval`.

        Args:
            epoch (int): Total training epochs. Generally, train network will be trained on complete dataset per epoch.
                         If `dataset_sink_mode` is set to True and `sink_size` is greater than 0, each epoch will
                         train `sink_size` steps instead of total steps of dataset.
                         If `epoch` used with `initial_epoch`, it is to be understood as "final epoch".
            train_dataset (Dataset): A training dataset iterator. If `loss_fn` is defined, the data and label will be
                                     passed to the `network` and the `loss_fn` respectively, so a tuple (data, label)
                                     should be returned from dataset. If there is multiple data or labels, set `loss_fn`
                                     to None and implement calculation of loss in `network`,
                                     then a tuple (data1, data2, data3, ...) with all data returned from dataset
                                     will be passed to the `network`.
            valid_dataset (Dataset): Dataset to evaluate the model. If `valid_dataset` is provided, evaluation process
                                     will be performed on the end of training process. Default: None.
            valid_frequency (int, list): Only relevant if `valid_dataset` is provided.  If an integer, specifies
                         how many training epochs to run before a new validation run is performed,
                         e.g. `valid_frequency=2` runs validation every 2 epochs.
                         If a list, specifies the epochs on which to run validation,
                         e.g. `valid_frequency=[1, 5]` runs validation at the end of the 1st, 5th epochs.
                         Default: 1
            callbacks (Optional[list[Callback], Callback]): List of callback objects or callback object,
                                                            which should be executed while training.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the train data through dataset channel.
                                      Configure pynative mode or CPU, the training process will be performed with
                                      dataset not sink. Default: True.
            valid_dataset_sink_mode (bool): Determines whether to pass the validation data through dataset channel.
                                      Default: True.
            sink_size (int): Control the amount of data in each sink. `sink_size` is invalid if `dataset_sink_mode`
                             is False.
                             If sink_size = -1, sink the complete dataset for each epoch.
                             If sink_size > 0, sink sink_size data for each epoch.
                             Default: -1.
            initial_epoch (int): Epoch at which to start train, it useful for resuming a previous training run.
                                 Default: 0.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> train_dataset = create_custom_dataset()
            >>> valid_dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = ms.Model(net, loss_fn=loss, optimizer=optim, metrics={"accuracy"})
            >>> model.fit(2, train_dataset, valid_dataset)
        """
        super().fit(epoch, train_dataset, valid_dataset, valid_frequency, callbacks, dataset_sink_mode,
                    valid_dataset_sink_mode, sink_size, initial_epoch)

    @set_from_config
    def build(self,
              train_dataset: FromConfig = None,
              valid_dataset: FromConfig = None,
              sink_size: FromConfig = -1,
              epoch: FromConfig = 1):
        """
        Build computational graphs and data graphs with the sink mode.

        .. warning::
            This is an experimental prototype that is subject to change or deletion.

        Note:
            The interface builds the computational graphs, when the interface is executed first, 'Model.train' only
            performs the graphs execution. Pre-build process only supports `GRAPH_MODE` and `Ascend` target currently.
            It only supports dataset sink mode.

        Args:
            train_dataset (Dataset): A training dataset iterator. If `train_dataset` is defined, training graphs will be
                                     built. Default: None.
            valid_dataset (Dataset): An evaluating dataset iterator. If `valid_dataset` is defined, evaluation graphs
                                     will be built, and `metrics` in `Model` can not be None. Default: None.
            sink_size (int): Control the amount of data in each sink. Default: -1.
            epoch (int): Control the training epochs. Default: 1.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> loss_scale_manager = ms.FixedLossScaleManager()
            >>> optim = nn.Momentum(params=net.trainable_params(), learning_rate=0.1, momentum=0.9)
            >>> model = ms.Model(net, loss_fn=loss, optimizer=optim, metrics=None,
            ...                  loss_scale_manager=loss_scale_manager)
            >>> model.build(dataset, epoch=2)
            >>> model.train(2, dataset)
        """
        super().build(train_dataset, valid_dataset, sink_size, epoch)

    @set_from_config
    def eval(self,
             valid_dataset: FromConfig,
             callbacks: FromConfig = None,
             dataset_sink_mode: FromConfig = False):
        """
        Evaluation API.

        Configure to pynative mode or CPU, the evaluating process will be performed with dataset non-sink mode.

        Note:
            If dataset_sink_mode is True, data will be sent to device. At this point, the dataset will be bound to this
            model, so the dataset cannot be used by other models. If the device is Ascend, features
            of data will be transferred one by one. The limitation of data transmission per time is 256M.

            The interface builds the computational graphs and then executes the computational graphs. However, when
            the `Model.build` is executed first, it only performs the graphs execution.

        Args:
            valid_dataset (Dataset): Dataset to evaluate the model.
            callbacks (Optional[list(Callback), Callback]): List of callback objects or callback object,
                                                            which should be executed while evaluation.
                                                            Default: None.
            dataset_sink_mode (bool): Determines whether to pass the data through dataset channel.
                Default: True.

        Returns:
            Dict, the key is the metric name defined by users and the value is the metrics value for
            the model in the test mode.

        Examples:
            >>> import mindspore as ms
            >>> from mindspore import nn
            >>>
            >>> # For details about how to build the dataset, please refer to the tutorial
            >>> # document on the official website.
            >>> dataset = create_custom_dataset()
            >>> net = Net()
            >>> loss = nn.SoftmaxCrossEntropyWithLogits()
            >>> model = ms.Model(net, loss_fn=loss, optimizer=None, metrics={'acc'})
            >>> acc = model.eval(dataset, dataset_sink_mode=False)
        """
        super().eval(valid_dataset, callbacks, dataset_sink_mode)

    @set_from_config
    def predict(self,
                *predict_data: FromConfig,
                backend: FromConfig = None):
        """
        Generate output predictions for the input samples.

        Args:
            predict_data (Union[Tensor, list[Tensor], tuple[Tensor]], optional):
                The predict data, can be a single tensor,
                a list of tensor, or a tuple of tensor.

        Returns:
            Tensor, array(s) of predictions.

        Examples:
            >>> import numpy as np
            >>> import mindspore as ms
            >>> from mindspore import Tensor
            >>>
            >>> input_data = Tensor(np.random.randint(0, 255, [1, 1, 32, 32]), ms.float32)
            >>> model = ms.Model(Net())
            >>> result = model.predict(input_data)
        """
        super().predict(*predict_data, backend)


class ImageClassificationTrainConfig(BaseArgsFromConfig):
    @copy_signature(ImageClassificationTrainer.train)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ImageClassificationFitConfig(BaseArgsFromConfig):
    @copy_signature(ImageClassificationTrainer.fit)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ImageClassificationBuildConfig(BaseArgsFromConfig):
    @copy_signature(ImageClassificationTrainer.build)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ImageClassificationEvalConfig(BaseArgsFromConfig):
    @copy_signature(ImageClassificationTrainer.eval)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ImageClassificationPredictConfig(BaseArgsFromConfig):
    @copy_signature(ImageClassificationTrainer.predict)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
