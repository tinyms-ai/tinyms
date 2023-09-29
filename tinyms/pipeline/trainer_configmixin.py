from .configmixin import ConfigMixin, _key_var_keyword, _key_var_position
from typing import NewType, Any, Union
from functools import partial, wraps
import inspect
from typing import TypeVar, Callable, Generic
import logging
from typing import get_origin, get_args
import pathlib

logger = logging.getLogger(__name__)
FromConfig = NewType('FromConfig', Any)


def _is_from_pre_config(annotation):
    if annotation is FromConfig:
        return True
    elif get_origin(annotation) is Union:
        type_list = get_args(annotation)
        if FromConfig in type_list:
            return True
    return False


_F = TypeVar('_F', bound=Callable[..., Any])


class TrainerConfigMixin(ConfigMixin):
    """
    A base class for trainer pipeline mixin. This class provides methods for
    saving and loading model config. A class that inherits from this class
    can apply `@save_config` to `__init__` method to record the
    config of the class.

    If you wrap `__init__` with `@save_config`, the argument of Ignore type
    will not be saved into the config. The SubFolder type will be saved into a sub folder.

    For a trainer, you may want to implement some methods like `train`, `eval`, `predict`.
    You can use `@set_from_config` to set the arguments from the config. The FromConfig
    type arguments having the following property.

    The arguments that are not in the config will be set to default value. The arguments set 
    in running time will override the arguments in the config.

    If you wrap a method with `@set_from_config`, you can use `BaseArgsFromConfig`
    to generate the arguments class. To use `BaseArgsFromConfig`, you should wrap the `__init__`
    method with `@copy_signature(Trainer.method)`. The arguments of `__init__` method should
    be `__init__(self, *args, **kwargs)`.

    You should define the arguments class in `__init__`
    method. The default name of the arguments class is `{method_name}_config`. You can
    change the name by passing the name to `@set_from_config(name)`.


    Examples:
        >>> class Trainer(TrainerConfigMixin):
        ...     @save_config
        ...     def __init__(self, train_args=None):
        ...        self.train_args = train_args
        ...
        ...     @set_from_config
        ...     def train(self, epoch: FromConfig):
        ...        ...
        >>>
        >>> class TrainConfig(BaseArgsFromConfig):
        ...     @copy_signature(Trainer.train)
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        >>>
        >>> train_config = TrainConfig(2)
        >>> trainer = Trainer(train_config=train_config)
        >>>
        >>> trainer.save_pretrained('model_config')
        >>> new_trainer = trainer.from_pretrained('model_config')
        >>>
        >>> new_trainer.train()
        >>>
        >>> new_trainer.train(4)

    """
    __prefix__ = "trainer"

    def _save_checkpoint(self, path):
        raise NotImplementedError

    def _load_checkpoint(self, path):
        raise NotImplementedError

    @classmethod
    def from_pretrained(cls, path: Union[str, pathlib.Path], repo: str, checkfiles: bool = True, download: bool = True) -> 'TrainerConfigMixin':
        """
            path (Union[str, pathlib.Path]): Path to save repo. Defaults to None.

            repo (str): The repo name.

            checkfiles (bool, optional): If this is set to False, this method will not check
            the files in `path`, and will not download from `repo`. Defaults to True.

            download (bool, optional): If this is set to False, this method will not download.
            If download is true, this method will first check whether the repo has been
            downloaded or completed, if not, it will download the wrong or missing files.
            If download is false, this method will check the whether the local repo in `path`
            is completed, if not, it will raise an Error.

        Returns:
            A instance of a subclass of TrainerConfigMixin.
        """
        return super().from_pretrained(path, repo, checkfiles)

    def init_model(self, model=None):
        ...

    def _compile(self):
        ...

    @classmethod
    def _check_cls_loaded(cls, loaded_cls):
        if cls is not TrainerConfigMixin:
            assert loaded_cls is cls, f"The repo loaded is not the same as the \
                class {cls.__name__}, it is {loaded_cls.__class__.__name__}"


class copy_signature(Generic[_F]):
    """
    A decorator to copy the signature of a function to another function.
    Using with `BaseArgsFromConfig` to generate the arguments class.
    """

    def __init__(self, target: _F) -> None:
        self.target_signature = inspect.signature(target)

    def __call__(self, wrapped: Callable[..., Any]) -> _F:
        def wrapped_with_signature(self_, *args, **kwargs):
            self_.func_signature = self.target_signature
            return wrapped(self_, *args, **kwargs)
        wrapped_with_signature.__signature__ = self.target_signature
        wrapped_with_signature.__name__ = wrapped.__name__
        return wrapped_with_signature


def _is_var_keyword_from_pre_config(func_signature):
    if _key_var_keyword(func_signature) is not None:
        return _is_from_pre_config(
            func_signature.parameters[_key_var_keyword(func_signature)].annotation)
    return False


def _func_keys_from_pre_config(func_signature):
    preconfg = []
    flag = False
    for v in func_signature.parameters.values():
        if not flag:
            flag = True
            continue
        if _is_from_pre_config(v.annotation):
            if v.name == _key_var_keyword(func_signature):
                preconfg.append(f"**{v.name}")
            elif v.name == _key_var_position(func_signature):
                preconfg.append(f"*{v.name}")
            else:
                preconfg.append(v.name)
    return tuple(preconfg)


def _func_keys_add_star_to_var(func_signature):
    func_keys = []
    flag = False
    for k in func_signature.parameters:
        if not flag:
            flag = True
            continue
        if k == _key_var_keyword(func_signature):
            func_keys.append(f"**{k}")
        elif k == _key_var_position(func_signature):
            func_keys.append(f"*{k}")
        else:
            func_keys.append(k)
    return tuple(func_keys)


class BaseArgsFromConfig(ConfigMixin):
    """
    The base class to generate arguments class for `@set_from_config`.
    """

    def __init__(self, *args, **kwargs):
        self.func_signature: inspect.Signature
        if _key_var_position(self.func_signature) is not None:
            raise AttributeError(f"Var position args {_key_var_position(self.func_signature)} is not supported")

        self.func_signature.bind(self, *args, **kwargs)

        not_in_preconfg_warning = []
        default_value_warning = []

        self._internal_config = {}
        allocated_kwargs = []
        func_keys = _func_keys_add_star_to_var(self.func_signature)
        func_from_pre_config = _func_keys_from_pre_config(self.func_signature)
        is_var_keyword_from_pre_config = _is_var_keyword_from_pre_config(self.func_signature)

        for idx, v in enumerate(args):
            kwargs[func_keys[idx]] = v

        for k, v in kwargs.items():
            if k in func_keys:
                if k in func_from_pre_config:
                    self._internal_config[k] = v
                else:
                    not_in_preconfg_warning.append(k)
            else:
                if is_var_keyword_from_pre_config:
                    self._internal_config[k] = v
                else:
                    not_in_preconfg_warning.append(k)
            allocated_kwargs.append(k)

        for k in set(func_from_pre_config) - set(allocated_kwargs):
            default_value_warning.append(k)

        if len(not_in_preconfg_warning) > 0:
            logger.warning(f"Parameters {not_in_preconfg_warning} will be ignored in pre config")
        if len(default_value_warning) > 0:
            logger.warning(f"Parameters {default_value_warning} will use default value")

    def __getattr__(self, item):
        return self._internal_config[item]

    def __getitem__(self, item):
        return self._internal_config[item]

    def keys(self):
        return self._internal_config.keys()

    def __contains__(self, key):
        return key in self._internal_config


def set_from_config(config_name_or_callable):
    """
    A decorator to set the arguments from the config. Using with `BaseArgsFromConfig`.
    """
    def _set_from_config(func: Callable[..., Any], config_name=None):
        if config_name is None:
            config_name = f"{func.__name__}_config"
        sig = inspect.signature(func)
        keys = _func_keys_add_star_to_var(sig)
        pre_config_keys = _func_keys_from_pre_config(sig)
        if config_name is None:
            config_name = f"{func.__name__}_config"

        @wraps(func)
        def wrapper(self, *args, **kwargs):
            if _key_var_position(sig) is not None:
                raise AttributeError(f"Var position args {_key_var_position(sig)} is not supported")

            if not hasattr(self, config_name):
                func(self, *args, **kwargs)

            allocated_keys = []
            for idx in range(len(args)):
                if idx < len(keys):
                    k = keys[idx]
                    if k in sig.parameters:
                        allocated_keys.append(k)

            allocated_keys += list(kwargs.keys())
            pre_config = getattr(self, config_name)

            for k in set(pre_config_keys) - set(allocated_keys):
                if k[0] != "*":
                    if k in pre_config:
                        kwargs[k] = pre_config[k]
                        allocated_keys.append(k)

            if _is_var_keyword_from_pre_config(sig) is not None:
                for k in set(getattr(self, config_name).keys()) - set(allocated_keys):
                    kwargs[k] = getattr(self, config_name)[k]
                    allocated_keys.append(k)

            return func(self, *args, **kwargs)

        return wrapper

    if callable(config_name_or_callable):
        return _set_from_config(config_name_or_callable, None)

    return partial(_set_from_config, config_name=config_name_or_callable)
