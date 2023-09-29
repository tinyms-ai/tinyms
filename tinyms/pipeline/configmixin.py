import copy

import functools
import inspect
from typing import Dict, Callable, Union
import json
import pathlib
import logging
from typing import NewType, Any, get_origin, get_args, TypeVar
from importlib.metadata import version
from mindspore.train.serialization import load_checkpoint, save_checkpoint
from .download import RepoDownloaderWithCode, make_filelist
import shutil

logger = logging.getLogger(__name__)


Ignore = NewType('Ignore', Any)
SubFolder = NewType('SubFolder', Any)

_T = TypeVar('_T')


def _key_var_keyword(func_signature):
    name = None
    for key, v in func_signature.parameters.items():
        if v.kind == inspect.Parameter.VAR_KEYWORD:
            name = key
            break
    return name


def _key_var_position(func_signature):
    name = None
    for key, v in func_signature.parameters.items():
        if v.kind == inspect.Parameter.VAR_POSITIONAL:
            name = key
            break
    return name


def wrap_config_mixin(cls: _T) -> _T:
    """
    Wrap a class with ConfigMixin. This function will add ConfigMixin to the base class of 
    the class and wrap `__init__` method with `@save_config`.

    Args:
        cls (object): a class to be wrapped.

    Returns:
        return a wrapped class. This class is a subclass of ConfigMixin and the `__init__`
        method is wrapped with `@save_config`.

    Examples:
        >>> class Model:
        ...     ...
        >>>
        >>> Model = wrap_config_mixin(Model)
        >>> model = Model()
        >>> model.save_pretrained("config_path")
    """
    class Wrapped(cls, ConfigMixin):
        ...
    Wrapped.__module__ = cls.__module__
    Wrapped.__name__ = cls.__name__
    Wrapped.__init__ = save_config(cls.__init__)
    return Wrapped


def _walk(obj, func: Callable, walk_path=None, replace=False, invoke_func_on_first=False):
    if walk_path is None:
        walk_path = []
    elif not isinstance(walk_path, (list, tuple)):
        walk_path = [walk_path]

    having_walk_path = 'walk_path' in inspect.signature(func).parameters

    if isinstance(obj, dict):
        new_dict = []
        for k, v in obj.items():
            kwargs = {'walk_path': walk_path + [k]} if having_walk_path else {}
            new_dict.append((func(k, **kwargs), func(_walk(v, func, replace=replace, **kwargs), **kwargs)))
        if replace:
            obj = dict(new_dict)

    elif isinstance(obj, list):
        new_list = []
        for idx, v in enumerate(obj):
            kwargs = {'walk_path': walk_path + [idx]} if having_walk_path else {}
            new_list.append(func(_walk(v, func, replace=replace, **kwargs), **kwargs))
        if replace:
            obj = new_list

    elif isinstance(obj, tuple):
        new_tuple = []
        for idx, v in enumerate(obj):
            kwargs = {'walk_path': walk_path + [idx]} if having_walk_path else {}
            new_tuple.append(func(_walk(v, func, replace=replace, **kwargs), **kwargs))
        if replace:
            obj = tuple(new_tuple)

    elif isinstance(obj, PRIMITIVE_TYPE + CONFIG_TYPE) or obj is inspect.Parameter.empty:
        kwargs = {'walk_path': walk_path[:]} if having_walk_path else {}
        r = func(obj, **kwargs)
        if replace:
            obj = r

    else:
        raise TypeError(f'walk obj type {type(obj)} is not supported')

    if invoke_func_on_first:
        kwargs = {'walk_path': walk_path[:]} if having_walk_path else {}
        return func(obj, **kwargs)
    return obj


def _func_args_dict_with_default_value(func, args: list, kwargs: dict):
    init_args = inspect.signature(func)
    kwargs = copy.copy(kwargs)

    init_args_keys = list(init_args.parameters.keys())

    args_kwargs = {}
    for idx, v in enumerate(args):
        args_kwargs[init_args.parameters[init_args_keys[idx]].name] = v
    kwargs.update(args_kwargs)

    params = dict(init_args.parameters)
    for k in kwargs:
        if k in params:
            del params[k]
    for k in params:
        kwargs[k] = params[k].default
    return kwargs


def _attrtype_args_list_factory(Type):
    def _args_list(func):
        init_args = inspect.signature(func)
        args_with_type = []
        for k, v in init_args.parameters.items():
            if v.annotation is Type:
                args_with_type.append(k)
            elif get_origin(v.annotation) is Union:
                type_list = get_args(v.annotation)
                if Type in type_list:
                    args_with_type.append(k)
        return args_with_type
    return _args_list


_ignore_args_list = _attrtype_args_list_factory(Ignore)
_subfolder_args_list = _attrtype_args_list_factory(SubFolder)


class TypeConverter:

    @classmethod
    def obj_to_config(cls, obj):
        if isinstance(obj, CONFIG_TYPE):
            module_path = cls._module_path(obj)
            _internal_config = obj._config
            return cls._config_class_to_dict(module_path, _internal_config)
        return obj

    @classmethod
    def obj_to_config_and_seperate_save(cls, obj, walk_path):
        if not isinstance(walk_path, (list, tuple)):
            walk_path = [walk_path]
        if isinstance(obj, CONFIG_TYPE):
            module_path = cls._module_path(obj)
            path = pathlib.Path()
            for p in walk_path:
                path = path / str(p)
            _internal_config = obj._internal_config_dict_converted_seperate_save(path, module_path)
            if obj.__subfolder_save__:
                _internal_config = {'__subfolder__': None}
            return cls._config_class_to_dict(module_path, _internal_config)
        return obj

    @classmethod
    def _load_config(cls, config, walk_path):
        if isinstance(config, dict) and '__module__' in config:
            path = pathlib.Path()
            for p in walk_path:
                path = path / str(p)
            module = cls._load_module(config)
            if '__subfolder__' in config:
                config = module._load_config(path)
            return config
        return config

    @staticmethod
    def _module_path(obj):
        return f"{obj.__module__}.{obj.__class__.__name__}"

    @classmethod
    def _config_class_to_dict(cls, module_path, _internal_config):
        cls.attribution_check(_internal_config)
        config = {'__module__': module_path}
        config.update(_internal_config)
        return config

    @staticmethod
    def _add_module_to_dict(config, module):
        config = copy.copy(config)
        config['__module__'] = module
        return config

    @classmethod
    def config_to_obj(cls, config):
        if isinstance(config, Dict):
            if '__module__' in config:
                module = cls._load_module(config)
                config = cls._remove_internal_argument(config)
                if issubclass(module, CONFIG_TYPE):
                    config = module._from_config_pre_process(config)
                return module(**config)
        return config

    @classmethod
    def _load_module(cls, config):
        cls._version_check(config)
        loaded_cls = _locate(config['__module__'])
        if not isinstance(loaded_cls, ConfigMixin):
            loaded_cls = wrap_config_mixin(loaded_cls)
        return loaded_cls

    @classmethod
    def _load_check(cls, config):
        if '__module__' not in config:
            raise KeyError(f'__module__ is not found in config: {config}')

    @staticmethod
    def is_valid_type(obj):
        if isinstance(obj, VALID_TYPE) or obj is inspect.Parameter.empty:
            return True
        raise TypeError(f'obj type {type(obj)} is not supported')

    @staticmethod
    def attribution_check(config):
        if '__module__' in config:
            raise KeyError('__module__ is not allowed as a argument in __init__ method')
        if '__version__' in config:
            raise KeyError('__version__ is not allowed as a argument in __init__ method')

    @staticmethod
    def _version_check(config):
        pass

    @staticmethod
    def _remove_internal_argument(config):
        config = copy.copy(config)
        if '__module__' in config:
            del config['__module__']

        if '__version__' in config:
            del config['__version__']
        return config

    @staticmethod
    def add_version(config):
        config = copy.copy(config)
        config['__version__'] = version('tinyms')
        return config

    @classmethod
    def check_module(cls, config_dict, module):
        if '__module__' not in config_dict:
            raise KeyError(f'__module__ is not found in {module}')
        # if config_dict['__module__'] != cls._module_path(module):
        #     raise KeyError(f'__module__ is not {module}')


def _locate(path: str):
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.
    """
    if path == "":
        raise ImportError("Empty path")
    from importlib import import_module
    from types import ModuleType

    parts = [part for part in path.split(".")]
    for part in parts:
        if not len(part):
            raise ValueError(
                f"Error loading '{path}': invalid dotstring."
                + "\nRelative imports are not supported."
            )
    assert len(parts) > 0
    part0 = parts[0]
    try:
        obj = import_module(part0)
    except Exception as exc_import:
        raise ImportError(
            f"Error loading '{path}':\n{repr(exc_import)}"
            + f"\nAre you sure that module '{part0}' is installed?"
        ) from exc_import
    for m in range(1, len(parts)):
        part = parts[m]

        parent_dotpath = ".".join(parts[:m])
        if isinstance(obj, ModuleType):
            mod = ".".join(parts[: m + 1])
            try:
                obj = import_module(mod)
                continue
            except ModuleNotFoundError as exc_import:
                try:
                    obj = getattr(obj, part)
                except AttributeError:
                    raise ImportError(
                        f"Error loading '{path}':\n{repr(exc_import)}"
                        + f"\nAre you sure that '{part}' is an attribute of '{parent_dotpath}'?"
                    ) from exc_import
            except Exception as exc_import:
                raise ImportError(
                    f"Error loading '{path}':\n{repr(exc_import)}"
                ) from exc_import
    return obj


def save_config(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        self = args[0]
        if not isinstance(self, ConfigMixin):
            raise RuntimeError(
                f"`@save_config` was applied to {self.__class__.__name__} init method, but this class does "
                "not inherit from `ConfigMixin`."
            )
        if func.__name__ != '__init__':
            raise RuntimeError(
                f"`@save_config` was applied to {self.__class__.__name__} init method, but this mathod is "
                f"`{func.__name__}`."
            )

        config = _func_args_dict_with_default_value(func, args, kwargs)
        config = self._filter_ignore_keys(config, func)
        config = self._set_subfolder_save(config, func)
        self.__type_converter__.attribution_check(config)

        self._config_value_type_check(config)
        self._internal_config = config
        func(*args, **kwargs)

    return wrapped


class ConfigMixin:
    """
    A base class for pipeline mixin. This class provides methods for
    saving and loading model config. A class that inherits from this class
    can apply `@save_config` to `__init__` method to record the
    config of the class.
    If you wrap `__init__` with `@save_config`, the argument of Ignore type
    will not be saved into the config. The SubFolder type will be saved into a sub folder.

    Set `__prefix__` to change the name of the folder to save config and checkpoint.
    Set `__weight__` to change the name of the checkpoint file.

    Examples:
        >>> class Model(ConfigMixin):
        ...     @save_config
        ...     def __init__(
        ...        self, a: Ignore=1, b: SubFolder=2, c: Union[Ignore, int]=3, d: Union[SubFolder, int]=4):
        ...        ...
        >>>
        >>> model = Model()
        >>> model.save_pretrained('model_config')
        >>> new_model = Model.from_pretrained('model_config')

        >>> config = model.config
        >>> new_model = Model.from_config(config)

    """
    __save_name__ = None
    __type_converter__ = TypeConverter()
    __subfolder_save__ = None
    _internal_config: Dict
    __prefix__ = "model"
    __weight__ = 'weight.ckpt'
    __cache_path_before__: pathlib.Path
    __repo__: str

    @classmethod
    def from_config(cls, config: Dict):
        """Instantiates a Model from config dictionary.

        Args:
            config (Dict): A dictionary specifying the config of the model. This can be
            obtained by calling `model.config` or `model.load_config('config_path'))`.

        Examples:
            >>> class Model(ConfigMixin):
            ...     @save_config
            ...     def __init__(
            ...        self, a: Ignore=1, b: SubFolder=2, c: Union[Ignore, int]=3, d: Union[SubFolder, int]=4):
            ...        ...
            >>>
            >>> model = Model()
            >>> config = model.config
            >>> new_model = Model.from_config(config)

        Returns:
            A instance of a subclass of ConfigMixin.
        """
        if cls is not ConfigMixin:
            cls.__type_converter__.check_module(config, cls)
        return _walk(config, cls.__type_converter__.config_to_obj, replace=True, invoke_func_on_first=True)

    def save_config(self, path: Union[str, pathlib.Path]):
        """
        Save model config to a json file.

        Args:
            path (Union[str, pathlib.Path]): The path to save the config.

        Examples:
            >>> class Model(ConfigMixin):
            ...     @save_config
            ...     def __init__(self):
            ...        ...
            >>>
            >>> model = Model()
            >>> model.save_config("config_path")
        """
        if not hasattr(self, '_internal_config'):
            raise AttributeError(f'`{self.__class__.__name__}` object has no atrribute `_internal_config`.'
                                 'You can apply @save_config to init method to record `_internal_config` automatically.')

        path = pathlib.Path(path)
        path = path / self.__prefix__
        path.mkdir(parents=True, exist_ok=True)
        path = path / self._save_name()
        if path.exists():
            raise FileExistsError(f'{path} already exists')

        config = self.__type_converter__.obj_to_config_and_seperate_save(self, path.parent)
        config = self.__type_converter__.add_version(config)
        with path.open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

        logger.info(f"Model config saved in directory {path.parent}")

    @property
    def config(self):
        return self.__type_converter__.obj_to_config(self)

    @property
    def _config(self):
        return _walk(self._internal_config, self.__type_converter__.obj_to_config, replace=True)

    def _internal_config_dict_converted_seperate_save(self, path, module_name=None):
        path = pathlib.Path(path)

        config = _walk(
            self._internal_config,
            self.__type_converter__.obj_to_config_and_seperate_save,
            replace=True,
            walk_path=path,)

        if self.__subfolder_save__:
            path.mkdir(parents=True, exist_ok=True)
            with (path / self._save_name()).open("w") as f:
                if module_name is not None:
                    _config = self.__type_converter__._add_module_to_dict(config, module_name)
                else:
                    _config = config
                json.dump(_config, f, indent=2, sort_keys=True)

        return config

    @property
    def _internal_config_dict(self):
        return copy.copy(self._internal_config)

    @classmethod
    def _from_config_pre_process(cls, config):
        cls._config_value_type_check(config)
        cls._class_attribution_check(config)
        config = cls._filter_unexpected_keys(config)
        return config

    @classmethod
    def load_config(cls, path: Union[str, pathlib.Path]):
        """
        Load model config from a path.

        Args:
            path (Union[str, pathlib.Path]): Path to load the config.

        Returns:
            A instance of a subclass of ConfigMixin.
        """
        path = pathlib.Path(path)
        path = path / cls.__prefix__
        return cls._load_config(path)

    @classmethod
    def _load_config(cls, path):
        path = pathlib.Path(path)
        path = path / cls._save_name()
        if not path.exists():
            raise FileNotFoundError(f'{path} not found')
        with open(path, 'r') as f:
            config = json.load(f)
        cls.__type_converter__._load_check(config)
        return _walk(config, cls.__type_converter__._load_config, replace=True, walk_path=path.parent)

    @classmethod
    def _config_value_type_check(cls, config):
        for k, v in config.items():
            try:
                _walk(v, cls.__type_converter__.is_valid_type)
            except TypeError:
                raise TypeError(f'The value of {k}={v} is not valid type')

    @classmethod
    def _class_attribution_check(cls, config):
        config_self = _func_args_dict_with_default_value(cls.__init__, [], {})
        config_self = cls._filter_ignore_keys(config_self, cls.__init__)
        config_self_keys = set(config_self.keys())
        config_keys = set(config.keys())
        missing_keys = config_self_keys - config_keys
        unexpected_keys = config_keys - config_self_keys

        if len(missing_keys) > 0:
            non_default_key = [k for k in missing_keys if config_self[k] is inspect.Parameter.empty]
            if len(non_default_key) > 0:
                raise AttributeError(f'missing keys {list(non_default_key)} to instantiate {cls.__name__}')
            missing_keys = tuple(missing_keys)
            if len(missing_keys) == 1:
                missing_keys = missing_keys[0]
            logger.warning(f'missing keys {missing_keys} to instantiate {cls.__name__}, using default value')

        if len(unexpected_keys) > 0:
            unexpected_keys = tuple(unexpected_keys)
            if len(unexpected_keys) == 1:
                unexpected_keys = unexpected_keys[0]
            logger.warning(f'unexpected keys {unexpected_keys} to instantiate {cls.__name__}, ignore them')

    @classmethod
    def _filter_unexpected_keys(cls, config):
        config_self = _func_args_dict_with_default_value(cls.__init__, [], {})
        config_self = cls._filter_ignore_keys(config_self, cls.__init__)
        config_self_keys = set(config_self.keys())
        config_keys = set(config.keys())
        unexpected_keys = config_keys - config_self_keys

        config = copy.copy(config)
        for k in unexpected_keys:
            del config[k]
        return config

    @classmethod
    def _filter_ignore_keys(cls, config, func):
        config = copy.copy(config)
        del config['self']
        ignore_keys = _ignore_args_list(func)
        sig = inspect.signature(func)
        var_position = _key_var_position(sig)
        var_keyword = _key_var_keyword(sig)
        if var_position is not None:
            ignore_keys.append(var_position)
        if var_keyword is not None:
            ignore_keys.append(var_keyword)
        if ignore_keys:
            for k in ignore_keys:
                if k in config:
                    del config[k]
        return config

    def _save_checkpoint(self, path):
        path = pathlib.Path(path)
        path = path / self.__prefix__ / self.__weight__
        save_checkpoint(self, str(path))

    def _load_checkpoint(self, path):
        path = pathlib.Path(path)
        path = path / self.__prefix__ / self.__weight__
        load_checkpoint(str(path), self)

    def save_pretrained(self, path: Union[str, pathlib.Path]):
        """
        Save model config and checkpoint to a path.

        Args:
            path (Union[str, pathlib.Path]): The path to save the config and checkpoint.

        Examples:
            >>> class Model(ConfigMixin):
            ...     @save_config
            ...     def __init__(self):
            ...        ...
            >>>
            >>> model = Model()
            >>> model.save_pretrained("model_path")
        """
        path = pathlib.Path(path)
        if hasattr(self, "__repo__"):
            code_path = self.__cache_path_before__ / f"{self.__repo__}_code"
            code_new_path = path / f"{path.name}_code"
            if code_new_path.exists():
                code_new_path.unlink()
            shutil.copytree(code_path, code_new_path)

        self.save_config(path)
        self._save_checkpoint(path)
        path = path / self.__prefix__
        make_filelist(path)

    @classmethod
    def from_pretrained(
            cls, path: Union[str, pathlib.Path],
            repo, checkfiles: bool = True, download: bool = True):
        """
        Load model config and checkpoint from a path or a repo.

        Args:
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
            A instance of a subclass of ConfigMixin.
        """
        if not download:
            path = pathlib.Path(path) / repo
            repo = None
        assert path is not None or repo is not None, \
            'You have to set path or repo to load pretrained model'
        path = pathlib.Path(path)
        _path = path
        if repo is None:
            _path = pathlib.Path(path) / cls.__prefix__
        downloader = RepoDownloaderWithCode(_path, repo, checkfiles)
        downloader.download()

        __repo__ = path.name if repo is None else repo
        __path__ = path.parent if repo is None else path

        if repo is not None:
            path = path / (repo).split('/')[0]
        loaded_cls = cls.load_config(path)
        loaded_cls = cls.from_config(loaded_cls)

        cls._check_cls_loaded(loaded_cls)
        loaded_cls._load_checkpoint(path)
        loaded_cls.__cache_path_before__ = __path__
        loaded_cls.__repo__ = __repo__
        return loaded_cls

    @classmethod
    def _check_cls_loaded(cls, loaded_cls):
        if cls is not ConfigMixin:
            assert loaded_cls is cls, f"The repo loaded is not the same as the \
                class {cls.__name__}, it is {loaded_cls.__class__.__name__}"

    @classmethod
    def _set_subfolder_save(cls, config, func):
        subfolder_keys = _subfolder_args_list(func)
        for k in subfolder_keys:
            if k in config:
                if config[k] is not None:
                    if not isinstance(config[k], CONFIG_TYPE):
                        raise TypeError(f'{k} must be a ConfigMixin')
                    config[k].__subfolder_save__ = True
        return config

    @classmethod
    def _save_name(cls):
        save_name = cls.__save_name__ or 'config.json'
        if not save_name.endswith('.json'):
            save_name += '.json'
        return save_name


ITERALE_TYPE = (list, tuple, set, dict)
PRIMITIVE_TYPE = (str, int, float, bool, type(None))
CONFIG_TYPE = (ConfigMixin,)

VALID_TYPE = ITERALE_TYPE + PRIMITIVE_TYPE + CONFIG_TYPE
