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

import os
import tempfile
from tinyms.utils.train import load_checkpoint as _load_checkpoint, \
    load_param_into_net, save_checkpoint as _save_checkpoint

from .hubconf import MODEL_HUB
from .utils.download import download_file_from_url
from .utils.check import HubAssetInfo


def _get_hub_root():
    """
    Get the absolute path of hub package.

    Returns:
        str, return a string of path.
    """
    return os.path.abspath(os.path.dirname(__file__))


def _get_model_asset_path(uid_info):
    """Get model asset yaml name from UidInfo."""
    return os.path.join(_get_hub_root(), 'assets', uid_info.provider, uid_info.version,
                        uid_info.model_info + '.yaml')


def _load_weights(path):
    """
    Load network weights from local asset yaml file.

    This function is used to check whether the network can be loaded normally.

    Args:
        path (str): The path of asset file.
    """
    if not isinstance(path, str):
        raise TypeError(f'`path` must be a string, but got {type(path)}')
    if not os.path.exists(path):
        raise ValueError('Please make sure input is a path.')

    try:
        asset_info = HubAssetInfo(path)
        with tempfile.TemporaryDirectory() as target_path:
            download_url = asset_info.asset['asset-link']
            asset_sha256 = asset_info.asset["asset-sha256"]
            ckpt_path = download_file_from_url(download_url, asset_sha256, target_path)
            return _load_checkpoint(ckpt_path)
    except Exception as e:
        raise Exception(e)


def load(uid, pretrained=True, **kwargs):
    '''
    Load a model from remote TinyMS Hub.

    Args:
        uid (str): Uid. The format should be strictly consistent with
            the official example: `tinyms/0.2/lenet5_v1_mnist`.
        pretrained (bool): Specified if to load pretrained weight ckpt file. Default: True.
        kwargs (dict, optional): Keyword arguments for network initialization.

    Return:
        layers.Layer, the initialized network instance.

    Examples:
        >>> from tinyms import hub
        >>>
        >>> hub.load('tinyms/0.2/lenet5_v1_mnist', class_num=10)
    '''
    uid_info = UidInfo(uid)
    model_info = uid_info.model_name + '_' + uid_info.model_version
    net_func = MODEL_HUB.get(model_info)
    if net_func is None:
        raise ValueError("Currently model_name only supports " + str(list(MODEL_HUB.keys())) + "!")

    net = net_func(**kwargs)
    if pretrained is True:
        asset_path = _get_model_asset_path(uid_info)
        ckpt_params = _load_weights(asset_path)
        load_param_into_net(net, ckpt_params)

    return net


def load_checkpoint(uid, dst, pretrained=True, **kwargs):
    '''
    Load model checkpoint file from remote TinyMS Hub.

    Args:
        uid (str): Uid. The format should be strictly consistent with
            the official example: `tinyms/0.2/lenet5_v1_mnist`.
        dst (str): Full path of filename where the checkpoint file
            will be loaded, e.g. `/tmp/lenet5.ckpt`.
        pretrained (bool): Specified if to load pretrained weight ckpt file. Default: True.
        kwargs (dict, optional): Keyword arguments for network initialization.

    Examples:
        >>> from tinyms import hub
        >>>
        >>> hub.load_checkpoint('tinyms/0.2/lenet5_v1_mnist', '/tmp/lenet5.ckpt', class_num=10)
    '''
    uid_info = UidInfo(uid)
    model_info = uid_info.model_name + '_' + uid_info.model_version
    net_func = MODEL_HUB.get(model_info)
    if net_func is None:
        raise ValueError("Currently model_name only supports " + str(list(MODEL_HUB.keys())) + "!")

    net = net_func(**kwargs)
    if pretrained is True:
        asset_path = _get_model_asset_path(uid_info)
        ckpt_params = _load_weights(asset_path)
        load_param_into_net(net, ckpt_params)
    _save_checkpoint(net, dst)


def load_weights(uid):
    '''
    Load model pretrained weights from remote TinyMS Hub.

    Args:
        uid (str): Uid. The format should be strictly consistent with
            the official example: `tinyms/0.2/lenet5_v1_mnist`.

    Return:
        Dict, the pretrained network weight dict.

    Examples:
        >>> from tinyms import hub
        >>> from tinyms.model import lenet5
        >>> from tinyms.utils.train import load_param_into_net
        >>>
        >>> param_dict = hub.load_weights('tinyms/0.2/lenet5_v1_mnist')
        >>> net = lenet5()
        >>> load_param_into_net(net, param_dict)
    '''
    uid_info = UidInfo(uid)
    asset_path = _get_model_asset_path(uid_info)
    return _load_weights(asset_path)


class UidInfo:
    def __init__(self, uid):
        uid_slice = uid.split('/')
        if len(uid_slice) != 3:
            raise ValueError('The format of uid is not correct! \
                An example for the format is: tinyms/0.2/lenet5_v1_mnist')
        self.provider = uid_slice[0]
        self.version = uid_slice[1]
        self.model_info = uid_slice[2]

        model_slice = self.model_info.split('_')
        if len(uid_slice) != 3:
            raise ValueError('The model format of uid is not correct! \
                An example for the format is: tinyms/0.2/lenet5_v1_mnist')
        self.model_name = model_slice[0]
        self.model_version = model_slice[1]
        self.train_dataset = model_slice[2]

    def __str__(self):
        model_info = '_'.join([self.model_name, self.model_version, self.train_dataset])
        uid_info = '/'.join([self.provider, self.version, model_info])
        return uid_info
