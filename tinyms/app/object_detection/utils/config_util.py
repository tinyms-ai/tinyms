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
import json

from tinyms.hub.utils.download import download_file_from_url

__all__ = ['load_and parse_config']


def _download_ckeckpoint(checkpoint_url, sha256, checkpoint_path):
    if not checkpoint_url:
        err_msg = 'When set download_from_hub to true, the checkpoint_url can not be empty.'
        raise ValueError(err_msg)

    if not checkpoint_path:
        err_msg = 'When set download_from_hub to true, the checkpoint_path can not be empty.'
        raise ValueError(err_msg)

    if not sha256:
        err_msg = 'When set download_from_hub to true, the sha256 can not be empty.'
        raise ValueError(err_msg)

    download_file_from_url(checkpoint_url, hash_sha256=sha256, save_path=checkpoint_path)


def load_and_parse_config(config_path):
    r"""
    Load and parse the json file the object detection model.

    Args:
        config_path (numpy.ndarray): the config json file path.

    Returns:
        dict, the model configuration.
    """
    # Check if config_path existed
    if not os.path.exists(config_path):
        raise FileNotFoundError("The config file path {} does not exist!".format(config_path))

    with open(config_path, 'r') as f:
        configs = json.load(f)
        if configs.get('download_from_hub'):
            _download_ckeckpoint(configs.get('checkpoint_url'),
                                 configs.get('sha256'),
                                 configs.get('checkpoint_path'))
    return configs
