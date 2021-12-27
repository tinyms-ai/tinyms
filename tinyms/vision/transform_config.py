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
import yaml

__all__ = ['get_specified_config']


def _get_config_path():
    cur_path = os.path.dirname(__file__)
    return os.path.join(cur_path, "configs")


def _get_all_config(yaml_file):
    if not os.path.exists(yaml_file):
        raise FileNotFoundError('The transform yaml_file config file not found!')

    all_transform_configs = []
    with open(yaml_file, mode='r', encoding='utf-8') as f:
        for transform_config in yaml.safe_load_all(f):
            all_transform_configs.append(transform_config)
    return all_transform_configs


def get_specified_config(transforms_op, yaml_path=None):
    r'''
    Get specified vision transform parameters from transform yaml file.

    Args:
        transforms_op (str): Specified vision transforms class, such as: VOCTransform, MnistTransform.
        yaml_path (str): The yaml file path of the vision transform configuration. Default: None.

    Returns:
        dict, the vision transform parameters of the specified transforms_op.
    '''
    if not yaml_path:
        yaml_path = os.path.join(_get_config_path(), 'transform_config.yaml')
    all_transform_configs = _get_all_config(yaml_path)
    data = None
    for transform_config in all_transform_configs:
        if transforms_op in transform_config:
            data = transform_config[transforms_op]
    return data
