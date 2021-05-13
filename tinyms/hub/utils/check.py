# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
"""URL check list"""
import numbers
import socket
import ssl
import yaml
from urllib.parse import urlparse
from urllib.request import Request, urlopen, HTTPError

_URL_CHECK_LIST = ('https://tinyms-hub.obs.cn-north-4.myhuaweicloud.com',
                   'https://github.com/tinyms-ai/tinyms')
_URL_CHECK_LIST_INFO = None


class UrlInfo:
    def __init__(self, url):
        parsed = urlparse(url)
        self.domain = parsed.netloc
        self.ip = socket.gethostbyname(parsed.netloc)
        self.path = parsed.path


def _get_checklist_info():
    global _URL_CHECK_LIST_INFO
    if _URL_CHECK_LIST_INFO is None:
        _URL_CHECK_LIST_INFO = set()
        for url in _URL_CHECK_LIST:
            info = UrlInfo(url)
            _URL_CHECK_LIST_INFO.add(info)
    return _URL_CHECK_LIST_INFO


def verify_url(url):
    """
    Verify that the URL is in the url checklist.

    Args:
        url (str): A string of url.

    Returns:
        bool, whether the url in url checklist.
    """
    target_info = UrlInfo(url)
    checklist_info = _get_checklist_info()
    for info in checklist_info:
        if (target_info.ip == info.ip or target_info.domain == info.domain) and \
                target_info.path.startswith(info.path):
            return True
    return False


class HubAssetInfo:
    """
    Information of hub asset model.
    """

    def __init__(self, asset_path):
        asset_dict = ValidHubAsset(asset_path).validate_asset()
        self.name = asset_dict.get('model-name')
        self.backbone_name = asset_dict.get('backbone-name')
        self.type = asset_dict.get('module-type')
        self.fine_tunable = asset_dict.get('fine-tunable')
        self.input_shape = asset_dict.get('input-shape')
        self.author = asset_dict.get('author')
        self.update_time = asset_dict.get('update-time')
        self.repo_link = asset_dict.get('repo-link')
        self.user_id = asset_dict.get('user-id')
        self.backend = asset_dict.get('infer-backend')
        self.dataset = asset_dict.get('train-dataset')
        self.license = asset_dict.get('license')
        self.accuracy = asset_dict.get('accuracy')
        self.used_for = asset_dict.get('used-for')
        self.model_version = asset_dict.get('model-version')
        self.tinyms_version = asset_dict.get('tinyms-version')
        self.asset = asset_dict.get('asset')


class ValidHubAsset:
    r"""
    Check Hub Asset files and extract info.
    """

    def __init__(self, asset_file):
        self.asset_file = asset_file
        self.required_user_fields = ['backbone-name', 'module-type', 'fine-tunable', 'input-shape', 'model-version',
                                     'train-dataset', 'author', 'update-time', 'user-id', 'used-for', 'infer-backend',
                                     'tinyms-version', 'license', 'summary']
        self.optional_backend_field = 'train-backend'
        self.optional_accuracy_field = 'accuracy'

        self.valid_module_type = ['audio', 'cv', 'nlp', 'recommend', 'other']
        self.valid_train_dataset = ['mnist', 'cifar10', 'cifar100', 'mushroom', 'voc2007']
        self.valid_file_format = ['ckpt', 'mindir', 'onnx']
        self.valid_used_for = ['inference', 'transfer-learning']
        self.valid_backend = ['cpu', 'gpu']

        # The current sections list below will be required.
        # If necessary, please add it, like 'Model Description'.
        self.required_sections = []

    def _validate_asset_link_field(self, link):
        r"""
        Make sure the github or gitee repo exists
        """
        if link is None:
            return
        link = link.strip('<>')
        if not verify_url(link):
            raise ValueError('url: ``{}`` is not trust in {}'.format(link, self.asset_file))

        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            req = Request(link)
            urlopen(req, context=ctx)
        except HTTPError:
            raise ValueError('``{}`` is not valid url in {}'.format(link, self.asset_file))

    def _validate_file_format_field(self, file_format):
        r"""
        Only allow file_format in predefined set
        """
        if file_format.lower() not in self.valid_file_format:
            raise ValueError('file_format ``{}`` is not valid in {}. Choose from {}'
                             .format(file_format, self.asset_file, self.valid_file_format))

    def _validate_asset_field(self, asset):
        require_keys = ["asset-link", "asset-sha256", "file-format"]
        for k in require_keys:
            if k not in asset:
                raise ValueError('field: ``{}`` is required in {}, but not found.'
                                 .format(k, self.asset_file))
        self._validate_asset_link_field(asset['asset-link'])
        self._validate_file_format_field(asset['file-format'])

    def _validate_train_dataset_field(self, train_dataset):
        r"""
        Only allow train_dataset in predefined set
        """
        if train_dataset not in self.valid_train_dataset:
            raise ValueError('train_dataset ``{}`` is not valid in {}. Choose from {}'
                             .format(train_dataset, self.asset_file, self.valid_train_dataset))

    def _validate_used_for_field(self, used_for):
        r"""
        Only allow used_for in predefined set
        """
        used_for_list = used_for.split('/')
        for item in used_for_list:
            if item.lower() not in self.valid_used_for:
                raise ValueError('used_for ``{}`` is not valid in {}. Choose from {}'
                                 .format(item, self.asset_file, self.valid_used_for))

    def _validate_backend_field(self, backend):
        r"""
        Only allow backend in predefined set
        """
        if not isinstance(backend, list):
            backend = [backend]
        for bk in backend:
            if bk.lower() not in self.valid_backend:
                raise ValueError('backend ``{}`` is not valid in {}. Choose from {}'
                                 .format(bk, self.asset_file, self.valid_backend))

    def _validate_module_type_field(self, module_type):
        r"""
        Only allow module_type in predefined set
        """
        items = module_type.lower().split('-')
        if len(items) > 2:
            raise Exception("module-type could only no more than one '-' ")
        first_class = items[0]
        if first_class not in self.valid_module_type:
            raise ValueError('module_type ``{}`` is not valid in {}. Valid module_type set is {}'
                             .format(module_type, self.asset_file, self.valid_module_type))

    def _validate_header(self, header):
        r"""
        Make sure the header is in the required format
        """
        for field in self.required_user_fields:
            if field not in header:
                raise ValueError("field: ``{}`` is required in {}, but not found."
                                 .format(field, self.asset_file))

        if not isinstance(header['fine-tunable'], bool):
            raise TypeError("`fine-tunable` must be `bool`, but got {}".format(header['fine-tunable']))

        if not isinstance(header['input-shape'], list):
            raise TypeError("`input-shape` must be `list` of `int`, but got {}".format(header['input-shape']))
        for i in header['input-shape']:
            if not isinstance(i, int) and not isinstance(i, list):
                raise TypeError("`input-shape` must be `list` of `int`, but got {}".format(header['input-shape']))

        if not isinstance(header['infer-backend'], list):
            raise TypeError("`infer-backend` must be `list` of `str`, but got {}".format(header['infer-backend']))
        for i in header['infer-backend']:
            if not isinstance(i, str) and not isinstance(i, list):
                raise TypeError("`infer-backend` must be `list` of `str`, but got {}".format(header['infer-backend']))

        self._validate_train_dataset_field(header['train-dataset'])
        self._validate_used_for_field(header['used-for'])
        self._validate_backend_field(header['infer-backend'])
        self._validate_module_type_field(header['module-type'])
        if not header.get('asset') is None:
            self._validate_asset_field(header['asset'])

        if self.optional_accuracy_field in header.keys():
            if not isinstance(header[self.optional_accuracy_field], numbers.Number):
                raise TypeError("`accuracy` must be `number`, but got {}".
                                format(header[self.optional_accuracy_field]))
        if self.optional_backend_field in header.keys():
            self._validate_backend_field(header[self.optional_backend_field])

    def validate_asset(self):
        with open(self.asset_file) as f:
            asset_dict = yaml.load(f, Loader=yaml.FullLoader)
        if not asset_dict:
            raise TypeError("Failed to parse a valid yaml header")

        self._validate_header(asset_dict)
        return asset_dict
