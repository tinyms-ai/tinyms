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
"""Download or extract file."""

import os
import shutil
import hashlib
import errno
import stat
import urllib
from urllib.request import urlretrieve, HTTPError, URLError


REAL_PATH = os.path.split(os.path.realpath(__file__))[0]
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024  # 5GB
SUFFIX_LIST = ['.ckpt', 'mindir', '.onnx']


def handle_remove_read_only(func, path, exc):
    exc_value = exc[1]
    if func in (os.rmdir, os.remove, os.unlink) and exc_value.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU | stat.S_IRWXG | stat.S_IRWXO)  # 0777
        func(path)


def url_exist(url):
    """
    Whether the url exist.
    """
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    try:
        opener.open(url)
        return True
    except HTTPError as e:
        print(e.code)
    except URLError as e:
        print(e.reason)
    return False


def _remove_path_if_exists(path):
    if os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            shutil.rmtree(path, ignore_errors=False, onerror=handle_remove_read_only)


def _create_path_if_not_exists(path):
    if not os.path.exists(path):
        if os.path.isfile(path):
            os.remove(path)
        else:
            os.mkdir(path)


def download_file_from_url(url, hash_sha256=None, save_path='.'):
    """
    download checkpoint weight from giving url.

    Args:
       url(string): checkpoint url path.
       hash_sha256(string): checkpoint file sha256.
       save_path(string): checkpoint download save path.

    Returns:
       string.
    """

    def reporthook(a, b, c):
        percent = a * b * 100.0 / c
        percent = 100 if percent > 100 else percent
        if c > 0:
            print("\rDownloading...%5.1f%%" % percent, end="")

    def sha256sum(file_name, hash_sha256):
        fp = open(file_name, 'rb')
        content = fp.read()
        fp.close()
        m = hashlib.sha256()
        m.update(content)
        download_sha256 = m.hexdigest()
        return download_sha256 == hash_sha256

    _create_path_if_not_exists(os.path.realpath(save_path))
    ckpt_name = os.path.basename(url.split("/")[-1])
    # identify file exist or not
    file_path = os.path.join(save_path, ckpt_name)
    if os.path.isfile(file_path):
        if hash_sha256 and sha256sum(file_path, hash_sha256):
            print('File already exists!')
            return file_path
        print('File already exists, but sha256 checking failed. Will download again')

    _remove_path_if_exists(file_path)

    # download the checkpoint file
    print('Downloading data from url {}'.format(url))
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
        urllib.request.install_opener(opener)
        urlretrieve(url, file_path, reporthook=reporthook)
    except HTTPError as e:
        raise Exception(e.code, e.msg, url)
    except URLError as e:
        raise Exception(e.errno, e.reason, url)
    print('\nDownload finished!')

    # Check file integrity
    if hash_sha256:
        result = sha256sum(file_path, hash_sha256)
        if not result:
            raise Exception('INTEGRITY ERROR: File: {} is not integral'.format(file_path))

    # Check if file size over MAX_FILE_SIZE
    file_size = os.path.getsize(file_path)
    if file_size > MAX_FILE_SIZE:
        os.remove(file_path)
        raise Exception('SIZE ERROR: Download file is too large,'
                        'the max size is {}Mb'.format(MAX_FILE_SIZE / 1024 / 1024))

    # Check file type
    suffix = os.path.splitext(file_path)[1]
    if suffix not in SUFFIX_LIST:
        os.remove(file_path)
        raise Exception('SUFFIX ERROR: File: {} with suffix: {} '
                        'can not be recognized'.format(file_path, suffix))
    return file_path
