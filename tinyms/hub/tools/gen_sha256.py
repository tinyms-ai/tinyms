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
"""Tool for generate sha256 for given file path."""

import argparse
import hashlib
import os


def parse_args():
    args_parser = argparse.ArgumentParser("Generate sha256 from specific file")
    args_parser.add_argument("--input_file", type=str, required=True, help="which file to calculate the sha256")
    return args_parser.parse_args()


def gen_sha256(path):
    if not os.path.exists(path):
        print(f"File {path} not exists")
        return

    with open(path, 'rb') as fr:
        content = fr.read()
        m = hashlib.sha256()
        m.update(content)
        print(f"sha256: {m.hexdigest()}")


if __name__ == "__main__":
    args = parse_args()
    gen_sha256(args.file)
