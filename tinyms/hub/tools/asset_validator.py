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
"""Tool for validating the model asset yaml file."""

import argparse
import os

from tinyms.hub.utils.check import ValidHubAsset


def parse_args():
    args_parser = argparse.ArgumentParser("Validate the model asset from specific file")
    args_parser.add_argument("--input_file", type=str, required=True, help="which file to verify")
    return args_parser.parse_args()


def validate_asset(path):
    if not os.path.exists(path):
        print(f"File {path} not exists")
        return

    if ValidHubAsset(path).validate_asset():
        print(f"Asset file {path} validation passed")
    return


if __name__ == "__main__":
    args = parse_args()
    validate_asset(args.input_file)
