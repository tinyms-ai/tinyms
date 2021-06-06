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

from easydict import EasyDict as ed

from tinyms import model

MODEL_HUB = ed({
    "alexnet_v1": model.alexnet,
    "lenet5_v1": model.lenet5,
    "resnet50_v1": model.resnet50,
    "mobilenet_v2": model.mobilenetv2,
    "ssd300_v1": model.ssd300_mobilenetv2,
    "vgg16_v1": model.vgg16,
})
