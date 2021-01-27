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
import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.train.serialization import load_checkpoint

from .model import lenet5, resnet50


def predict(instance, name="lenet5", model_format="ckpt", class_num=10):
    # check if servable name is valid
    if name not in ("lenet5", "resnet50"):
        err_msg = "Currently model_name only supports `lenet5` and `resnet50`!"
        return {"status": 1, "err_msg": err_msg}
    input = np.array(json.loads(instance['data']), dtype='uint8')
    net = lenet5(class_num=class_num) if name == "lenet5" else resnet50(class_num=class_num)
    input = input.reshape((1, 1, 28, 28)) if name == "lenet5" else input.reshape((1, 3, 224, 224))

    # check if model_format is valid
    if model_format not in ("ckpt"):
        err_msg = "Currently model_format only supports `ckpt`!"
        return {"status": 1, "err_msg": err_msg}
    # load checkpoint
    ckpt_path = os.path.join("ckpt", name+"."+model_format)
    if not os.path.isfile(ckpt_path):
        err_msg = "The model path "+ckpt_path+" not exist!"
        return {"status": 1, "err_msg": err_msg}
    load_checkpoint(ckpt_path, net=net)

    # execute the network to perform model prediction
    data = net(Tensor(input, mindspore.float32)).asnumpy()
    return {"status": 0, "instance": {"shape": data.shape, "data": json.dumps(data.tolist())}}
