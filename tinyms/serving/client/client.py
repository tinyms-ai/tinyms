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
import sys
import cv2
import requests
import numpy as np
from PIL import Image
from tinyms.vision import mnist_transform, cifar10_transform, imagefolder_transform


def list_servables():
    headers = {'Content-Type': 'application/json'}
    url = "http://127.0.0.1:5000/servables"
    res = requests.get(url=url, headers=headers)
    res_body = res.json()
    if res.status_code != requests.codes.ok:
        print("Request error! Status code: ", res.status_code)
    elif res_body['status'] != 0:
        print(res_body['err_msg'])
    else:
        print(res_body['servables'])


def predict(img_path, servable_name, dataset_name="mnist"):
    # TODO: The preprocess would be moved to data module later
    # check if dataset_name and img_path are valid
    if dataset_name not in ("mnist", "cifar10", "imagenet2012"):
        print("Currently dataset_name only supports `mnist`, `cifar10` and `imagenet2012`!")
        sys.exit(0)
    if not os.path.isfile(img_path):
        print("The image path "+img_path+" not exist!")
        sys.exit(0)

    if dataset_name == "mnist":
        img_data = np.asarray(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE), dtype=np.float32)
        img_data = mnist_transform(img_data)
    elif dataset_name == "cifar10":
        img_data = np.asarray(Image.open(img_path), dtype=np.float32)
        img_data = cifar10_transform(img_data)
    else:
        img_data = np.asarray(Image.open(img_path), dtype=np.float32)
        img_data = imagefolder_transform(img_data)

    # Construct the request payload
    payload = {
        'instance': {
            'shape': list(img_data.shape),
            'dtype': img_data.dtype.name,
            'data': json.dumps(img_data.tolist())
        },
        'servable_name': servable_name
    }
    headers = {'Content-Type': 'application/json'}
    url = "http://127.0.0.1:5000/predict"
    res = requests.post(url=url, headers=headers, data=json.dumps(payload))
    res_body = res.json()
    if res.status_code != requests.codes.ok:
        print("Request error! Status code: ", res.status_code)
    elif res_body['status'] != 0:
        print(res_body['err_msg'])
    else:
        instance = res_body['instance']
        if dataset_name == "mnist":
            data = mnist_transform.postprocess(np.array(json.loads(instance['data'])), strategy='TOP1_CLASS')
            print("Prediction is: "+str(data))
        elif dataset_name == "imagenet2012":
            data = imagefolder_transform.postprocess(np.array(json.loads(instance['data'])), strategy='TOP1_CLASS')
            print("Prediction is: "+str(data))
        else:
            data = cifar10_transform.postprocess(np.array(json.loads(instance['data'])), strategy='TOP1_CLASS')
            print("Prediction is: "+str(data))
