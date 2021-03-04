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
import requests
import numpy as np
from PIL import Image
from tinyms.vision import mnist_transform, cifar10_transform, \
    imagefolder_transform, voc_transform

transform_checker = {
    'mnist': mnist_transform,
    'cifar10': cifar10_transform,
    'imagenet2012': imagefolder_transform,
    'voc': voc_transform,
}


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


def predict(img_path, servable_name, dataset_name="mnist", strategy="TOP1_CLASS"):
    # Check if args are valid
    if not os.path.isfile(img_path):
        print("The image path {} not exist!".format(img_path))
        sys.exit(0)
    trans_func = transform_checker.get(dataset_name)
    if trans_func is None:
        print("Currently dataset_name only supports {}!".format(list(transform_checker.keys())))
        sys.exit(0)
    if strategy not in ("TOP1_CLASS", "TOP5_CLASS"):
        print("Currently strategy only supports `TOP1_CLASS` and `TOP5_CLASS`!")
        sys.exit(0)

    # Perform the tranform operation for the input image
    img_data = trans_func(Image.open(img_path))
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
        data = trans_func.postprocess(np.array(json.loads(instance['data'])), strategy)
        print(data)
