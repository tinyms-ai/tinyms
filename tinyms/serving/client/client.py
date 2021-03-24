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
"""The client for TinyMS serving """
import os
import json
import sys
import socket
import requests
import numpy as np
from PIL import Image
from tinyms.vision import mnist_transform, cifar10_transform, imagefolder_transform, voc_transform, cyclegan_transform
from tinyms.data.utils import load_resized_img

transform_checker = {
    'mnist': mnist_transform,
    'cifar10': cifar10_transform,
    'imagenet2012': imagefolder_transform,
    'voc': voc_transform,
    'cityscape': cyclegan_transform,
}


def server_started(host='127.0.0.1', port=5000):
    """
    Detect whether the serving server is started or not.

    A bool value of True will be returned if the server is started, else False.

    Args:
        host (str): the ip address of the server, default is `127.0.0.1`
        port (int): the port address of the server, default is `5000`

    Returns:
        A bool value of True(if server started) or False(if server not started).

    Examples:
        >>> # Running the quickstart tutorial, after starting the server
        >>> if server_started() is True:
        >>>     print(predict(image_path, 'lenet5', 'mnist', strategy))
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((host, port))
        s.shutdown(2)
        return True
    except:
        return False


def list_servables():
    """
    List the model that is currently served by the backend server.

    A `GET` request will be sent to the server(127.0.0.1:5000) which will then 
    be routed to 127.0.0.1:5000/servables, and the backend servalbe information 
    will be returned to the client.

    Returns:
        res_body['servables'] (str) will be returned, the backend servable information.
        Error message will be returned and printed if requests.status_code is not ok.
        'Server not started' will be returned if server is not started

    Examples:
        >>> # Running the quickstart tutorial, after server started and servable json defined
        >>> list_servables()
        [{'description': 'This servable hosts a lenet5 model predicting numbers', 'model': {'class_num': 10, 'format': 'ckpt', 'name': 'lenet5'}, 'name': 'lenet5'}]
    """

    headers = {'Content-Type': 'application/json'}
    url = "http://127.0.0.1:5000/servables"
    if server_started() is True:
        res = requests.get(url=url, headers=headers)
        res_body = res.json()
        if res.status_code != requests.codes.ok:
            print("Request error! Status code: ", res.status_code)
        elif res_body['status'] != 0:
            print(res_body['err_msg'])
        else:
            return res_body['servables']
    else:
        return 'Server not started'


def predict(img_path, servable_name, dataset_name="mnist", strategy="TOP1_CLASS"):
    """
    Send the predict request to the backend server, get the return value and do the post process

    Predict the input image, and get the result. User must specify the image_path, servable_name, dataset_name and output_strategy to get the predict result.

    Args:
        img_path (str): path to the image
        servable_name (str): the `name` in `servable_json`, now supports 6 servables: `lenet5`, `resnet50_imagenet2012`, `resnet50_cifar10`, `mobilenetv2`, `ssd300` and `cyclegan_cityscape`.
        dataset_name (str): the name of the dataset that is used to train the model, now supports 5 datasets: `mnist`, `imagenet2012`, `cifar10`, `voc`, `cityscape`
        strategy (str): the output strategy, for lenet5, resnet50 and mobilenetv2, select between 'TOP1_CLASS' and 'TOP5_CLASS', for ssd300, only `TOP1_CLASS`, for cyclegan_cityscape, select between `gray2color` and `color2gray`

    Returns:
        For lenet5, resnet50, mobilenetv2, the output is a string of predict result.
        For ssd300, the output is a string of bounding boxes coordinates and labels, which can be further processed using `ImageViewer` function
        For cyclegan, the output is a numpy of image, which can be transformed to image using `Image.fromarray`

    Examples:
        >>> # Running the quickstart tutorial, after server started and servable json defined
        >>> print(predict('/root/7.png', 'lenet5', 'mnist', 'TOP1_CLASS'))
        TOP1: 7, score: 0.99943381547927856445
    """
    # Check if args are valid
    if not os.path.isfile(img_path):
        print("The image path {} not exist!".format(img_path))
        sys.exit(0)
    trans_func = transform_checker.get(dataset_name)
    if trans_func is None:
        print("Currently dataset_name only supports {}!".format(list(transform_checker.keys())))
        sys.exit(0)
    if strategy not in ("TOP1_CLASS", "TOP5_CLASS", "gray2color", "color2gray"):
        print("Currently strategy only supports `TOP1_CLASS`, `TOP5_CLASS`, `gray2color` and`color2gray`!")
        sys.exit(0)

    # Perform the transform operation for the input image
    if servable_name == 'cyclegan_cityscape':
        img = np.array(load_resized_img(img_path))
    else:
        img = Image.open(img_path)
    img_data = trans_func(img)

    # Construct the request payload
    payload = {
        'instance': {
            'shape': list(img_data.shape),
            'dtype': img_data.dtype.name,
            'data': json.dumps(img_data.tolist())
        },
        'servable_name': servable_name,
        'strategy': strategy
    }
    headers = {'Content-Type': 'application/json'}
    url = "http://127.0.0.1:5000/predict"
    res = requests.post(url=url, headers=headers, data=json.dumps(payload))
    res.content.decode("utf-8")
    res_body = res.json()
    if res.status_code != requests.codes.ok:
        print("Request error! Status code: ", res.status_code)
        sys.exit(0)
    elif res_body['status'] != 0:
        print(res_body['err_msg'])
        sys.exit(0)
    else:
        instance = res_body['instance']
        res_data = np.array(json.loads(instance['data']))
        if dataset_name == 'voc':
            iw, ih = img.size
            data = trans_func.postprocess(res_data, (ih, iw), strategy)
        elif dataset_name == 'cityscape':
            data = res_data
        else:
            data = trans_func.postprocess(res_data, strategy)
        return data
