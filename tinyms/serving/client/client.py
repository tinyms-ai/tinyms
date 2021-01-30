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
import json
import requests

from .vision import preprocess as v_preprocess


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
    img_data = v_preprocess(img_path, dataset_name)

    # Construct the request payload
    payload = {
        'instance': {
            'shape': list(img_data.shape),
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
        print(res_body['instance'])
