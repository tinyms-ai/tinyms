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


def predict(img_path, servable_path, dataset_name="mnist"):
    with open(servable_path, 'r') as f:
        req = json.load(f)

    img_data = v_preprocess(img_path, dataset_name)
    req['instance'] = {'shape': list(img_data.shape),
                       'data': json.dumps(img_data.tolist())}

    headers = {'Content-Type': 'application/json'}
    url = "http://127.0.0.1:5000/predict"
    res = requests.post(url=url, headers=headers, data=json.dumps(req))
    res_body = res.json()
    if res.status_code != requests.codes.ok:
        print(res_body['err_msg'])
    else:
        print(res_body['instance'])
