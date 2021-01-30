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
from flask import request, Flask, jsonify
from ..servable import predict, servable_search

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_server():
    json_data = request.get_json()
    instance = json_data['instance']
    servable_name = json_data['servable_name']

    res = servable_search(servable_name)
    if res['status'] != 0:
        return jsonify(res)
    servable = res['servables'][0]
    res = predict(instance, servable_name, servable['model'])
    return jsonify(res)


@app.route('/servables', methods=['GET'])
def list_servables():
    return jsonify(servable_search())


def start(host='127.0.0.1', port=5000):
    app.run(host=host, port=port)


def shutdown():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()
    return 'Server shutting down...'
