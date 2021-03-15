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
import subprocess
import signal
import sys
import logging

from flask import request, Flask, jsonify
from ..servable import predict, servable_search
from ..client import server_started

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_server():
    if server_started() is True:
        json_data = request.get_json()
        instance = json_data['instance']
        servable_name = json_data['servable_name']
        strategy = json_data['strategy']

        res = servable_search(servable_name)
        if res['status'] != 0:
            return jsonify(res)
        servable = res['servables'][0]
        res = predict(instance, servable_name, servable['model'], strategy)
        return jsonify(res)
    else:
        return 'No server detected'


@app.route('/servables', methods=['GET'])
def list_servables():
    if server_started() is True:
        return jsonify(servable_search())
    else:
        return 'No server detected'


def run_flask(host='127.0.0.1', port=5000):
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port)


def start_server(host='127.0.0.1', port=5000):
    if server_started() is True:
        print('Server already started at host %s, port %d'%(host, port))
    else:
        cmd = ['python -c "from tinyms.serving import run_flask; run_flask()"']
        server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
        print('Server starts at host %s, port %d' %(host, port))

    def signal_handler(signal, frame):
        shutdown()    
        sys.exit(0)
    
    for sig in [signal.SIGINT, signal.SIGHUP, signal.SIGTERM]:
        signal.signal(sig, signal_handler)
   

def shutdown():
    if server_started() is True:
        server_pid = subprocess.getoutput("netstat -anp | grep 5000 | awk '{printf $7}' | cut -d/ -f1")
        subprocess.run("kill -9 " + str(server_pid), shell=True)
        return 'Server shutting down...'
    else:
        return 'No server detected'
