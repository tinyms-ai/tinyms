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
import sys
import json
import signal
import socket
import logging
import platform
import subprocess

from flask import request, Flask, jsonify
from ..servable import predict, servable_search

app = Flask(__name__)


@app.route('/servables', methods=['GET'])
def list_servables():
    """
    Handle the list_servables request sent by client, call the servable search in tinyms.serving.servable

    Whether the server started or not will be detected first.

    Returns:
        A json object of servable information in the backend will be sent back to the client.

    Examples:
        >>> # In the client part, the servable search request will be routed and processed here
        >>> res = requests.get(url=url, headers=headers)
        >>> res_body = res.json()
    """

    return jsonify(servable_search())


@app.route('/servables/<name>', methods=['POST'])
def predict_server(name):
    """
    Handle the request sent by client, call the servable_search, predict function
    in tinyms.serving.servable and return the json result to the client.

    Whether the server started or not will be detected first.

    Args:
        name (str): Specifies the servable name by client.

    Returns:
        A json object of predicted result will be sent back to the client.
    """

    json_data = request.get_json()
    instance = json_data['instance']
    strategy = json_data['strategy']

    res = servable_search(name)
    if res['status'] != 0:
        return jsonify(res)
    servable = res['servables'][0]
    res = predict(instance, name, servable['model'], strategy)
    return jsonify(res)


class _FlaskServer(object):
    """
    Create a flask service, only be used to trigger starting the flask server
    in subprocess.
    """

    def __init__(self, host='127.0.0.1', port=5000):
        self.system_name = platform.system().lower()
        self.host = host
        self.port = port

    def signal_handler(self, signal, frame):
        self.shutdown()
        sys.exit(0)

    def shutdown(self):
        """
        Shutdown the flask server.
        """

        if self.system_name == "windows":
            server_res = subprocess.getoutput("netstat -ano | findstr 5000")
            server_pid = None
            for line in server_res.split("\n"):
                temp = [i for i in line.split(' ') if i != '']
                if len(temp) > 4:
                    if temp[1] == f'{self.host}:{self.port}' and temp[3] == 'LISTENING':
                        server_pid = temp[4]
                        continue
            if server_pid:
                os.system("taskkill /t /f /pid %s" % server_pid)
        else:
            server_pid = subprocess.getoutput("netstat -anp | grep %s | awk '{printf $7}' | cut -d/ -f1" % self.port)
            subprocess.run("kill -9 " + str(server_pid), shell=True)
        return 'Server shutting down...'

    def windows_run_server(self):
        cmd = 'python -c "from tinyms.serving import run_flask; run_flask()"'
        server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

        for sig in [signal.SIGINT, signal.SIGTERM]:
            signal.signal(sig, self.signal_handler)

    def linux_run_server(self):
        cmd = [f'python -c "from tinyms.serving import run_flask; run_flask()"']
        server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)

        for sig in [signal.SIGINT, signal.SIGHUP, signal.SIGTERM]:
            signal.signal(sig, self.signal_handler)

    def run(self):
        if self.system_name == "windows":
            self.windows_run_server()
        else:
            self.linux_run_server()


def run_flask(host='127.0.0.1', port=5000):
    """
    Start the flask server, only be used to trigger starting the flask server in subprocess.

    Directly calling this function is not recommended, please use start_server(). Only Error message will be displayed.

    Args:
        host (str): the ip address of the flask server. Default: '127.0.0.1'.
        port (int): the port of the server. Default: 5000.

    Returns:
        Server Started

    Examples:
        >>> # In the start_server function
        >>> cmd = ['python -c "from tinyms.serving import run_flask; run_flask()"']
        >>> server_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    """

    if os.path.exists('temp.json'):
        with open('temp.json', 'r') as f:
            config_data = json.load(f)
            host = config_data.get('host')
            port = config_data.get('port')

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    app.run(host=host, port=port)


class Server:
    '''
    Server is the entrance to initialize and shutdown the serving server, and
    also accept all requests from the client side by calling Flask server.

    Args:
        host (str): Serving server host ip. Default: '127.0.0.1'.
        port (int): Serving server listen port. Default: 5000.
        serving_path (str, optional): Set the read path of a service configuration.
            Default: '/etc/tinyms/serving/'.

    Examples:
        >>> from tinyms.serving import Server
        >>>
        >>> server = Server()
    '''

    def __init__(self, host='127.0.0.1', port=5000, serving_path='/etc/tinyms/serving/'):
        json_data = {}
        json_data.update({
            'host': host,
            'port': port,
            'serving_path': serving_path
        })
        with open('temp.json', 'w') as f:
            json.dump(json_data, f)

        self.host = host
        self.port = port

    def _check_started(self):
        """
        Detect whether the serving server is started or not.

        A bool value of True will be returned if the server is started, else False.

        Returns:
            A bool value of True (if server started) or False (if server not started).
        """
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.connect((self.host, self.port))
            s.shutdown(2)
            return True
        except:
            return False

    def start_server(self):
        """
        Start the flask server in a subprocess.

        Catch the signal of CTRL + D to shutdown, otherwise call shutdown() function
        to shutdown the server, if the ip and port already in use, server won't start
        for a second time.

        Returns:
            Start the server in a sub process.

        Examples:
            >>> from tinyms.serving import Server
            >>>
            >>> server = Server()
            >>> server.start_server()
            Server starts at host 127.0.0.1, port 5000
        """

        if self._check_started() is True:
            print('Server already started at host %s, port %d' % (self.host, self.port))
        else:
            # TODO: Add dynamic host ip and port support
            _FlaskServer(self.host, self.port).run()
            print('Server started at host %s, port %d' % (self.host, self.port))

    def shutdown(self):
        """
        Shutdown the flask server.

        Search fot the pid of the process running on port 5000, and kill it. This function
        will be automatically called when SIGINT, SIGHUP and SIGTERM signals caught.

        Returns:
            A string message of server shutting down or not.

        Examples:
            >>> from tinyms.serving import Server
            >>>
            >>> server = Server()
            >>> server.shutdown()
            Server shutting down...
        """
        if not self._check_started() is True:
            print('Server already shutdown at host %s, port %d' % (self.host, self.port))
        else:
            if os.path.exists('temp.json'):
                os.remove('temp.json')
            _FlaskServer(self.host, self.port).shutdown()
