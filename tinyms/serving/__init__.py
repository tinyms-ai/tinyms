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
"""
.. TinyMS Serving package.

This module refers to the process of serving pre-trained models so that
they can quickly and efficiently process data input by users and obtain
results. TinyMS provides a complete set of start server (`start_server`),
check backend (`list_servables`), check start status (`server_started`)
and shut down the server (`shutdown`) and other functions based on
`Flask` (https://flask.palletsprojects.com/en/1.1.x/).

Examples:
    >>> from tinyms.serving import Server, Client
    >>>
    >>> server = Server()
    >>> server.start_server()
    Server starts at host 127.0.0.1, port 5000
    >>> client = Client()
    >>> client.list_servables()
    >>> client.predict('example.jpg', 'servable_name', dataset_name='mnist')
"""
from . import client, server
from .client import *
from .server import *

__all__ = []
__all__.extend(client.__all__)
__all__.extend(server.__all__)
