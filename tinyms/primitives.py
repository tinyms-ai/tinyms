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
Primitive.
"""
from mindspore.ops import composite, operations, functional
from mindspore.ops.composite import *
from mindspore.ops.operations import *
from mindspore.ops.functional import *

__all__ = []
__all__.extend(composite.__all__)
__all__.extend(operations.__all__)
__all__.extend(functional.__all__)
