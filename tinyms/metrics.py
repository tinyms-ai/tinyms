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
Metrics module provides functions to measure the performance of the machine learning
models on the evaluation dataset. It's used to choose the best model.
"""
from mindspore.nn import metrics
from mindspore.nn.metrics import *
from sklearn.metrics import roc_auc_score

__all__ = ['AUCMetric']
__all__.extend(metrics.__all__)


class AUCMetric(Metric):
    """
    Calculates the auc value.
    Implement auc metric method.

    Note:
        The method `update` must receive input of the form :math:`(y_{pred}, y)`. If some samples have
        the same accuracy, the first sample will be chosen.

    Args:
        k (int): Specifies the top-k categorical accuracy to compute.

    Raises:
        TypeError: If `k` is not int.
        ValueError: If `k` is less than 1.

    Examples:
        >>> x = Tensor(np.array([[0.2, 0.5, 0.3, 0.6, 0.2], [0.1, 0.35, 0.5, 0.2, 0.],
        ...         [0.9, 0.6, 0.2, 0.01, 0.3]]), mindspore.float32)
        >>> y = Tensor(np.array([2, 0, 1]), mindspore.float32)
        >>> topk = nn.TopKCategoricalAccuracy(3)
        >>> topk.clear()
        >>> topk.update(x, y)
        >>> output = topk.eval()
        >>> print(output)
        0.6666666666666666
    """
    def __init__(self):
        super(AUCMetric, self).__init__()
        self.pred_probs = []
        self.true_labels = []

    def clear(self):
        """Clear the internal evaluation result."""
        self.pred_probs = []
        self.true_labels = []

    def update(self, *inputs):
        batch_predict = inputs[1].asnumpy()
        batch_label = inputs[2].asnumpy()
        self.pred_probs.extend(batch_predict.flatten().tolist())
        self.true_labels.extend(batch_label.flatten().tolist())

    def eval(self):
        if len(self.true_labels) != len(self.pred_probs):
            raise RuntimeError('true_labels.size() is not equal to pred_probs.size()')
        auc = roc_auc_score(self.true_labels, self.pred_probs)
        return auc
