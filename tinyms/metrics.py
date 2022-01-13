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

    Computes the Area Under the Curve (AUC) using the trapezoidal rule. This is a general function, given points on a
    curve. For computing the area under the ROC-curve.

    Args:
        x (Union[np.array, list]): From the ROC curve(fpr), np.array with false positive rates. If multiclass,
                                   this is a list of such np.array, one for each class. The shape :math:`(N)`.
        y (Union[np.array, list]): From the ROC curve(tpr), np.array with true positive rates. If multiclass,
                                   this is a list of such np.array, one for each class. The shape :math:`(N)`.
        reorder (boolean): If True, assume that the curve is ascending in the case of ties, as for an ROC curve.
                           If the curve is non-ascending, the result will be wrong. Default: False.

    Returns:
        area (float): Compute result.

    Examples:
        >>> from tinyms.metrics import AUCMetric
        >>>
        >>> metric = AUCMetric()
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
