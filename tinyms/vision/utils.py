# Copyright 2020-2021 Huawei Technologies Co., Ltd
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

"""Bbox utils"""
import json
import math
import itertools as it
import numpy as np
from easydict import EasyDict as ed

ssd300_config = ed({
    "img_shape": [300, 300],
    "num_ssd_boxes": 1917,
    "match_threshold": 0.5,
    "nms_threshold": 0.6,
    "min_score": 0.4,
    "max_boxes": 100,

    # learing rate settings
    "lr_init": 0.001,
    "lr_end_rate": 0.001,
    "warmup_epochs": 2,
    "momentum": 0.9,
    "weight_decay": 1.5e-4,

    # network
    "num_default": [3, 6, 6, 6, 6, 6],
    "extras_in_channels": [256, 576, 1280, 512, 256, 256],
    "extras_out_channels": [576, 1280, 512, 256, 256, 128],
    "extras_strides": [1, 1, 2, 2, 2, 2],
    "extras_ratio": [0.2, 0.2, 0.2, 0.25, 0.5, 0.25],
    "feature_size": [19, 10, 5, 3, 2, 1],
    "min_scale": 0.2,
    "max_scale": 0.95,
    "aspect_ratios": [(), (2, 3), (2, 3), (2, 3), (2, 3), (2, 3)],
    "steps": (16, 32, 64, 100, 150, 300),
    "prior_scaling": (0.1, 0.2),
    "gamma": 2.0,
    "alpha": 0.75,
})


class GenerateDefaultBoxes():
    """
    Generate Default boxes for SSD300, follows the order of (W, H, archor_sizes).
    `self.default_boxes` has a shape of [archor_sizes, H, W, 4], the last dimension is [y, x, h, w].
    `self.default_boxes_tlbr` has a shape as `self.default_boxes`, the last dimension is [y1, x1, y2, x2].
    """

    def __init__(self):
        fk = ssd300_config.img_shape[0] / np.array(ssd300_config.steps)
        scale_rate = (ssd300_config.max_scale - ssd300_config.min_scale) / (len(ssd300_config.num_default) - 1)
        scales = [ssd300_config.min_scale + scale_rate * i for i in range(len(ssd300_config.num_default))] + [1.0]
        self.default_boxes = []
        for idex, feature_size in enumerate(ssd300_config.feature_size):
            sk1 = scales[idex]
            sk2 = scales[idex + 1]
            sk3 = math.sqrt(sk1 * sk2)
            if idex == 0 and not ssd300_config.aspect_ratios[idex]:
                w, h = sk1 * math.sqrt(2), sk1 / math.sqrt(2)
                all_sizes = [(0.1, 0.1), (w, h), (h, w)]
            else:
                all_sizes = [(sk1, sk1)]
                for aspect_ratio in ssd300_config.aspect_ratios[idex]:
                    w, h = sk1 * math.sqrt(aspect_ratio), sk1 / math.sqrt(aspect_ratio)
                    all_sizes.append((w, h))
                    all_sizes.append((h, w))
                all_sizes.append((sk3, sk3))

            assert len(all_sizes) == ssd300_config.num_default[idex]

            for i, j in it.product(range(feature_size), repeat=2):
                for w, h in all_sizes:
                    cx, cy = (j + 0.5) / fk[idex], (i + 0.5) / fk[idex]
                    self.default_boxes.append([cy, cx, h, w])

        def to_tlbr(cy, cx, h, w):
            return cy - h / 2, cx - w / 2, cy + h / 2, cx + w / 2

        # For IoU calculation
        self.default_boxes_tlbr = np.array(tuple(to_tlbr(*i) for i in self.default_boxes), dtype='float32')
        self.default_boxes = np.array(self.default_boxes, dtype='float32')


ssd_default_boxes_tlbr = GenerateDefaultBoxes().default_boxes_tlbr
ssd_default_boxes = GenerateDefaultBoxes().default_boxes


def ssd_bboxes_encode(boxes):
    """
    Labels anchors with ground truth inputs.

    Args:
        boxes: ground truth with shape [N, 5], for each row, it stores
            [ymin, xmin, ymax, xmax, cls].

    Returns:
        gt_loc: location ground truth with shape [num_anchors, 4].
        gt_label: class ground truth with shape [num_anchors, 1].
        num_matched_boxes: number of positives in an image.
    """
    y1, x1, y2, x2 = np.split(ssd_default_boxes_tlbr[:, :4], 4, axis=-1)
    vol_anchors = (x2 - x1) * (y2 - y1)

    def jaccard_with_anchors(bbox):
        """Compute jaccard score a box and the anchors."""
        # Intersection bbox and volume.
        ymin = np.maximum(y1, bbox[0])
        xmin = np.maximum(x1, bbox[1])
        ymax = np.minimum(y2, bbox[2])
        xmax = np.minimum(x2, bbox[3])
        w = np.maximum(xmax - xmin, 0.)
        h = np.maximum(ymax - ymin, 0.)

        # Volumes.
        inter_vol = h * w
        union_vol = vol_anchors + (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) - inter_vol
        jaccard = inter_vol / union_vol
        return np.squeeze(jaccard)

    pre_scores = np.zeros((ssd300_config.num_ssd_boxes), dtype=np.float32)
    t_boxes = np.zeros((ssd300_config.num_ssd_boxes, 4), dtype=np.float32)
    t_label = np.zeros((ssd300_config.num_ssd_boxes), dtype=np.int64)
    for bbox in boxes:
        label = int(bbox[4])
        scores = jaccard_with_anchors(bbox)
        idx = np.argmax(scores)
        scores[idx] = 2.0
        mask = (scores > ssd300_config.match_threshold)
        mask = mask & (scores > pre_scores)
        pre_scores = np.maximum(pre_scores, scores * mask)
        t_label = mask * label + (1 - mask) * t_label
        for i in range(4):
            t_boxes[:, i] = mask * bbox[i] + (1 - mask) * t_boxes[:, i]

    index = np.nonzero(t_label)

    # Transform to tlbr.
    bboxes = np.zeros((ssd300_config.num_ssd_boxes, 4), dtype=np.float32)
    bboxes[:, [0, 1]] = (t_boxes[:, [0, 1]] + t_boxes[:, [2, 3]]) / 2
    bboxes[:, [2, 3]] = t_boxes[:, [2, 3]] - t_boxes[:, [0, 1]]

    # Encode features.
    bboxes_t = bboxes[index]
    default_boxes_t = ssd_default_boxes[index]
    bboxes_t[:, :2] = (bboxes_t[:, :2] - default_boxes_t[:, :2]) / \
        (default_boxes_t[:, 2:] * ssd300_config.prior_scaling[0])
    tmp = np.maximum(bboxes_t[:, 2:4] / default_boxes_t[:, 2:4], 0.000001)
    bboxes_t[:, 2:4] = np.log(tmp) / ssd300_config.prior_scaling[1]
    bboxes[index] = bboxes_t

    num_match = np.array([len(np.nonzero(t_label)[0])], dtype=np.int32)
    return bboxes, t_label.astype(np.int32), num_match


def ssd_bboxes_filter(boxes, box_scores, image_shape):
    """
    Filter predict boxes with minimum score and nms threshold.

    Args:
    boxes: ground truth with shape [N, 4], for each row, it stores
        [ymin, xmin, ymax, xmax].
    box_scores: class scores with shape [N, 21].
    image_shape: the shape of original image with the format [h, w].
    """
    final_boxes = []
    final_label = []
    final_score = []
    h, w = image_shape
    # Ignore background(0) label class
    for c in range(1, box_scores.shape[1]):
        class_box_scores = box_scores[:, c]
        score_mask = class_box_scores > ssd300_config.min_score
        class_box_scores = class_box_scores[score_mask]
        class_boxes = boxes[score_mask] * [h, w, h, w]

        if score_mask.any():
            nms_index = apply_nms(class_boxes, class_box_scores,
                                  ssd300_config.nms_threshold)
            class_boxes = class_boxes[nms_index]
            class_box_scores = class_box_scores[nms_index]

            final_boxes += class_boxes.tolist()
            final_score += class_box_scores.tolist()
            final_label += [c+1] * len(class_box_scores)

    return final_boxes, final_score, final_label


def jaccard_numpy(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes."""
    def intersect(box_a, box_b):
        """Compute the intersect of two sets of boxes."""
        max_yx = np.minimum(box_a[:, 2:4], box_b[2:4])
        min_yx = np.maximum(box_a[:, :2], box_b[:2])
        inter = np.clip((max_yx - min_yx), a_min=0, a_max=np.inf)
        return inter[:, 0] * inter[:, 1]

    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2] - box_a[:, 0]) *
              (box_a[:, 3] - box_a[:, 1]))
    area_b = ((box_b[2] - box_b[0]) *
              (box_b[3] - box_b[1]))
    union = area_a + area_b - inter
    return inter / union


def apply_nms(all_boxes, all_scores, thres=0.6, max_boxes=100):
    """Apply NMS to all bounding boxes."""
    y1 = all_boxes[:, 0]
    x1 = all_boxes[:, 1]
    y2 = all_boxes[:, 2]
    x2 = all_boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    order = all_scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        if len(keep) >= max_boxes:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= thres)[0]
        order = order[inds + 1]
    return keep


def coco_eval(pred_data, anno_file):
    """Calculate mAP of predicted bboxes."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    predictions = []
    img_ids = []
    for sample in pred_data:
        pred_boxes = sample['boxes']
        box_scores = sample['box_scores']
        img_id = sample['img_id']
        img_ids.append(img_id)

        final_pred = ssd_bboxes_filter(pred_boxes, box_scores, sample['image_shape'])

        for loc, score, label in zip(final_pred[0], final_pred[1], final_pred[2]):
            res = {}
            res['image_id'] = img_id
            res['bbox'] = [loc[1], loc[0], loc[3] - loc[1], loc[2] - loc[0]]
            res['score'] = score
            res['category_id'] = label
            predictions.append(res)
    with open('predictions.json', 'w') as f:
        json.dump(predictions, f)

    coco_gt = COCO(anno_file)
    coco_dt = coco_gt.loadRes('predictions.json')
    E = COCOeval(coco_gt, coco_dt, iouType='bbox')
    E.params.imgIds = img_ids
    E.evaluate()
    E.accumulate()
    E.summarize()
    return E.stats[0]
