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
import cv2

__all__ = ['save_image', 'draw_boxes_on_image', 'visualize_boxes_on_image']

IMG_FORMAT = ('jpg', 'jpeg', 'png')


def save_image(img, save_dir='./', img_name='no_name', img_format='jpg'):
    r"""
    Save the prediction image.

    Args:
        img (numpy.ndarray): the input image.
        save_dir (str): the dir to save the prediction image.
        img_name (str): the name of the prediction image. Default: 'no_name'.
        img_format (str): the format of the prediction image. Default: 'jpg'.
    """
    if img_format.lower() not in IMG_FORMAT:
        raise Exception("当前图片格式仅支持", IMG_FORMAT)
    output_image = os.path.join(save_dir, '{}.{}'.format(img_name, img_format))
    cv2.imwrite(output_image, img)


def draw_boxes_on_image(img, boxes, box_scores, box_classes, box_color=(0, 255, 0),
                        box_thickness=3, text_font=cv2.FONT_HERSHEY_PLAIN,
                        font_scale=3, text_color=(0, 0, 255), font_size=3, show_scores=True):
    r"""
    Draw the prediction box for the input image.

    Args:
        img (numpy.ndarray): the input image.
        boxes (list): the box coordinates.
        box_color (list): the box color. Default: (0, 255, 0).
        box_thickness (int): box thickness. Default: 3.
        text_font (Enum): text font. Default: cv2.FONT_HERSHEY_PLAIN.
        font_scale (int): font scale. Default: 3.
        text_color (list): text color. Default: (0, 0, 255).
        font_size (int): font size. Default: 3.
        show_scores (bool): whether to show scores. Default: True.

    Returns:
        numpy.ndarray, the output image drawed the prediction box.
    """
    x = int(boxes[0])
    y = int(boxes[1])
    w = int(boxes[2])
    h = int(boxes[3])
    cv2.rectangle(img, (x, y), (x+w, y+h), box_color, box_thickness)
    text = '{}, {}'.format(box_classes, str(round(box_scores, 3))) if show_scores else box_classes
    cv2.putText(img, text, (x, y), text_font, font_scale, text_color, font_size)
    return img


def visualize_boxes_on_image(img, bbox_data, box_color=(0, 255, 0),
                             box_thickness=3, text_font=cv2.FONT_HERSHEY_PLAIN,
                             font_scale=3, text_color=(0, 0, 255), font_size=3, show_scores=True):
    r"""
    Visualize the prediction image.

    Args:
        img (numpy.ndarray): the input image.
        bbox_data (dict): the predictions box data.
        box_color (list): the box color. Default: (0, 255, 0).
        box_thickness (int): box thickness. Default: 3.
        text_font (Enum): text font. Default: cv2.FONT_HERSHEY_PLAIN.
        font_scale (int): font scale. Default: 3.
        text_color (list): text color. Default: (0, 0, 255).
        font_size (int): font size. Default: 3.
        show_scores (bool): whether to show scores. Default: True.

    Returns:
        numpy.ndarray, the output image drawed the prediction box.
    """
    bbox_num = len(bbox_data)
    if bbox_num:
        for i in range(bbox_num):
            img = draw_boxes_on_image(img,
                                      bbox_data[i]['bbox'],
                                      bbox_data[i]['score'],
                                      bbox_data[i]['category_id'],
                                      box_color, box_thickness,
                                      text_font,
                                      font_scale,
                                      text_color,
                                      font_size,
                                      show_scores)
    return img
