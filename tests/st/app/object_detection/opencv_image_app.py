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
import argparse

from tinyms.app.object_detection.utils.config_util import load_and_parse_config
from tinyms.app.object_detection.object_detector import ObjectDetector, object_detection_predict
from tinyms.app.object_detection.utils.view_util import visualize_boxes_on_image


def parse_args():
    parser = argparse.ArgumentParser(description='TinyMS Object Detection Using OpenCV Example')
    parser.add_argument('--config_path', type=str, default=None,
                        help='the json file of model config under the tinyms/app/object_detection/configs/.')
    parser.add_argument('--img_path', type=str, default="./pic/test.jpg", help='the path of the image to be detected.')
    parser.add_argument("--window_size", type=list, default=(800, 600), help="the size of the display window.")
    args_opt = parser.parse_args()
    return args_opt


if __name__ == '__main__':
    args_opt = parse_args()
    config_path = args_opt.config_path
    if not config_path:
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('tests')[0],
                                   "tinyms/app/object_detection/configs/tinyms/0.3/ssd300_shanshui.json")
    # 1.Load and parse the config json file
    config = load_and_parse_config(config_path=config_path)

    # 2.Generate the instance of ObjectDetector
    detector = ObjectDetector(config=config)

    # 3.Read the input image
    image_np = cv2.imread(args_opt.img_path)
    input = image_np.copy()

    # 4.Detect the input image
    detection_bbox_data = object_detection_predict(input, detector, is_training=False)

    # 5.Draw the box for the input image and visualize in the opencv window.
    detection_image_np = visualize_boxes_on_image(image_np, detection_bbox_data, box_color=(0, 255, 0),
                                                  box_thickness=3, text_font=cv2.FONT_HERSHEY_PLAIN,
                                                  font_scale=3, text_color=(0, 0, 255), font_size=3, show_scores=True)
    cv2.imshow('object detection image', cv2.resize(detection_image_np, args_opt.window_size))
    cv2.waitKey(0)



