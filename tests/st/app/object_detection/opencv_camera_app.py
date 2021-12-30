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

from tinyms.app.object_detection.utils.config_util import load_and_parse_config
from tinyms.app.object_detection.object_detector import ObjectDetector, object_detection_predict
from tinyms.app.object_detection.utils.view_util import visualize_boxes_on_image

# 1.Load and parse the config json file
config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('tests')[0],
                           "tinyms/app/object_detection/configs/tinyms/0.3/ssd300_voc.json")
config = load_and_parse_config(config_path=config_path)

# 2.Generate the instance of ObjectDetector
detector = ObjectDetector(config=config)

cap = cv2.VideoCapture(0)
while True:
    # 3.Read the frame image from the camera using OpenCV
    ret, image_np = cap.read()
    input = image_np.copy()

    # 4.Detect the input frame image
    detection_bbox_data = object_detection_predict(input, detector, is_training=False)

    # 5.Draw the box for the input frame image and view it using OpenCV.
    detection_image_np = visualize_boxes_on_image(image_np, detection_bbox_data, box_color=(0, 255, 0),
                                                  box_thickness=3, text_font=cv2.FONT_HERSHEY_PLAIN,
                                                  font_scale=2, text_color=(0, 0, 255), font_size=3, show_scores=True)
    cv2.imshow('object detection camera', cv2.resize(detection_image_np, (800, 600)))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
