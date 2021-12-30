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
import numpy as np

import tinyms as ts
from tinyms import model
from tinyms.vision import voc_transform, shanshui_tranform

__all__ = ['ObjectDetector', 'object_detection_predict']

model_checker = {
    "ssd300": model.ssd300_mobilenetv2
}

transform_checker = {
    'voc': voc_transform,
    'shanshui': shanshui_tranform
}


class ObjectDetector():
    r"""
    ObjectDetector is a high-level class defined for building model，preproceing the input image,
    predicting and postprocessing the prediction output data.

    Args:
        config (dict): model config parsed from the json file under the app/object_detection/configs dir.
    """
    def __init__(self, config=None):
        self.config = config

    def data_preprocess(self, input):
        r"""
        Preprocess the input image.

        Args:
            input (numpy.ndarray): the input image.

        Returns:
            list, the preprocess image shape.
            numpy.ndarray, the preprocess image result.
        """
        if not isinstance(input, np.ndarray):
            err_msg = 'The input type should be numpy.ndarray, got {}.'.format(type(input))
            raise TypeError(err_msg)
        image_height, image_width, _ = input.shape
        image_shape = (image_height, image_width)
        cvt_input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

        transform_func = transform_checker.get(self.config.get('dataset'))
        if not transform_func:
            err_msg = 'Currently dataset only supports {} transform!'.format(str(list(transform_checker.keys())))
            raise KeyError(err_msg)
        transform_input = transform_func(cvt_input)
        return image_shape, transform_input

    def convert2tensor(self, transform_input):
        r"""
        Convert the numpy data to the tensor format.

        Args:
            transform_input (numpy.ndarray): the preprocessing image.

        Returns:
            Tensor, the converted image.
        """
        if not isinstance(transform_input, np.ndarray):
            err_msg = 'The transform_input type should be numpy.ndarray, got {}.'.format(type(transform_input))
            raise TypeError(err_msg)
        input_tensor = ts.expand_dims(ts.array(list(transform_input)), 0)
        return input_tensor

    def model_build(self, is_training=False):
        r"""
        Build the object detection model to predict the image.

        Args:
            is_training (bool): default: False.

        Returns:
            model.Model, generated object detection model.
        """
        model_net = model_checker.get(self.config.get('model_net'))
        if not model_net:
            err_msg = 'Currently model_net only supports {}!'.format(str(list(model_checker.keys())))
            raise KeyError(err_msg)

        class_num = self.config.get('class_num')
        if class_num <= 0:
            err_msg = 'The class_num should be an integer greater than 0, got {}.'.format(class_num)
            raise ValueError(err_msg)

        net = model_net(class_num=class_num, is_training=is_training)
        serve_model = model.Model(net)
        return serve_model

    def model_load_and_predict(self, serve_model, input_tensor):
        r"""
        Load the object detection model to predict the image.

        Args:
            serve_model (model.Model): object detection model.
            input_tensor (Tensor): the converted input image.

        Returns:
            model.Model, object detection model loaded the checkpoint file.
            list, predictions output result.
        """
        ckpt_path = self.config.get('checkpoint_path')
        if not ckpt_path:
            err_msg = 'The ckpt_path {} can not be none.'.format(ckpt_path)
            raise TypeError(err_msg)

        ckpt_name = self.config.get('checkpoint_name')
        if not ckpt_name.endswith('.ckpt'):
            err_msg = 'Currently model only supports `ckpt` format, got {}.'.format(ckpt_name)
            raise TypeError(err_msg)

        ckpt_file = os.path.join(ckpt_path, ckpt_name)
        if not os.path.isfile(ckpt_file):
            raise FileNotFoundError("The model checkpoint file path {} does not exist!".format(ckpt_file))
        serve_model.load_checkpoint(ckpt_file)

        predictions_output = serve_model.predict(input_tensor)
        return serve_model, predictions_output

    def data_postprocess(self, predictions_output, image_shape):
        r"""
        Postprocessing the predictions output data.

        Args:
            predictions_output (list): predictions output data.
            image_shape (list): the shape of the input image.

        Returns:
            dict, the postprocessing result.
        """
        output_np = (ts.concatenate((predictions_output[0], predictions_output[1]), axis=-1).asnumpy())
        transform_func = transform_checker.get(self.config.get('dataset'))
        if not transform_func:
            raise KeyError("Currently dataset only supports {} transform!".format(str(list(transform_checker.keys()))))
        bbox_data = transform_func.postprocess(output_np, image_shape)
        return bbox_data


def object_detection_predict(input, object_detector, is_training=False):
    r"""
    An easy object detection model predicting method for beginning developers to use.

    Args:
        input (numpy.ndarray): the input image.
        object_detector (ObjectDetector): the instance of the ObjectDetector class.
        is_training (bool): default: False.

    Returns:
        dict, the postprocessing result.
    """
    if not isinstance(object_detector, ObjectDetector):
        err_msg = 'The object_detector is not the instance of ObjectDetector'
        raise TypeError(err_msg)
    image_shape, transform_input = object_detector.data_preprocess(input)
    input_tensor = object_detector.convert2tensor(transform_input)
    serve_model = object_detector.model_build(is_training=is_training)
    _, predictions_output = object_detector.model_load_and_predict(serve_model, input_tensor)
    detection_bbox_data = object_detector.data_postprocess(predictions_output, image_shape)
    return detection_bbox_data
