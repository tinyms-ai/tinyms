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
import json
import time
import argparse
import xml.etree.ElementTree as et

import tinyms as ts
from tinyms import context, layers, primitives as P, Tensor
from tinyms.data import VOCDataset, download_dataset
from tinyms.vision import voc_transform, coco_eval
from tinyms.model import Model, ssd300_mobilenetv2
from tinyms.losses import net_with_loss
from tinyms.optimizers import Momentum
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from tinyms.utils.train.lr_generator import mobilenetv2_lr as ssd300_lr
from tinyms.initializers import initializer, TruncatedNormal


def parse_args():
    parser = argparse.ArgumentParser(description="SSD300 object detection")
    parser.add_argument('--device_target', type=str, default="CPU", choices=['Ascend', 'GPU', 'CPU'],
                        help='device where the code will be implemented (default: CPU)')
    parser.add_argument('--dataset_path', type=str, default=None, help='VOC2007/VOC2012 dataset path.')
    parser.add_argument("--num_classes", type=int, default=21, help="The VOC dataset class number, default is 21.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate, default is 0.01.")
    parser.add_argument("--epoch_size", type=int, default=800, help="Epoch size, default is 800.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size, default is 32.")
    parser.add_argument("--pre_trained_epoch_size", type=int, default=0, help="Pretrained epoch size.")
    parser.add_argument('--save_checkpoint_epochs', type=int, default=10,
                        help='Specify epochs interval to save each checkpoints.')
    parser.add_argument("--loss_scale", type=int, default=1, help="Loss scale, default is 1.")
    parser.add_argument('--do_eval', type=bool, default=False, help='Do eval or not. (default: False)')
    parser.add_argument('--load_pretrained', type=str, choices=['hub', 'local'], default='local',
                        help='Specify where to load pretrained model, only valid in do_eval mode. (default: local)')
    parser.add_argument('--checkpoint_path', type=str, default=None,
                        help='Checkpoint file path. Only valid when load_pretrained is `local`.')
    parser.add_argument('--hub_uid', type=str, default=None,
                        help='Model asset uid. Only valid when load_pretrained is `hub`.')
    args_opt = parser.parse_args()

    return args_opt


def create_dataset(data_path, batch_size=32, repeat_size=1, num_parallel_workers=4,
                   is_training=True):
    """ create V0C2007 dataset for train or eval.
    Args:
        data_path: Data path
        batch_size: The number of data records in each group
        repeat_size: The number of replicated data records
        num_parallel_workers: The number of parallel workers
    """
    # define dataset and apply the transform func
    usage = 'trainval' if is_training else 'val'
    voc_ds = VOCDataset(data_path, task='Detection', usage=usage, num_parallel_workers=num_parallel_workers,
                        shuffle=is_training, decode=True)
    voc_ds = voc_transform.apply_ds(voc_ds,
                                    repeat_size=repeat_size,
                                    batch_size=batch_size,
                                    num_parallel_workers=num_parallel_workers,
                                    is_training=is_training)

    return voc_ds


def init_net_param(network, initialize_mode='TruncatedNormal'):
    """Init the parameters in net."""
    params = network.trainable_params()
    for p in params:
        if 'beta' not in p.name and 'gamma' not in p.name and 'bias' not in p.name:
            if initialize_mode == 'TruncatedNormal':
                p.set_data(initializer(TruncatedNormal(0.02), p.data.shape, p.data.dtype))
            else:
                p.set_data(initialize_mode, p.data.shape, p.data.dtype)


class TrainingWrapper(layers.Layer):
    """
    Encapsulation class of SSD300 network training.

    Append an optimizer to the training network after that the construct
    function can be called to create the backward graph.

    Args:
        network (Layer): The training network. Note that loss function should have been added.
        optimizer (Optimizer): Optimizer for updating the weights.
        sens (float): The adjust parameter. Default: 1.0.
    """

    def __init__(self, network, optimizer, sens=1.0):
        super(TrainingWrapper, self).__init__(auto_prefix=False)
        self.network = network
        self.network.set_grad()
        self.weights = ts.ParameterTuple(network.trainable_params())
        self.optimizer = optimizer
        self.grad = P.GradOperation(get_by_list=True, sens_param=True)
        self.sens = sens
        self.hyper_map = P.HyperMap()

    def construct(self, *args):
        weights = self.weights
        loss = self.network(*args)
        sens = P.Fill()(P.DType()(loss), P.Shape()(loss), self.sens)
        grads = self.grad(self.network, weights)(*args, sens)
        return P.depend(loss, self.optimizer(grads))


def create_voc_label(voc_dir, voc_cls, usage='val'):
    """Get image path and annotation from VOC."""
    if not os.path.isdir(voc_dir):
        raise ValueError(f'Cannot find {voc_dir} dataset path.')
    anno_dir = voc_dir
    if os.path.isdir(os.path.join(voc_dir, 'Annotations')):
        anno_dir = os.path.join(voc_dir, 'Annotations')

    cls_map = {name: i for i, name in enumerate(voc_cls)}
    # Fetch the specific xml files path
    xml_files = []
    with open(os.path.join(voc_dir, 'ImageSets', 'Main', usage+'.txt'), 'r') as f:
        for line in f:
            xml_files.append(line.strip('\n')+'.xml')

    json_dict = {"images": [], "type": "instances", "annotations": [],
                 "categories": []}
    bnd_id = 1
    for xml_file in xml_files:
        img_id = xml_files.index(xml_file)
        tree = et.parse(os.path.join(anno_dir, xml_file))
        root_node = tree.getroot()
        file_name = root_node.find('filename').text

        for obj in root_node.iter('object'):
            cls_name = obj.find('name').text
            if cls_name not in cls_map:
                print(f'Label "{cls_name}" not in "{cls_map}"')
                continue

            bnd_box = obj.find('bndbox')
            x_min = int(float(bnd_box.find('xmin').text)) - 1
            y_min = int(float(bnd_box.find('ymin').text)) - 1
            x_max = int(float(bnd_box.find('xmax').text)) - 1
            y_max = int(float(bnd_box.find('ymax').text)) - 1
            o_width = abs(x_max - x_min)
            o_height = abs(y_max - y_min)
            ann = {'area': o_width * o_height, 'iscrowd': 0,
                   'image_id': img_id,
                   'bbox': [x_min, y_min, o_width, o_height],
                   'category_id': cls_map[cls_name], 'id': bnd_id,
                   'ignore': 0,
                   'segmentation': []}
            json_dict['annotations'].append(ann)
            bnd_id = bnd_id + 1

        size = root_node.find("size")
        width = int(size.find('width').text)
        height = int(size.find('height').text)
        image = {'file_name': file_name, 'height': height, 'width': width,
                 'id': img_id}
        json_dict['images'].append(image)

    for cls_name, cid in cls_map.items():
        cat = {'supercategory': 'none', 'id': cid, 'name': cls_name}
        json_dict['categories'].append(cat)

    anno_file = os.path.join(anno_dir, 'annotation.json')
    with open(anno_file, 'w') as f:
        json.dump(json_dict, f)
    return anno_file


if __name__ == '__main__':
    args_opt = parse_args()
    context.set_context(mode=context.GRAPH_MODE, device_target=args_opt.device_target)

    # download voc dataset
    if not args_opt.dataset_path:
        args_opt.dataset_path = download_dataset('voc')

    epoch_size = args_opt.epoch_size
    batch_size = args_opt.batch_size
    voc_path = args_opt.dataset_path
    dataset_sink_mode = not args_opt.device_target == "CPU"

    if not args_opt.do_eval:  # as for train, users could use model.train
        ds_train = create_dataset(voc_path, batch_size=batch_size)
        dataset_size = ds_train.get_dataset_size()
        # build the SSD300 network
        net = ssd300_mobilenetv2(class_num=args_opt.num_classes)
        # define the loss function
        if args_opt.device_target == "GPU":
            net.to_float(ts.float16)
        net = net_with_loss(net)
        init_net_param(net)
        # define the optimizer
        lr = ssd300_lr(global_step=args_opt.pre_trained_epoch_size * dataset_size,
                       lr_init=0.001, lr_end=0.001 * args_opt.lr, lr_max=args_opt.lr,
                       warmup_epochs=2, total_epochs=args_opt.epoch_size,
                       steps_per_epoch=dataset_size)
        loss_scale = 1.0 if args_opt.device_target == "CPU" else float(args_opt.loss_scale)
        opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr,
                       0.9, 1.5e-4, loss_scale)
        model = Model(TrainingWrapper(net, opt, loss_scale))
        model.compile()

        ckpoint_cb = ModelCheckpoint(prefix="ssd300", config=CheckpointConfig(
            save_checkpoint_steps=args_opt.save_checkpoint_epochs * dataset_size,
            keep_checkpoint_max=10))
        model.train(epoch_size, ds_train, callbacks=[ckpoint_cb, LossMonitor(), TimeMonitor(data_size=dataset_size)],
                    dataset_sink_mode=dataset_sink_mode)
    else:  # as for evaluation, users could use model.eval
        ds_eval = create_dataset(voc_path, batch_size=1, is_training=False)
        total = ds_eval.get_dataset_size()
        # define the infer wrapper
        if args_opt.load_pretrained == 'hub':
            from tinyms import hub
            eval_net = hub.load(args_opt.hub_uid, class_num=args_opt.num_classes, is_training=False)
        else:
            eval_net = ssd300_mobilenetv2(class_num=args_opt.num_classes, is_training=False)
        model = Model(eval_net)
        if args_opt.load_pretrained == 'local':
            if args_opt.checkpoint_path:
                model.load_checkpoint(args_opt.checkpoint_path)
        # perform the model predict operation
        print("\n========================================\n")
        print("total images num: ", total)
        print("Processing, please wait a moment...")
        start = time.time()
        pred_data = []
        id_iter = 0
        for data in ds_eval.create_dict_iterator(output_numpy=True):
            image_np = data['image']
            image_shape = data['image_shape']

            output = model.predict(Tensor(image_np))
            for batch_idx in range(image_np.shape[0]):
                pred_data.append({"boxes": output[0].asnumpy()[batch_idx],
                                  "box_scores": output[1].asnumpy()[batch_idx],
                                  "img_id": id_iter,
                                  "image_shape": image_shape[batch_idx]})
                id_iter += 1
        cost_time = int((time.time() - start) * 1000)
        print(f'    100% [{total}/{total}] cost {cost_time} ms')
        # calculate mAP for the predict data
        voc_cls = ['background',
                   'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
                   'bus', 'car', 'cat', 'chair', 'cow',
                   'diningtable', 'dog', 'horse', 'motorbike', 'person',
                   'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
        anno_file = create_voc_label(voc_path, voc_cls)
        mAP = coco_eval(pred_data, anno_file)
        print("\n========================================\n")
        print(f"mAP: {mAP}")
