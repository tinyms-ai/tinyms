import os
import json

import numpy as np
from PIL import Image
from tinyms import context
#from tinyms.serving import start_server, predict, list_servables, shutdown, server_started
from tinyms.data import Cifar10Dataset, download_dataset, ImageFolderDataset
from tinyms.vision import cifar10_transform, ImageViewer, imagefolder_transform
from tinyms.model import Model
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor, TimeMonitor
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits

from alexnet import AlexNet

def cosine_lr(base_lr, decay_steps, total_steps):
    lr_each_step=[]
    for i in range(total_steps):
        step_ = min(i, decay_steps)
        new_lr = base_lr * 0.5 * (1 + np.cos(np.pi * step_ / decay_steps))
        lr_each_step.append(new_lr)
    lr_each_step = np.array(lr_each_step).astype(np.float32)
    return lr_each_step

net = AlexNet(num_classes=10)
model = Model(net)

cifar10_path='./cifar-10-batches-bin'

# 检查ckpt文件和路径
cifar10_ckpt_folder = './serving/alexnet_cifar10'
cifar10_ckpt_path = './serving/alexnet_cifar10/alexnet.ckpt'
if not os.path.exists(cifar10_ckpt_folder):
    os.makedirs(cifar10_ckpt_folder)
else:
    print('alexnet_cifar10 ckpt folder already exists')

# 设置训练参数
lr = 0.01
epoch_size = 90 # default is 90
batch_size = 128

# 设置环境参数
dataset_sink_mode = True
device_target = "Ascend"
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

# 设置数据集参数
train_dataset = Cifar10Dataset(cifar10_path, usage='train',  num_parallel_workers=8, shuffle=True)
train_dataset = cifar10_transform.apply_ds(train_dataset, repeat_size=1, batch_size=batch_size, is_training=True)
eval_dataset = Cifar10Dataset(cifar10_path, usage='test', num_parallel_workers=4, shuffle=False)
eval_dataset = cifar10_transform.apply_ds(eval_dataset, repeat_size=1, batch_size=32, is_training=False)
step_size = train_dataset.get_dataset_size()

lr = cosine_lr(lr, epoch_size*step_size, epoch_size*step_size)

#save_checkpoint_epochs = 5
#ckpoint_cb = ModelCheckpoint(prefix="resnet_cifar10", config=CheckpointConfig(
#            save_checkpoint_steps=save_checkpoint_epochs * train_dataset.get_dataset_size(),
#            keep_checkpoint_max=10))

# 定义loss函数
net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# 定义optimizer
net_opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), lr, 0.9)
model.compile(loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": Accuracy()}, amp_level='O3')


print('************************Start training*************************')
model.train(epoch_size, train_dataset, callbacks=[LossMonitor(), TimeMonitor()],dataset_sink_mode=dataset_sink_mode)
model.save_checkpoint(cifar10_ckpt_path)
print('************************Finished training*************************')

model.load_checkpoint(cifar10_ckpt_path)
print('************************Start evaluation*************************')
acc = model.eval(eval_dataset, dataset_sink_mode=dataset_sink_mode)
print("============== Accuracy:{} ==============".format(acc))
