import os
import json

from PIL import Image
from tinyms import context
from tinyms.serving import start_server, predict, list_servables, shutdown, server_started
from tinyms.data import Cifar10Dataset, download_dataset, ImageFolderDataset
from tinyms.vision import cifar10_transform, ImageViewer, imagefolder_transform
from model import Model,alexnet
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits


# 构建网络
net = alexnet(class_num=10)
model = Model(net)


# download the cifar10 dataset
cifar10_path = './cifar10/cifar-10-batches-bin'
if not os.path.exists(cifar10_path):
    download_dataset('cifar10', './')
    print('************Download complete*************')
else:
    print('************Dataset already exists.**************')

# 检查ckpt文件和路径
cifar10_ckpt_folder = '/etc/tinyms/serving/alexnet_cifar10'
cifar10_ckpt_path = '/etc/tinyms/serving/alexnet_cifar10/alexnet_cifar10.ckpt'

# 设置训练参数
epoch_size = 5 # default is 90
batch_size = 32

# 设置环境参数
dataset_sink_mode = False
device_target = "CPU"
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)

# 设置数据集参数
train_dataset = Cifar10Dataset(cifar10_path, num_parallel_workers=8, shuffle=True)
train_dataset = cifar10_transform.apply_ds(train_dataset, repeat_size=1, batch_size=32, is_training=True)
eval_dataset = Cifar10Dataset(cifar10_path, num_parallel_workers=8, shuffle=True)
eval_dataset = cifar10_transform.apply_ds(eval_dataset, repeat_size=1, batch_size=32, is_training=False)
step_size = train_dataset.get_dataset_size()

save_checkpoint_epochs = 5
ckpoint_cb = ModelCheckpoint(prefix="alexnet_cifar10", config=CheckpointConfig(
            save_checkpoint_steps=save_checkpoint_epochs * train_dataset.get_dataset_size(),
            keep_checkpoint_max=10))

# 定义loss函数
net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# 定义optimizer
net_opt = Momentum(filter(lambda x: x.requires_grad, net.get_parameters()), 0.01, 0.9)
model.compile(loss_fn=net_loss, optimizer=net_opt, metrics={"Accuracy": Accuracy()})


print('************************Start training*************************')
model.train(epoch_size, train_dataset, callbacks=[ckpoint_cb, LossMonitor()],dataset_sink_mode=dataset_sink_mode)
model.save_checkpoint(cifar10_ckpt_path)
print('************************Finished training*************************')

model.load_checkpoint(cifar10_ckpt_path)
print('************************Start evaluation*************************')
acc = model.eval(eval_dataset, dataset_sink_mode=dataset_sink_mode)
print("============== Accuracy:{} ==============".format(acc))
