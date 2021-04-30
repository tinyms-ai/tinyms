import os
import json
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from PIL import Image
from tinyms import context
from tinyms.serving import start_server, predict, list_servables, shutdown, server_started
from tinyms.data import Cifar10Dataset, download_dataset, ImageFolderDataset
from tinyms.vision import cifar10_transform, ImageViewer, imagefolder_transform
from tinyms.model import Model, resnet50
from tinyms.callbacks import ModelCheckpoint, CheckpointConfig, LossMonitor
from tinyms.metrics import Accuracy
from tinyms.optimizers import Momentum
from tinyms.losses import SoftmaxCrossEntropyWithLogits
from model import densenet_BC_100

# build the network
net = densenet_BC_100(num_classes=10)
net.update_parameters_name(prefix='zjut')
model = Model(net)


# download the cifar10 dataset
cifar10_path = 'cifar10/cifar-10-batches-bin'


# check ckpt folder exists or not
cifar10_ckpt_folder = 'densenet_cifar10'
cifar10_ckpt_path = 'densenet_cifar10/densenet.ckpt'
if not os.path.exists(cifar10_ckpt_folder):
    os.makedirs(cifar10_ckpt_folder)
else:
    print('densenet_cifar10 ckpt folder already exists')

epoch_size = 1 # default is 90
batch_size = 128

# set environment parameters
dataset_sink_mode = False
device_target = "GPU"
# context.set_context(mode=context.PYNATIVE_MODE, device_target=device_target)
context.set_context(mode=context.GRAPH_MODE, device_target=device_target)
# set dataset parameters
train_dataset = Cifar10Dataset(cifar10_path, num_parallel_workers=4, shuffle=True)
train_dataset = cifar10_transform.apply_ds(train_dataset, repeat_size=1, batch_size=batch_size, is_training=True)
eval_dataset = Cifar10Dataset(cifar10_path, num_parallel_workers=4, shuffle=True)
eval_dataset = cifar10_transform.apply_ds(eval_dataset, repeat_size=1, batch_size=batch_size, is_training=False)
step_size = train_dataset.get_dataset_size()

save_checkpoint_epochs = 5
ckpoint_cb = ModelCheckpoint(prefix="densenet_cifar10", config=CheckpointConfig(
            save_checkpoint_steps=save_checkpoint_epochs * train_dataset.get_dataset_size(),
            keep_checkpoint_max=10))

# define the loss function
net_loss = SoftmaxCrossEntropyWithLogits(sparse=True, reduction="mean")

# define the optimizer
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