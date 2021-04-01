重新build之后,将test/st/alexnet.py放置在tinyms同级目录下面</br>
 ```
python alexnet.py \
--num_classes 10 \
--epoch_size 1 \
--device_target GPU \
--dataset_path {cifar10-path}\
--batch_size 128
 ```