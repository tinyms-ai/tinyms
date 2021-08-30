import os
import random

import numpy as np
from tinyms.data import GeneratorDataset, GanImageFolderDataset, DistributedSampler, generate_image_list, load_img
from data.transform import cyclegan_transform


def create_dataset(dataset_path, batch_size=1, repeat_size=1, max_dataset_size=None,
                   shuffle=True, num_parallel_workers=1, phase='train', data_dir='testA', use_S=False):
    """ create Mnist dataset for train or eval.
    dataset_path: Data path
    batch_size: The number of data records in each group
    repeat_size: The number of replicated data records
    num_parallel_workers: The number of parallel workers
    """
    # define dataset and apply the transform func
    if phase == 'train':
        ds = UnalignedDataset(dataset_path, phase, max_dataset_size=max_dataset_size, shuffle=True, use_S=use_S)
        column_names = ["image_A", "image_B"]
        if use_S:
            column_names.append('image_S')
        device_num = 1
        distributed_sampler = DistributedSampler(len(ds), num_replicas=device_num, rank=0, shuffle=shuffle)
        gan_generator_ds = GeneratorDataset(ds, column_names=column_names, sampler=distributed_sampler,
                                            num_parallel_workers=num_parallel_workers)
    else:
        data_dir = os.path.join(dataset_path, data_dir)
        ds = GanImageFolderDataset(data_dir, max_dataset_size=max_dataset_size)
        gan_generator_ds = GeneratorDataset(ds, column_names=["image", "image_name"],
                                            num_parallel_workers=num_parallel_workers)

    gan_generator_ds = cyclegan_transform.apply_ds(gan_generator_ds,
                                                   repeat_size=repeat_size,
                                                   batch_size=batch_size,
                                                   num_parallel_workers=num_parallel_workers,
                                                   shuffle=shuffle,
                                                   phase=phase,
                                                   use_S=use_S)
    dataset_size = len(ds)
    return gan_generator_ds, dataset_size


class UnalignedDataset:
    """
    This dataset class can load unaligned/unpaired datasets.

    Args:
        dataset_path (str): The path of images (should have subfolders trainA, trainB, testA, testB, etc).
        phase (str): Train or test. It requires two directories in dataset_path, like trainA and trainB to.
            host training images from domain A '{dataset_path}/trainA' and from domain B '{dataset_path}/trainB'
            respectively.
        max_dataset_size (int): Maximum number of return image paths.

    Returns:
        Two domain image path list.
    """

    def __init__(self, dataset_path, phase, max_dataset_size=float("inf"), shuffle=True, use_S=False):
        self.use_S = use_S
        self.dir_A = os.path.join(dataset_path, phase + 'A')
        self.dir_B = os.path.join(dataset_path, phase + 'B')

        self.A_paths = sorted(generate_image_list(self.dir_A,
                                                  max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(generate_image_list(self.dir_B,
                                                  max_dataset_size))  # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        if use_S:
            self.dir_S = os.path.join(dataset_path, phase + 'S')
            self.S_paths = sorted(generate_image_list(self.dir_S,
                                                      max_dataset_size))  # load images from '/path/to/data/trainS'
            self.S_size = len(self.S_paths)  # get the size of dataset S

        self.shuffle = shuffle

    def __getitem__(self, index):
        index_B = index % self.B_size
        if index % max(self.A_size, self.B_size) == 0 and self.shuffle:
            random.shuffle(self.A_paths)
            index_B = random.randint(0, self.B_size - 1)
        A_path = self.A_paths[index % self.A_size]
        B_path = self.B_paths[index_B]
        A_img = np.array(load_img(A_path))
        B_img = np.array(load_img(B_path))
        if self.use_S:
            S_path = self.S_paths[index_B]
            S_img = np.array(load_img(S_path))
            return A_img, B_img, S_img

        return A_img, B_img

    def __len__(self):
        return max(self.A_size, self.B_size)


