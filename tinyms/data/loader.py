# Copyright 2generate_image_list021 Huawei Technologies Co., Ltd
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
import random
import numpy as np
import math

from mindspore.dataset import engine
from mindspore.dataset.engine import *
from tinyms.data.utils import generate_image_list, load_img


__all__ = ['UnalignedDataset', "GanImageFolderDataset", "DistributedSampler"]
__all__.extend(engine.__all__)

random.seed(1)


class UnalignedDataset:
    """
    This dataset class can load unaligned/unpaired datasets.

    Args:
        dataset_path (str): path of images (should have subfolders trainA, trainB, testA, testB, etc).
        phase (str): Train or test. It requires two directories in dataroot, like trainA and trainB to
            host training images from domain A '{dataroot}/trainA' and from domain B '{dataroot}/trainB' respectively.
        max_dataset_size (int): Maximum number of return image paths.

    Returns:
        Two domain image path list.
    """

    def __init__(self, dataset_path, phase, max_dataset_size=float("inf"), shuffle=True):
        self.dir_A = os.path.join(dataset_path, phase + 'A')
        self.dir_B = os.path.join(dataset_path, phase + 'B')

        self.A_paths = sorted(generate_image_list(self.dir_A,
                                                  max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(generate_image_list(self.dir_B,
                                                  max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
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

        return A_img, B_img

    def __len__(self):
        return max(self.A_size, self.B_size)


class GanImageFolderDataset:
    """
    This dataset class can load images from image folder.

    Args:
        dataset_path (str): */testA, */testB, etc.
        max_dataset_size (int): Maximum number of return image paths.

    Returns:
        Image path list.
    """

    def __init__(self, dataset_path, max_dataset_size=float("inf")):
        self.dataset_path = dataset_path
        self.paths = sorted(generate_image_list(dataset_path, max_dataset_size))
        self.size = len(self.paths)

    def __getitem__(self, index):
        img_path = self.paths[index % self.size]
        img = np.array(load_img(img_path))

        return img, os.path.split(img_path)[1]

    def __len__(self):
        return self.size


# Dataset distributed sampler
class DistributedSampler:
    """Distributed sampler."""
    def __init__(self, dataset_size, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            print("***********Setting world_size to 1 since it is not passed in ******************")
            num_replicas = 1
        if rank is None:
            print("***********Setting rank to 0 since it is not passed in ******************")
            rank = 0
        self.dataset_size = dataset_size
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(dataset_size * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        if self.shuffle:
            indices = np.random.RandomState(seed=self.epoch).permutation(self.dataset_size)
            # np.array type. number from 0 to len(dataset_size)-1, used as index of dataset
            indices = indices.tolist()
            self.epoch += 1
            # change to list type
        else:
            indices = list(range(self.dataset_size))

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples
