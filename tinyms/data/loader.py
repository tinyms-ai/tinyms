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
"""
Introduction to data/loader:

data/loader supports various formats of datasets, including ImageNet, TFData,
MNIST, Cifar10/100, Manifest, MindRecord, etc. This module could load data in
high performance and parse data precisely. It also provides the following
operations for users to preprocess data: shuffle, batch, repeat, map, and zip.
"""
import os
import random
import numpy as np
import math
import gensim
from itertools import chain

from mindspore.dataset import engine
from mindspore.dataset.engine import *
from mindspore.mindrecord import FileWriter
from .utils import generate_image_list, load_img


common_dataset = ['UnalignedDataset', 'GanImageFolderDataset', 'ImdbDataset', 'BertDataset']
common_sampler = ['DistributedSampler']

__all__ = common_dataset + common_sampler
__all__.extend(engine.__all__)

random.seed(1)


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
        dataset_path (str): '{dataset_path}/testA', '{dataset_path}/testB', etc.
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


class BertDataset:
    """
    This dataset class can load bert from data folder.

    Args:
        data_dir (str): '{data_dir}/result1.tfrecord', '{data_dir}/result2.tfrecord', etc.
        num_parallel_workers (int): The number of concurrent workers. Default: None.
        shuffle (Union[bool, Shuffle level], optional): Perform reshuffling of the data every epoch
            (default=Shuffle.GLOBAL).
            If shuffle is False, no shuffling will be performed;
            If shuffle is True, the behavior is the same as setting shuffle to be Shuffle.GLOBAL
            Otherwise, there are two levels of shuffling:

            - Shuffle.GLOBAL: Shuffle both the files and samples.

            - Shuffle.FILES: Shuffle files only.
        schema (Union[str, Schema], optional): Path to the JSON schema file or schema object (default=None).
            If the schema is not provided, the meta data from the TFData file is considered the schema.

    Examples:
        >>> from tinyms.data import BertDataset
        >>>
        >>> bert_ds = BertDataset('data')
    """

    def __init__(self, data_dir, schema_dir=None, shuffle=True, num_parallel_workers=None):
        files = os.listdir(data_dir)

        self.data_files = []
        for file_name in files:
            if "tfrecord" in file_name:
                self.data_files.append(os.path.join(data_dir, file_name))
        self.data_set = TFRecordDataset(self.data_files, schema_dir if schema_dir != "" else None,
                                      columns_list=["input_ids", "input_mask", "segment_ids", "next_sentence_labels",
                                                    "masked_lm_positions", "masked_lm_ids", "masked_lm_weights"],
                                      shuffle=shuffle,
                                      shard_equal_rows=True,
                                      num_parallel_workers=num_parallel_workers)


class ImdbDataset:
    """
    parse aclImdb data to features and labels.
    sentence->tokenized->encoded->padding->features

    Args:
        imdb_path (str): The path where the aclImdb dataset stored.
        glove_path (str): The path where the GloVe stored.
        embed_size (int): Embed_size. Default: 300.

    Examples:
        >>> from tinyms.data import ImdbDataset
        >>>
        >>> imdb_ds = ImdbDataset('./aclImdb', './glove')
    """

    def __init__(self, imdb_path, glove_path, embed_size=300):
        self.__segs = ['train', 'test']
        self.__label_dic = {'pos': 1, 'neg': 0}
        self.__imdb_path = imdb_path
        self.__glove_dim = embed_size
        self.__glove_file = os.path.join(glove_path, 'glove.6B.' + str(self.__glove_dim) + 'd.txt')

        # properties
        self.__imdb_datas = {}
        self.__features = {}
        self.__labels = {}
        self.__vacab = {}
        self.__word2idx = {}
        self.__weight_np = {}
        self.__wvmodel = None
        self.parse()

    def parse(self):
        """
        parse imdb data to memory
        """
        self.__wvmodel = gensim.models.KeyedVectors.load_word2vec_format(self.__glove_file, binary=False)

        for seg in self.__segs:
            self.__parse_imdb_datas(seg)
            self.__parse_features_and_labels(seg)
            self.__gen_weight_np(seg)

    def __parse_imdb_datas(self, seg):
        """
        load data from txt
        """
        data_lists = []
        for label_name, label_id in self.__label_dic.items():
            sentence_dir = os.path.join(self.__imdb_path, seg, label_name)
            for file in os.listdir(sentence_dir):
                with open(os.path.join(sentence_dir, file), mode='r', encoding='utf8') as f:
                    sentence = f.read().replace('\n', '')
                    data_lists.append([sentence, label_id])
        self.__imdb_datas[seg] = data_lists

    def __parse_features_and_labels(self, seg):
        """
        parse features and labels
        """
        features = []
        labels = []
        for sentence, label in self.__imdb_datas[seg]:
            features.append(sentence)
            labels.append(label)

        self.__features[seg] = features
        self.__labels[seg] = labels

        # update feature to tokenized
        self.__updata_features_to_tokenized(seg)
        # parse vacab
        self.__parse_vacab(seg)
        # encode feature
        self.__encode_features(seg)
        # padding feature
        self.__padding_features(seg)

    def __updata_features_to_tokenized(self, seg):
        tokenized_features = []
        for sentence in self.__features[seg]:
            tokenized_sentence = [word.lower() for word in sentence.split(" ")]
            tokenized_features.append(tokenized_sentence)
        self.__features[seg] = tokenized_features

    def __parse_vacab(self, seg):
        # vocab
        tokenized_features = self.__features[seg]
        vocab = set(chain(*tokenized_features))
        self.__vacab[seg] = vocab

        # word_to_idx looks like {'hello': 1, 'world':111, ... '<unk>': 0}
        word_to_idx = {word: i + 1 for i, word in enumerate(vocab)}
        word_to_idx['<unk>'] = 0
        self.__word2idx[seg] = word_to_idx

    def __encode_features(self, seg):
        """ encode word to index """
        word_to_idx = self.__word2idx['train']
        encoded_features = []
        for tokenized_sentence in self.__features[seg]:
            encoded_sentence = []
            for word in tokenized_sentence:
                encoded_sentence.append(word_to_idx.get(word, 0))
            encoded_features.append(encoded_sentence)
        self.__features[seg] = encoded_features

    def __padding_features(self, seg, maxlen=500, pad=0):
        """ pad all features to the same length """
        padded_features = []
        for feature in self.__features[seg]:
            if len(feature) >= maxlen:
                padded_feature = feature[:maxlen]
            else:
                padded_feature = feature
                while len(padded_feature) < maxlen:
                    padded_feature.append(pad)
            padded_features.append(padded_feature)
        self.__features[seg] = padded_features

    def __gen_weight_np(self, seg):
        """
        generate weight by gensim
        """
        weight_np = np.zeros((len(self.__word2idx[seg]), self.__glove_dim), dtype=np.float32)
        for word, idx in self.__word2idx[seg].items():
            if word not in self.__wvmodel:
                continue
            word_vector = self.__wvmodel.get_vector(word)
            weight_np[idx, :] = word_vector

        self.__weight_np[seg] = weight_np

    def get_datas(self, seg):
        """
        get features, labels, and weight by gensim.
        """
        features = np.array(self.__features[seg]).astype(np.int32)
        labels = np.array(self.__labels[seg]).astype(np.int32)
        weight = np.array(self.__weight_np[seg])
        return features, labels, weight

    def convert_to_mindrecord(self, preprocess_path, shard_num=1):
        """
        convert imdb dataset to mindrecoed dataset
        """

        if not os.path.exists(preprocess_path):
            print(f"preprocess path {preprocess_path} is not exist")
            os.makedirs(preprocess_path)

        train_features, train_labels, train_weight_np = self.get_datas('train')
        self._convert_to_mindrecord(preprocess_path, train_features, train_labels, train_weight_np, shard_num=shard_num)

        test_features, test_labels, _ = self.get_datas('test')
        self._convert_to_mindrecord(preprocess_path, test_features, test_labels, training=False, shard_num=shard_num)

    def _convert_to_mindrecord(self, data_home, features, labels, weight_np=None, training=True, shard_num=1):
        """
        convert imdb dataset to mindrecoed dataset
        """
        if weight_np is not None:
            np.savetxt(os.path.join(data_home, 'weight.txt'), weight_np)

        # write mindrecord
        schema_json = {"id": {"type": "int32"},
                       "label": {"type": "int32"},
                       "feature": {"type": "int32", "shape": [-1]}}

        data_dir = os.path.join(data_home, "aclImdb_train.mindrecord")
        if not training:
            data_dir = os.path.join(data_home, "aclImdb_test.mindrecord")

        def get_imdb_data(features, labels):
            data_list = []
            for i, (label, feature) in enumerate(zip(labels, features)):
                data_json = {"id": i,
                             "label": int(label),
                             "feature": feature.reshape(-1)}
                data_list.append(data_json)
            return data_list

        writer = FileWriter(data_dir, shard_num=shard_num)
        data = get_imdb_data(features, labels)
        writer.add_schema(schema_json, "nlp_schema")
        writer.add_index(["id", "label"])
        writer.write_raw_data(data)
        writer.commit()


class DistributedSampler:
    """
    Distributed sampler.

    Args:
        dataset_size (int): Dataset list length
        num_replicas (int): Replicas num.
        rank (int): Device rank.
        shuffle (bool): Whether the dataset needs to be shuffled. Default: True.

    Returns:
        DistributedSampler instance.
    """

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