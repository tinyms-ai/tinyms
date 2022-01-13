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
import codecs
import collections
import pickle

from itertools import chain
from tqdm import tqdm

from mindspore.dataset import engine
from mindspore.dataset.engine import *
from mindspore.mindrecord import FileWriter
from .utils import generate_image_list, load_img


common_dataset = ['UnalignedDataset', 'GanImageFolderDataset', 'ImdbDataset', 'BertDataset',
                  'KaggleDisplayAdvertisingDataset']
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


class KaggleDisplayAdvertisingDataset:
    """
    parse aclImdb data to features and labels.
    sentence->tokenized->encoded->padding->features

    Args:
        data_dir (str): The path where the uncompressed dataset stored.
        num_parallel_workers (int): The number of concurrent workers. Default: None.
        shuffle (bol): Whether the dataset needs to be shuffled. Default: True.

    Examples:
        >>> from tinyms.data import KaggleDisplayAdvertisingDataset
        >>>
        >>> kaggle_display_advertising_ds = KaggleDisplayAdvertisingDataset('data')
        >>> kaggle_display_advertising_ds.stats_data()
        >>> kaggle_display_advertising_ds.convert_to_mindrecord()
        >>> train_ds = kaggle_display_advertising_ds.load_mindreocrd_dataset(usage='train')
        >>> test_ds = kaggle_display_advertising_ds.load_mindreocrd_dataset(usage='test')
    """
    def __init__(self, data_dir, num_parallel_workers=None, shuffle=True):
        self.data_dir = data_dir
        self.dense_dim = 13
        self.slot_dim = 26
        self.field_size = 39  # dense_dim + slot_dim
        self.skip_id_convert = False
        self.train_line_count = 45840617
        self.test_size = 0.1
        self.seed = 20191005
        self.line_per_sample = 1000
        self.epochs = 1
        self.num_parallel_workers = num_parallel_workers
        self._check_num_parallel_workers()
        self.shuffle = shuffle

        self.__init_stats()
        self.__init_mindrecord()

    def __check_mindrecord_dir(self):
        if os.path.exists(self.mindrecord_dir):
            print("mindrecord directory: {} exists! we will use it to save or read mindrecord dataset.".
                  format(self.mindrecord_dir), flush=True)
        else:
            print("create mindrecord directory: {} to save or read mindrecord dataset.".
                  format(self.mindrecord_dir), flush=True)
            os.makedirs(self.mindrecord_dir)

    def _check_num_parallel_workers(self):
        # use multiprocessing to get cpu count
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()

        if self.num_parallel_workers is not None:
            if cpu_count < self.num_parallel_workers:
                raise ValueError("num_parallel_workers: {} is bigger than cpu count: {}!".format(
                    self.num_parallel_workers, cpu_count))

    def __check_stats_dict_dir(self):
        if os.path.exists(self.stats_dict_dir):
            print("stats dict directory: {} exists! we will use it to save or read stats dict.".
                  format(self.stats_dict_dir), flush=True)
        else:
            print("create stats dict directory: {} to save or read stats dict.".
                  format(self.stats_dict_dir), flush=True)
            os.makedirs(self.stats_dict_dir)

    def __init_mindrecord(self):
        """
        init mindrecord
        """
        self.mindrecord_dir = os.path.join(self.data_dir, "mindrecord")
        self.__check_mindrecord_dir()

    def __init_stats(self):
        """
        init stats values
        """
        self.val_cols = ["val_{}".format(i + 1) for i in range(self.dense_dim)]
        self.cat_cols = ["cat_{}".format(i + 1) for i in range(self.slot_dim)]
        self.val_min_dict = {col: 0 for col in self.val_cols}
        self.val_max_dict = {col: 0 for col in self.val_cols}
        self.cat_count_dict = {col: collections.defaultdict(int) for col in self.cat_cols}
        self.oov_prefix = "OOV"
        self.cat2id_dict = {}
        self.cat2id_dict.update({col: i for i, col in enumerate(self.val_cols)})
        self.cat2id_dict.update(
            {self.oov_prefix + col: i + len(self.val_cols) for i, col in enumerate(self.cat_cols)})
        self.stats_dict_dir = os.path.join(self.data_dir, "stats_dict/")
        self.__check_stats_dict_dir()

    def __stats_vals(self, val_list):
        """
        handling weights column
        """
        assert len(val_list) == len(self.val_cols)
        for i, val in enumerate(val_list):
            key = self.val_cols[i]
            if val != "":
                if float(val) > self.val_max_dict[key]:
                    self.val_max_dict[key] = float(val)
                if float(val) < self.val_min_dict[key]:
                    self.val_min_dict[key] = float(val)

    def __stats_cats(self, cat_list):
        """
        handling cats column
        """
        assert len(cat_list) == len(self.cat_cols)
        for i, cat in enumerate(cat_list):
            key = self.cat_cols[i]
            self.cat_count_dict[key][cat] += 1

    def __save_stats_dict(self):
        """
        save stats dict
        """
        with open(os.path.join(self.stats_dict_dir, "val_max_dict.pkl"), "wb") as f:
            pickle.dump(self.val_max_dict, f)
        with open(os.path.join(self.stats_dict_dir, "val_min_dict.pkl"), "wb") as f:
            pickle.dump(self.val_min_dict, f)
        with open(os.path.join(self.stats_dict_dir, "cat_count_dict.pkl"), "wb") as f:
            pickle.dump(self.cat_count_dict, f)

    def __load_stats_dict(self,):
        """
        load stats dict
        """
        with open(os.path.join(self.stats_dict_dir, "val_max_dict.pkl"), "rb") as f:
            self.val_max_dict = pickle.load(f)
        with open(os.path.join(self.stats_dict_dir, "val_min_dict.pkl"), "rb") as f:
            self.val_min_dict = pickle.load(f)
        with open(os.path.join(self.stats_dict_dir, "cat_count_dict.pkl"), "rb") as f:
            self.cat_count_dict = pickle.load(f)
        print("val_max_dict.items()[:50]:{}".format(list(self.val_max_dict.items())), flush=True)
        print("val_min_dict.items()[:50]:{}".format(list(self.val_min_dict.items())), flush=True)

    def __get_cat2id(self, threshold=100):
        for key, cat_count_d in self.cat_count_dict.items():
            new_cat_count_d = dict(filter(lambda x: x[1] > threshold, cat_count_d.items()))
            for cat_str, _ in new_cat_count_d.items():
                self.cat2id_dict[key + "_" + cat_str] = len(self.cat2id_dict)
        print("cat2id_dict.size:{}".format(len(self.cat2id_dict)), flush=True)
        print("cat2id.dict.items()[:50]:{}".format(list(self.cat2id_dict.items())[:50]), flush=True)

    def __map_cat2id(self, values, cats):
        """
        category value to id value
        """
        id_list = []
        weight_list = []
        for i, val in enumerate(values):
            if val == "":
                id_list.append(i)
                weight_list.append(0)
            else:
                key = "val_{}".format(i + 1)
                id_list.append(self.cat2id_dict[key])
                max_v = float(self.val_max_dict["val_{}".format(i + 1)])
                minmax_scale_value = float(val) * 1.0 / max_v
                weight_list.append(minmax_scale_value)

        for i, cat_str in enumerate(cats):
            key = "cat_{}".format(i + 1) + "_" + cat_str
            if key in self.cat2id_dict:
                if self.skip_id_convert is True:
                    # For the synthetic data, if the generated id is between [0, max_vcoab], but the num examples is l
                    # ess than vocab_size/ slot_nums the id will still be converted to [0, real_vocab], where real_vocab
                    # the actually the vocab size, rather than the max_vocab. So a simple way to alleviate this
                    # problem is skip the id convert, regarding the synthetic data id as the final id.
                    id_list.append(cat_str)
                else:
                    id_list.append(self.cat2id_dict[key])
            else:
                id_list.append(self.cat2id_dict[self.oov_prefix + "cat_{}".format(i + 1)])
            weight_list.append(1.0)
        return id_list, weight_list

    def stats_data(self):
        """
        stats data
        """
        num_splits = self.dense_dim + self.slot_dim + 1
        error_stat_lines_num = []

        train_file_path = os.path.join(self.data_dir, "train.txt")
        with codecs.open(train_file_path, encoding="utf8", buffering=32*1024*1024) as f:
            t_f = tqdm(f, total=self.train_line_count)
            t_f.set_description("Processing StatsData")
            num_line = 0
            for line in t_f:
                num_line += 1
                line = line.strip("\n")
                items = line.split("\t")
                if len(items) != num_splits:
                    error_stat_lines_num.append(num_line)
                    # print("Found line length: {}, suppose to be {}, the line is {}".format(
                    #     len(items), num_splits, line))
                    continue
                # if num_line % 1000000 == 0:
                #     print("Have handled {}w lines.".format(num_line // 10000))
                values = items[1: self.dense_dim + 1]
                cats = items[self.dense_dim + 1:]
                assert len(values) == self.dense_dim, "values.size: {}".format(len(values))
                assert len(cats) == self.slot_dim, "cats.size: {}".format(len(cats))
                self.__stats_vals(values)
                self.__stats_cats(cats)
        self.__save_stats_dict()

        error_stat_path = os.path.join(self.data_dir, "error_stat_lines_num.npy")
        np.save(error_stat_path, error_stat_lines_num)

    def convert_to_mindrecord(self):
        test_size = int(self.train_line_count * self.test_size)
        all_indices = [i for i in range(self.train_line_count)]
        np.random.seed(self.seed)
        np.random.shuffle(all_indices)
        test_indices_set = set(all_indices[:test_size])

        train_data_list = []
        test_data_list = []
        ids_list = []
        wts_list = []
        label_list = []

        schema = {
            "label": {"type": "float32", "shape": [-1]},
            "feat_vals": {"type": "float32", "shape": [-1]},
            "feat_ids": {"type": "int32", "shape": [-1]}
        }

        train_writer = FileWriter(os.path.join(self.mindrecord_dir, "train_input_part.mindrecord"), 21)
        test_writer = FileWriter(os.path.join(self.mindrecord_dir, "test_input_part.mindrecord"), 3)
        train_writer.add_schema(schema, "CRITEO_TRAIN")
        test_writer.add_schema(schema, "CRITEO_TEST")

        part_rows = 2000000
        num_splits = self.dense_dim + self.slot_dim + 1
        error_conv_lines_num = []

        train_file_path = os.path.join(self.data_dir, "train.txt")
        with codecs.open(train_file_path, encoding="utf8", buffering=32*1024*1024) as f:
            t_f = tqdm(f, total=self.train_line_count)
            t_f.set_description("Processing Convert2MR")
            num_line = 0
            train_part_number = 0
            test_part_number = 0
            for line in t_f:
                num_line += 1
                # if num_line % 1000000 == 0:
                #     print("Converting to MindRecord. Have handle {}w lines.".format(num_line // 10000), flush=True)
                line = line.strip("\n")
                items = line.split("\t")
                if len(items) != num_splits:
                    error_conv_lines_num.append(num_line)
                    continue
                label = float(items[0])
                values = items[1: 1 + self.dense_dim]
                cats = items[1+self.dense_dim:]

                assert len(values) == self.dense_dim, "values.size: {}".format(len(values))
                assert len(cats) == self.slot_dim, "cats.size: {}".format(len(cats))

                ids, wts = self.__map_cat2id(values, cats)
                ids_list.extend(ids)
                wts_list.extend(wts)
                label_list.append(label)

                if num_line % self.line_per_sample == 0:
                    if num_line not in test_indices_set:
                        train_data_list.append({"feat_ids": np.array(ids_list, dtype=np.int32),
                                                "feat_vals": np.array(wts_list, dtype=np.float32),
                                                "label": np.array(label_list, dtype=np.float32)
                                                })
                    else:
                        test_data_list.append({"feat_ids": np.array(ids_list, dtype=np.int32),
                                               "feat_vals": np.array(wts_list, dtype=np.float32),
                                               "label": np.array(label_list, dtype=np.float32)
                                               })
                    if train_data_list and len(train_data_list) % part_rows == 0:
                        train_writer.write_raw_data(train_data_list)
                        train_data_list.clear()
                        train_part_number += 1

                    if test_data_list and len(test_data_list) % part_rows == 0:
                        test_writer.write_raw_data(test_data_list)
                        test_data_list.clear()
                        test_part_number += 1

                    ids_list.clear()
                    wts_list.clear()
                    label_list.clear()
            if train_data_list:
                train_writer.write_raw_data(train_data_list)
            if test_data_list:
                test_writer.write_raw_data(test_data_list)
        train_writer.commit()
        test_writer.commit()

        error_stat_path = os.path.join(self.data_dir, "error_conv_lines_num.npy")
        np.save(error_stat_path, error_conv_lines_num)

    def load_mindreocrd_dataset(self, usage='train', batch_size=1000):
        """
        load mindrecord dataset.
        Args:
            usage (str): Dataset mode. Default: 'train'.
            batch_size (int): batch size. Default: 1000.

        Returns:
            MindDataset
        """
        if usage == 'train':
            train_mode = True
        else:
            train_mode = False

        prefix_file_name = 'train_input_part.mindrecord' if train_mode else 'test_input_part.mindrecord'
        suffix_file_name = '00' if train_mode else '0'
        dataset_path = os.path.join(self.mindrecord_dir, prefix_file_name + suffix_file_name)

        dataset = MindDataset(dataset_path,
                              columns_list=['feat_ids', 'feat_vals', 'label'],
                              num_parallel_workers=self.num_parallel_workers,
                              shuffle=self.shuffle, num_shards=None, shard_id=None)

        dataset = dataset.batch(int(batch_size / self.line_per_sample), drop_remainder=True)
        dataset = dataset.map(operations=(lambda x, y, z: (np.array(x).flatten().reshape(batch_size, 39),
                                                           np.array(y).flatten().reshape(batch_size, 39),
                                                           np.array(z).flatten().reshape(batch_size, 1))),
                              input_columns=['feat_ids', 'feat_vals', 'label'],
                              column_order=['feat_ids', 'feat_vals', 'label'],
                              num_parallel_workers=self.num_parallel_workers)
        dataset = dataset.repeat(self.epochs)
        return dataset


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
