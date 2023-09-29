import os
import gc
from random import randint
from collections import defaultdict

import pandas as pd
import albumentations
import numpy as np
from PIL import Image
import imagesize
import mindspore as ms
from mindspore.dataset import GeneratorDataset

from toolz.sandbox import unzip

def control_collate(inputs):
    """
    Return:
    :img_feat     (batch_size, height, weight, 3)
    :txt_tokens   (n, max_txt_len)
    """
    img_feat, txt_tokens, control_feat = map(list, unzip(inputs))
    batch = {
        'img_feat': img_feat,
        'txt_tokens': txt_tokens,
        'control_feat': control_feat
    }
    return batch

data_column = [
    'img_feat',
    'txt_tokens',
    'control_feat'
]


def load_data(
            data_path,
            batch_size,
            tokenizer,
            image_size=512,
            image_filter_size=256,
            device_num=1,
            random_crop=False, 
            filter_small_size=True,
            rank_id=0,
            sample_num=-1
            ):
    
    
    if not os.path.exists(data_path):
        raise ValueError("Data directory does not exist!")
    all_images, all_captions, all_conds = list_image_files_captions_recursively(data_path)
    print(f"The first image path is {all_images[0]}, and the caption is {all_captions[0]}")
    print(f"total data num: {len(all_images)}")
    dataloaders = {}
    dataset = ImageDataset(
            batch_size,
            all_images,
            all_captions,
            all_conds,
            tokenizer,
            image_size,
            image_filter_size,
            random_crop=random_crop, 
            filter_small_size=filter_small_size
            )
    datalen = dataset.__len__
    loader = build_dataloader_ft(dataset, datalen, control_collate, batch_size, device_num, rank_id=rank_id)
    dataloaders["ftT2I"] = loader
    if sample_num==-1:
        batchlen = datalen//(batch_size * device_num)
    else:
        batchlen = sample_num
    metaloader = MetaLoader(dataloaders, datalen=batchlen, task_num=len(dataloaders.keys()))
    dataset = GeneratorDataset(metaloader, column_names=data_column, shuffle=True)

    return dataset


def build_dataloader_ft(dataset, datalens, collate_fn, batch_size, device_num, rank_id=0):
    sampler = BatchSampler(datalens, batch_size=batch_size, device_num=device_num)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_fn, device_num=device_num, drop_last=True, rank_id=rank_id)
    return loader


def list_image_files_captions_recursively(data_path):
    import json
    all_images = []
    all_conds = []
    all_captions = []
    with open(f'{data_path}/train.jsonl', 'r') as f:
        for line in f:
            data = json.loads(line)
            all_images.append(f'{data_path}/{data["image"]}')
            all_conds.append(f'{data_path}/{data["conditioning_image"]}')
            all_captions.append(data["text"])
    
    assert len(all_images) == len(all_captions)
    return all_images, all_captions, all_conds


class ImageDataset():
    def __init__(
        self,
        batch_size,
        image_paths,
        captions,
        conds,
        tokenizer,
        image_size,
        image_filter_size,
        shuffle=True,
        random_crop=False, 
        filter_small_size=False
    ):
        super().__init__()
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.image_filter_size = image_filter_size
        self.local_images = image_paths
        self.local_captions = captions
        self.local_control = conds
        self.shuffle = shuffle
        self.random_crop = random_crop
        self.filter_small_size = filter_small_size

    @property
    def __len__(self):
        return len(self.local_images)

    def random_sample(self):
        return self.__getitem__(randint(0, self.__len__() - 1))

    def sequential_sample(self, ind):
        if ind >= self.__len__() - 1:
            return self.__getitem__(0)
        return self.__getitem__(ind + 1)

    def skip_sample(self, ind):
        if self.shuffle:
            return self.random_sample()
        return self.sequential_sample(ind=ind)

    def __getitem__(self, idx):
        # images preprocess
        img_path = self.local_images[idx]
        img = Image.open(img_path).convert('RGB')
        img = np.asarray(img).astype(np.float32)
        img = (img / 127.5 - 1.0)

        # control
        control_path = self.local_control[idx]
        control = Image.open(control_path).convert('RGB')
        control = np.asarray(control).astype(np.float32)
        control = control / 255.0

        # caption preprocess
        caption = self.local_captions[idx]
        caption_input = self.tokenize(caption)
        return np.array(img, dtype=np.float32), np.array(caption_input, dtype=np.int32), np.array(control, dtype=np.float32)

    def tokenize(self, text):
        SOT_TEXT = "<|startoftext|>"
        EOT_TEXT = "<|endoftext|>"
        CONTEXT_LEN = 77

        sot_token = self.tokenizer.encoder[SOT_TEXT]
        eot_token = self.tokenizer.encoder[EOT_TEXT]
        tokens = [sot_token] + self.tokenizer.encode(text) + [eot_token]
        result = np.zeros([CONTEXT_LEN])
        if len(tokens) > CONTEXT_LEN:
            tokens = tokens[:CONTEXT_LEN - 1] + [eot_token]
        result[:len(tokens)] = tokens

        return result


class BatchSampler:
    """
        Batch Sampler
    """

    def __init__(self, lens, batch_size, device_num):
        self._lens = lens
        self._batch_size = batch_size * device_num

    def _create_ids(self):
        return list(range(self._lens))

    def __iter__(self):
        ids = self._create_ids()
        batches = [ids[i:i + self._batch_size] for i in range(0, len(ids), self._batch_size)]
        gc.collect()
        return iter(batches)

    def __len__(self):
        raise ValueError("NOT supported. "
                         "This has some randomness across epochs")


class DataLoader:
    """ DataLoader """

    def __init__(self, dataset, batch_sampler, collate_fn, device_num=1, drop_last=True, rank_id=0):
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collat_fn = collate_fn
        self.device_num = device_num
        self.rank_id = rank_id
        self.drop_last = drop_last
        self.batch_size = len(next(iter(self.batch_sampler)))

    def __iter__(self):
        self.step_index = 0
        self.batch_indices = iter(self.batch_sampler)

        return self

    def __next__(self):
        try:
            indices = next(self.batch_indices)
            if len(indices) != self.batch_size and self.drop_last:
                return self.__next__()
        except StopIteration:
            self.batch_indices = iter(self.batch_sampler)
            indices = next(self.batch_indices)
        data = []
        per_batch = len(indices) // self.device_num
        index = indices[self.rank_id * per_batch:(self.rank_id + 1) * per_batch]
        for idx in index:
            data.append(self.dataset[idx])

        data = self.collat_fn(data)
        return data


class MetaLoader():
    """ wraps multiple data loaders """

    def __init__(self, loaders, datalen, task_num=1):
        assert isinstance(loaders, dict)
        self.task_num = task_num
        self.name2loader = {}
        self.name2iter = {}
        self.sampling_pools = []
        self.loaders = loaders
        self.datalen = datalen
        for n, l in loaders.items():
            if isinstance(l, tuple):
                l, r = l
            elif isinstance(l, DataLoader):
                r = 1
            else:
                raise ValueError()
            self.name2loader[n] = l
            self.name2iter[n] = iter(l)
            self.sampling_pools.extend([n] * r)

        self.task = self.sampling_pools[0]
        self.task_label = [0] * self.task_num
        self.step = 0
        self.step_cnt = 0
        self.task_index_list = np.random.permutation(self.task_num)
        self.all_ids = []

    def init_iter(self, task_name):
        self.name2iter[task_name] = iter(self.name2loader[task_name])

    def return_ids(self):
        return self.all_ids

    def get_batch(self, batch, task):
        """ get_batch """
        batch = defaultdict(lambda: None, batch)
        img_feat = batch.get('img_feat', None)
        txt_tokens = batch.get('txt_tokens', None)
        control_feat = batch.get('control_feat', None)
        output = (img_feat, txt_tokens, control_feat)

        return output

    def __getitem__(self, index):
        if self.step_cnt == self.task_num:
            self.task_index_list = np.random.permutation(self.task_num)
            self.step_cnt = 0
        task_index = self.task_index_list[self.step_cnt]
        local_task = self.sampling_pools[task_index]

        iter_ = self.name2iter[local_task]

        name = local_task
        try:
            batch = next(iter_)
        except StopIteration:
            self.init_iter(local_task)
            iter_ = self.name2iter[local_task]
            batch = next(iter_)

        task = name.split('_')[0]
        for key, val in batch.items():
            if isinstance(val, np.ndarray):
                if val.dtype == np.int64:
                    batch[key] = val.astype(np.int32)

        output = self.get_batch(batch, task)
        self.step_cnt += 1
        return output

    def __len__(self):
        return self.datalen