import numpy as np
from PIL import Image
from tinyms.data import GeneratorDataset
from tinyms.vision._transform_ops import *


class CycleGanDatasetTransform:
    r'''
    CycleGan dataset transform class.

    Inputs:
        img (Union[numpy.ndarray, PIL.Image]): Image to be transformed in city_scape.

    Outputs:
        numpy.ndarray, transformed image.

    Examples:
        >>> from PIL import Image
        >>> from tinyms.vision import CycleGanDatasetTransform
        >>>
        >>> cyclegan_transform = CycleGanDatasetTransform()
        >>> img = Image.open('example.jpg')
        >>> img = cyclegan_transform(img)
    '''

    def __init__(self):
        self.random_resized_crop = RandomResizedCrop(256, scale=(0.5, 1.0), ratio=(0.75, 1.333))
        self.random_horizontal_flip = RandomHorizontalFlip(prob=0.5)
        self.resize = Resize((256, 256))
        self.normalize = Normalize(mean=[0.5 * 255] * 3, std=[0.5 * 255] * 3)

    def __call__(self, img):
        if not isinstance(img, (np.ndarray, Image.Image)):
            raise TypeError("Input type should be numpy.ndarray or PIL.Image, got {}.".format(type(img)))
        img = self.resize(img)
        img = self.normalize(img)
        img = hwc2chw(img)

        return img

    def apply_ds(self, gan_generator_ds, repeat_size=1, batch_size=1,
                 num_parallel_workers=1, shuffle=True, phase='train',use_S=False):
        r'''
        Apply preprocess operation on GeneratorDataset instance.

        Args:
            gan_generator_ds (data.GeneratorDataset): GeneratorDataset instance.
            repeat_size (int): The repeat size of dataset. Default: 1.
            batch_size (int): Batch size. Default: 32.
            num_parallel_workers (int): The number of concurrent workers. Default: 1.
            shuffle (bool): Specifies if applying shuffle operation. Default: True.
            phase (str): Specifies the current phase. Default: train.

        Returns:
            data.GeneratorDataset, the preprocessed GeneratorDataset instance.

        Examples:
            >>> from tinyms.vision import CycleGanDatasetTransform
            >>>
            >>> cyclegan_transform = CycleGanDatasetTransform()
            >>> gan_generator_ds = cyclegan_transform.apply_ds(gan_generator_ds)

        Raises:
            TypeError: If `gan_generator_ds` is not instance of GeneratorDataset.
        '''
        if not isinstance(gan_generator_ds, GeneratorDataset):
            raise TypeError("Input type should be GeneratorDataset, got {}.".format(type(gan_generator_ds)))

        trans_func = []
        if phase == 'train':
            if shuffle:
                trans_func += [self.random_resized_crop, self.random_horizontal_flip, self.normalize, hwc2chw]
            else:
                trans_func += [self.resize, self.normalize, hwc2chw]

            # apply transform functions on gan_generator_ds dataset
            gan_generator_ds = gan_generator_ds.map(operations=trans_func,
                                                    input_columns=["image_A"],
                                                    num_parallel_workers=num_parallel_workers)
            gan_generator_ds = gan_generator_ds.map(operations=trans_func,
                                                    input_columns=["image_B"],
                                                    num_parallel_workers=num_parallel_workers)
            if use_S:
                gan_generator_ds = gan_generator_ds.map(operations=trans_func,
                                                        input_columns=["image_S"],
                                                        num_parallel_workers=num_parallel_workers)
        else:
            trans_func += [self.resize, self.normalize, hwc2chw]
            gan_generator_ds = gan_generator_ds.map(operations=trans_func,
                                                    input_columns=["image"],
                                                    num_parallel_workers=num_parallel_workers)
        gan_generator_ds = gan_generator_ds.batch(batch_size, drop_remainder=True)
        gan_generator_ds = gan_generator_ds.repeat(repeat_size)
        return gan_generator_ds


cyclegan_transform = CycleGanDatasetTransform()
