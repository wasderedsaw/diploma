import os
import shutil
from os.path import exists
from random import shuffle
from typing import List, Callable

import cv2
from keras import backend as K
from keras.utils import Sequence

from mask_converters import *
from split_generator import dataset_generator
from utils import get_data_paths, files_cnt, have_diff_cols


def preprocess_batch(batch):
    batch /= 256
    batch -= 0.5
    return batch


def copy_from_tmp_folder(tmp_dir_path: str, dst_dir_path: str, indices: List[int]):
    i = 0
    for idx in indices:
        img = cv2.imread('{}/{}_img.jpg'.format(tmp_dir_path, idx))
        mask = cv2.imread('{}/{}_mask.png'.format(tmp_dir_path, idx), 0)

        cv2.imwrite('{}/{}_img.jpg'.format(dst_dir_path, i), img)
        cv2.imwrite('{}/{}_mask.png'.format(dst_dir_path, i), mask)

        i += 1


# deprecated, use dataset_utils instead
def prepare_data(source_path: str,
                 mask_converter: Callable[[np.ndarray], np.ndarray] = convert_to_binary_water,
                 only_distinct: bool = True,
                 test_size: float = 0.2,
                 step_x: int = 224,
                 step_y: int = 224,
                 size_x: int = 224,
                 size_y:int = 224,
                 verbose:bool = True):
    def print_if_verbose(text):
        if verbose:
            print(text)

    tmp_dir_path = '{}_tmp'.format(source_path)
    train_dir_path = '{}_train'.format(source_path)
    test_dir_path = '{}_test'.format(source_path)

    if exists(train_dir_path):
        shutil.rmtree(train_dir_path)
    if exists(test_dir_path):
        shutil.rmtree(test_dir_path)
    if exists(tmp_dir_path):
        shutil.rmtree(tmp_dir_path)

    os.makedirs(tmp_dir_path)
    os.makedirs(train_dir_path)
    os.makedirs(test_dir_path)

    args = get_data_paths(source_path)
    generator = dataset_generator(*args,
                                  size_x=size_x,
                                  size_y=size_y,
                                  step_x=step_x,
                                  step_y=step_y,
                                  mask_converter=mask_converter)


    print_if_verbose('Writing images to tmp folder...')
    # writing all images to tmp folder
    # TODO try to replace with zip infinite generator
    n = 0
    for img, mask in generator:
        if not only_distinct or have_diff_cols(mask):
            cv2.imwrite('{}/{}_img.jpg'.format(tmp_dir_path, n), img)
            cv2.imwrite('{}/{}_mask.png'.format(tmp_dir_path, n), mask)
            n += 1

    print_if_verbose("Done!")

    indices = list(range(n))
    shuffle(indices)
    test_cnt = int(n * test_size)

    train_indices = indices[test_cnt:]
    test_indices = indices[0:test_cnt]

    print_if_verbose("Writing to train folder...")
    copy_from_tmp_folder(tmp_dir_path, train_dir_path, train_indices)
    print_if_verbose("Done!")

    print_if_verbose("Writing to test folder...")
    copy_from_tmp_folder(tmp_dir_path, test_dir_path, test_indices)
    print_if_verbose("Done!")
    
    shutil.rmtree(tmp_dir_path)


class DatasetSequence(Sequence):
    def __init__(self, source_path: str, batch_size: int, input_size: int):
        self.source_path = source_path
        self.batch_size = batch_size
        self.cnt = files_cnt(source_path) // 2
        self.input_size = input_size

    def __len__(self):
        return self.cnt // self.batch_size if self.cnt % self.batch_size == 0 else self.cnt // self.batch_size + 1

    def __getitem__(self, idx):
        i = idx * self.batch_size
        size = self.batch_size if i + self.batch_size < self.cnt else self.cnt - i

        image_list = map(lambda j: cv2.imread('{}/{}_img.jpg'.format(self.source_path, j)), range(i, i + size))
        mask_list = map(lambda j: cv2.imread('{}/{}_mask.png'.format(self.source_path, j), 0), range(i, i + size))
        mask_list = map(lambda x: x.reshape(self.input_size, self.input_size, 1), mask_list)

        image_list = np.array(list(image_list), dtype=np.float32)
        if K.image_dim_ordering() == 'th':
            image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)
        mask_list = np.array(list(mask_list), dtype=np.float32)
        mask_list /= 255.0

        return image_list, mask_list

    def __iter__(self):
        while True:
            for item in (self[i] for i in range(len(self))):
                yield item

    def on_epoch_end(self):
        pass


if __name__ == '__main__':
    prepare_data('data/buildings',
                 only_distinct=True,
                 size_x=64,
                 size_y=64,
                 step_x=16,
                 step_y=16,
                 mask_converter=convert_to_binary_buildings)
    # seq = DatasetSequence('data/water_overfit_train', 2)
    # for i, (img, mask) in zip(range(10), seq):
    #     print('{})'.format(i))
    #     print('\t', img.shape)
    #     print('\t', mask.shape)
    # wrapper = np.vectorize(lambda x: [x])
    # arr = np.array([[1, 2, 3], [4, 5, 6]])
    # print(arr.reshape((1, 2, 3)))
