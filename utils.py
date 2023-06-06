import re
import shutil
from os import listdir, makedirs
from os.path import exists
from os.path import isfile, join
from typing import Iterable, Tuple, List, Callable, Optional

import cv2
import numpy as np


def clear_and_create(dir_path: str):
    if exists(dir_path):
        shutil.rmtree(dir_path)
    makedirs(dir_path)


def files_cnt(dir_name: str):
    return len([name for name in listdir('./{}'.format(dir_name))])


def get_name(filename: str,
             img_ext: str = 'jpg',
             mask_suffix: str = '_mask',
             mask_ext: str = 'png') -> str:
    res = re.sub('\.' + img_ext, '', filename)
    res = re.sub(mask_suffix + '\.' + mask_ext, '', res)
    return res


def mask_name(filename: str, mask_suffix: str, mask_ext: str) -> str:
    return '{}{}.{}'.format(filename, mask_suffix, mask_ext)


def origin_name(filename: str, img_ext: str) -> str:
    return '{}.{}'.format(filename, img_ext)


def get_result(dir_path: str,
               filename: str,
               img_ext: str,
               mask_suffix: str,
               mask_ext: str) -> Tuple[str, str]:
    origin_path = join(dir_path, origin_name(filename, img_ext))
    mask_path = join(dir_path, mask_name(filename, mask_suffix, mask_ext))
    return origin_path, mask_path


def get_pure_paths(
        dir_path: str,
        img_ext: str = "jpg",
        mask_suffix: str = "_mask",
        mask_ext: str = "png"
) -> List[str]:
    files = [f for f in listdir(dir_path) if isfile(join(dir_path, f)) and mask_suffix not in f]
    files = map(lambda f: get_name(f, img_ext, mask_suffix, mask_ext), files)
    files = list(set(files))
    return files


# Generates (img_path, mask_path) pairs list from specified folder
def get_data_paths(
        dir_path: str,
        img_ext: str = "jpg",
        mask_suffix: str = "_mask",
        mask_ext: str = "png",
        pair_creator: Optional[Callable[[str], Tuple[str, str]]] = None
) -> List[Tuple[str, str]]:
    files = get_pure_paths(dir_path, img_ext, mask_suffix, mask_ext)

    if pair_creator is None:
        pair_creator = lambda f: get_result(dir_path, f, img_ext, mask_suffix, mask_ext)

    return list(map(pair_creator, files))


def have_diff_cols(img) -> bool:
    height, width = img.shape
    return 0 < cv2.countNonZero(img) < height * width


def create_if_not_exists(dirs_path):
    if not exists(dirs_path):
        makedirs(dirs_path)
        print(dirs_path, 'has been created!')


def prepare_environment():
    create_if_not_exists('data')
    create_if_not_exists('weights')
    create_if_not_exists('weights/tmp')
    create_if_not_exists('out')
    create_if_not_exists('results')
    create_if_not_exists('pretrained')
    create_if_not_exists('statistics')


def view_images(imgs: List[List[np.ndarray]],
                win_names: List[str]):
    LEFT = 37
    RIGHT = 39
    idx = 0
    c = 0
    while c != 27:  # escape
        for img_list, win_name in zip(imgs, win_names):
            cv2.imshow(win_name, img_list[idx])
        c = cv2.waitKeyEx(0)

        if c == LEFT:
            idx = (idx - 1) % len(imgs[0])
        elif c == RIGHT:
            idx = (idx + 1) % len(imgs[0])


def main():
    res = get_data_paths("data/water_overfit_train")
    for r in res:
        print(r)


if __name__ == '__main__':
    main()
