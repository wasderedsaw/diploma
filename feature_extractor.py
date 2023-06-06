import os
from collections import Counter
from typing import Dict

import cv2
import numpy as np
from os.path import exists

from colors import ColorT, COLOR_2_TYPE
from split_generator import generate_random_chunks
from utils import get_data_paths


def chunk_type(mask_chunk: np.ndarray,
               color2type: Dict[ColorT, int] = COLOR_2_TYPE) -> int:
    counter = Counter()

    for row in mask_chunk:
        for pixel in row:
            counter[tuple(pixel)] += 1

    most_recent_color = counter.most_common(1)[0][0]
    return color2type[most_recent_color]


def chunk_descriptor(img_chunk: np.ndarray) -> np.ndarray:
    height, width, _ = img_chunk.shape
    pixel_sum = np.sum(img_chunk, axis=(0, 1), dtype=np.float32)
    pixel_cnt = height * width
    return pixel_sum / pixel_cnt


def extract_features_from_img(
        img: np.ndarray,
        mask: np.ndarray,
        out_path: str,
        chunk_size: int = 4,
        size: float = 0.02,
        file_mode: str = 'w') -> None:
    height, width, _ = img.shape
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    chunks = generate_random_chunks(img, mask, chunk_size=chunk_size, size=size)

    with open(out_path, file_mode) as file:
        for img_chunk, mask_chunk in chunks:
            chunk_t = chunk_type(mask_chunk)
            chunk_desc1, chunk_desc2, chunk_desc3 = chunk_descriptor(img_chunk)
            file.write('{},{},{},{}\n'.format(chunk_t, chunk_desc1, chunk_desc2, chunk_desc3))


def extract_features(img_path: str,
                     mask_path: str,
                     out_path: str,
                     chunk_size: int = 4,
                     size: float = 0.02,
                     file_mode: str = 'w') -> None:
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    extract_features_from_img(img, mask, out_path, chunk_size, size, file_mode)


def extract_features_from_folder(
        data_path: str,
        out_path: str,
        chunk_size: int = 4,
        size: float = 0.02) -> None:
    if exists(out_path):
        os.remove(out_path)
    data_paths = get_data_paths(data_path)

    for img_path, mask_path in data_paths:
        extract_features(img_path, mask_path, out_path, chunk_size, size, file_mode='a')



if __name__ == '__main__':
    # img = np.array(
    #     [
    #         [[10, 10, 0], [20, 20, 20]],
    #         [[5, 5, 5], [5, 5, 5]]
    #     ]
    # )
    # mask = np.array(
    #     [
    #         [[224, 224, 224], [224, 224, 224]],
    #         [[255, 128, 0], [224, 224, 224]]
    #     ]
    # )
    # print(chunk_descriptor(img))
    # print(chunk_type(mask))
    # extract_features('data/water/00.32953.jpg', 'data/water/00.32953_mask.png', 'out/features.csv')
    extract_features_from_folder('data/old_methods_dataset', 'out/features.csv', size=0.02)