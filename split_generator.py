import sys
from itertools import chain
from random import shuffle
from typing import Generator, Callable, Tuple, Iterator

import cv2
import numpy as np

from mask_converters import convert_to_binary_water


def generate_chunks_and_positions(
        img: np.ndarray,
        mask: np.ndarray,
        size_x: int,
        size_y: int,
        step_x: int,
        step_y: int) -> Generator[Tuple[int, int, np.ndarray, np.ndarray], None, None]:
    height, width, _ = img.shape
    assert height == mask.shape[0] and width == mask.shape[1]

    assert height >= step_y and height >= size_y
    assert width >= step_x and width >= size_x

    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            if y + size_y <= height and x + size_x <= width:
                yield x, y, img[y:y + size_y, x:x + size_x], mask[y:y + size_y, x:x + size_x]


def generate_chunks_and_positions_from_file(
        img_path: str,
        mask_path: str,
        size_x: int = 224,
        size_y: int = 224,
        step_x: int = 224,
        step_y: int = 224
):
    img = cv2.imread(img_path)
    mask = cv2.imread(mask_path)
    return generate_chunks_and_positions(
        img,
        mask,
        size_x=size_x,
        size_y=size_y,
        step_x=step_x,
        step_y=step_y
    )


def generate_chunks_from_file(img_path: str,
                              size_x: int = 224,
                              size_y: int = 224,
                              step_x: int = 224,
                              step_y: int = 224) -> Generator[np.ndarray, None, None]:
    img = cv2.imread(img_path)
    return generate_chunks_from_img(img, size_x, size_y, step_x, step_y)


def generate_chunks_from_img(img: np.ndarray,
                             size_x: int = 224,
                             size_y: int = 224,
                             step_x: int = 224,
                             step_y: int = 224) -> Generator[np.ndarray, None, None]:
    height, width, _ = img.shape

    assert height >= step_y and height >= size_y
    assert width >= step_x and width >= size_x

    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            if y + size_y <= height and x + size_x <= width:
                yield img[y:y + size_y, x:x + size_x]


def generate_random_chunks(
        img: np.ndarray,
        mask: np.ndarray,
        chunk_size: int = 4,
        size: float = 0.1) -> Generator[np.ndarray, None, None]:
    height, width, _ = img.shape
    assert height == mask.shape[0] and width == mask.shape[1]

    ys = list(range(height - chunk_size))
    xs = list(range(width - chunk_size))

    shuffle(ys)
    shuffle(xs)

    size_x = int((len(xs) - 1) * size)
    size_y = int((len(xs) - 1) * size)
    xs = xs[0: size_x]
    ys = ys[0: size_y]

    for y in ys:
        for x in xs:
            yield img[y:y + chunk_size, x:x + chunk_size], mask[y:y + chunk_size, x:x + chunk_size]


def data_generator(img_path: str,
                   mask_path: str,
                   size_x: int,
                   size_y: int,
                   step_x: int,
                   step_y: int,
                   mask_converter: Callable[[np.ndarray], np.ndarray] = convert_to_binary_water
                   ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    x_generator = generate_chunks_from_file(img_path, size_x, size_y, step_x, step_y)
    y_generator = map(mask_converter, generate_chunks_from_file(mask_path, size_x, size_y, step_x, step_y))

    return zip(x_generator, y_generator)


def dataset_generator(
        *args: Tuple[str, str],
        size_x: int = 224,
        size_y: int = 224,
        step_x: int = 224,
        step_y: int = 224,
        mask_converter: Callable[[np.ndarray], np.ndarray] = convert_to_binary_water
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    print('creating dataset generator...')
    generators = []
    print(args[:30])
    for idx, (image_path, mask_path) in enumerate(args):
        if idx % 100 == 0:
            print('progress: {}/{}'.format(idx, len(args)))
        generators.append(data_generator(image_path, mask_path, size_x, size_y, step_x, step_y, mask_converter))
    print('Done!')

    return chain(*generators)


def main(_):
    pass


if __name__ == '__main__':
    main(sys.argv[1:])
