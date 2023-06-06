import sys

import cv2

from batch_generator import DatasetSequence
from split_generator import dataset_generator
from utils import get_data_paths


def dataset_gen_sample():
    args = [("/home/semyon/Programming/bachelor_thesis/terrain/unsorted/00.11884.jpg",
             "/home/semyon/Programming/bachelor_thesis/terrain/unsorted/00.11884_mask.png"),
            ("/home/semyon/Programming/bachelor_thesis/terrain/unsorted/01.30800.jpg",
             "/home/semyon/Programming/bachelor_thesis/terrain/unsorted/01.30800_mask.png"),
            ("/home/semyon/Programming/bachelor_thesis/terrain/unsorted/08.99471.jpg",
             "/home/semyon/Programming/bachelor_thesis/terrain/unsorted/08.99471_mask.png")]
    img_gen, mask_gen = dataset_generator(*args)

    for img, mask in zip(img_gen, mask_gen):
        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)


def dataset_from_dir_sample():
    args = get_data_paths("data/water")
    # args = get_data_paths("data/water_small")

    cnt = 0
    for img, mask in dataset_generator(*args, step_x=56, step_y=56):
        cnt += 1
        print('{})'.format(cnt))

        cv2.imshow("img", img)
        cv2.imshow("mask", mask)
        cv2.waitKey(0)
        # cv2.imwrite('data/splitted_water/ex{}.jpg'.format(cnt), img)

    print('total count:', cnt)


def dataset_seq_exapmle(dir_path='data/water_train', batch_size=10):
    seq = DatasetSequence(dir_path, batch_size, input_size=224)

    i = 0
    for imgs, masks in seq:
        print('{})'.format(i))
        print(imgs.shape)
        print(masks.shape)
        print('-' * 30)

        i += 1
        if i > 36:
            break


def main(_):
    dataset_from_dir_sample()
    # prepare_data('data/water')
    # dataset_seq_exapmle()


if __name__ == '__main__':
    main(sys.argv[1:])
