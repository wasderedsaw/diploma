import os
import re
from collections import defaultdict
from typing import Set, Optional, Any, Callable

import cv2

from mask_converters import *
from mask_generators import run_old_methods, run_unet
from metrics import class_jacard_index


def calc_metrics(
        img_name: str,
        methods: Set[str]
) -> Dict[str, Dict[str, Optional[float]]]:
    mask = cv2.imread('comparing/{}_mask.png'.format(img_name))
    res = {}
    for method in methods:
        pred = cv2.imread('comparing/{}_pred_{}.png'.format(img_name, method))

        # print('{}:'.format(method))
        cur_scores = {}
        for t in TO_BIN_CONVERTERS:
            cur_conv = TO_BIN_CONVERTERS[t]
            score = class_jacard_index(mask, pred, cur_conv)
            cur_scores[t] = score if score != 1.0 else None
            # print('\t{}: {}'.format(t, score if score != 1.0 else "not present"))
        res[method] = cur_scores
    return res


def calc_mean_metrics(
        data_path: str,
        methods: Set[str]
) -> Dict[str, Dict[str, float]]:
    all_imgs_scores = []

    def append_metrics(img_name):
        all_imgs_scores.append(calc_metrics(img_name, methods))

    apply_to_each_img(append_metrics, data_path)

    res_sum, res_cnt = defaultdict(lambda: defaultdict(float)), defaultdict(lambda: defaultdict(int))
    for scores in all_imgs_scores:
        for method in scores:
            method_scores = scores[method]
            for class_name in method_scores:
                cur_score = method_scores[class_name]
                res_sum[method][class_name] += cur_score if cur_score is not None else 0
                res_cnt[method][class_name] += 1 if cur_score is not None else 0

    res = defaultdict(dict)
    for method in res_sum:
        method_scores = res_sum[method]
        for class_name in method_scores:
            if res_cnt[method][class_name] != 0:
                res[class_name][method] = res_sum[method][class_name] / res_cnt[method][class_name]
    return res


def apply_to_each_img(fun: Callable[[str], Any],
                      path: str = 'comparing'):
    for _, _, files in os.walk(path):
        for filename in files:
            if 'mask' in filename or 'pred' in filename:
                continue
            img_name = re.sub('.jpg', '', filename)
            print('processing {}...'.format(img_name))
            fun(img_name)


def gen_scores_table(source_path: str, dst_path: str):
    scores = calc_mean_metrics(data_path=source_path, methods=methods)
    with open(dst_path, 'w') as file:
        classes = ';'.join([method for method in methods])
        file.write('class/method;{}\n'.format(classes))
        for class_name in scores:
            class_scores = scores[class_name]
            file.write('{}'.format(class_name))
            for method in class_scores:
                file.write(';{:.4f}'.format(class_scores[method]) if method in class_scores else ';-')
            file.write('\n')


if __name__ == '__main__':
    methods = {
        'unet',
        # 'svm',
        # 'rtrees',
        # 'mlp',
        # 'knearest',
        # 'boost'
    }

    def predict_old(img_name: str):
        run_old_methods(img_name, out_path='comparing')

    def predict_unet(img_name: str):
        run_unet(img_name, mode='new', out_path='comparing')

    apply_to_each_img(fun=predict_unet)

    gen_scores_table('comparing', 'out/scores_unet.csv')
