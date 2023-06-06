from typing import Callable

import numpy as np
from keras import backend as K


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1.0)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jacard_coef_loss(y_true, y_pred):
    return -jacard_coef(y_true, y_pred)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


# noinspection PyTypeChecker
def class_jacard_index(mask_true: np.ndarray,
                       mask_pred: np.ndarray,
                       class_converter: Callable[[np.ndarray], np.ndarray]) -> float:
    bin_mask_true = class_converter(mask_true)
    bin_mask_pred = class_converter(mask_pred)

    assert bin_mask_pred.shape == bin_mask_true.shape, 'mask_true and mask_pred shapes must be equal'
    height, width = bin_mask_true.shape

    bin_mask_true_f = bin_mask_true.reshape(height * width) / 255
    bin_mask_pred_f = bin_mask_pred.reshape(height * width) / 255
    intersection = np.sum(bin_mask_true_f * bin_mask_pred_f)

    res = (intersection + 1.0) / (np.sum(bin_mask_true_f) + np.sum(bin_mask_pred_f) - intersection + 1.0)

    return res
