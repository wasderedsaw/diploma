#!/home/semyon/anaconda/bin/python3.6
# coding: utf-8
import getopt
import os
import random
import sys

import cv2
import numpy as np
import pandas as pd
from keras import __version__
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, SGD

from batch_generator import preprocess_batch
from metrics import dice_coef_loss, dice_coef
from unet_model import *

def batch_generator(batch_size, next_image):
    while True:
        image_list = []
        mask_list = []
        for i in range(batch_size):
            img, mask = next_image()
            image_list.append(img)
            mask_list.append([mask])

        image_list = np.array(image_list, dtype=np.float32)
        if K.image_dim_ordering() == 'th':
            image_list = image_list.transpose((0, 3, 1, 2))
        image_list = preprocess_batch(image_list)

        mask_list = np.array(mask_list, dtype=np.float32)
        mask_list /= 255.0

        yield image_list, mask_list


def gen_random_image():
    img = np.zeros((input_size, input_size, 3), dtype=np.uint8)
    mask = np.zeros((input_size, input_size), dtype=np.uint8)

    # Background
    dark_color0 = random.randint(0, 100)
    dark_color1 = random.randint(0, 100)
    dark_color2 = random.randint(0, 100)
    img[:, :, 0] = dark_color0
    img[:, :, 1] = dark_color1
    img[:, :, 2] = dark_color2

    # Object
    light_color0 = random.randint(dark_color0 + 1, 255)
    light_color1 = random.randint(dark_color1 + 1, 255)
    light_color2 = random.randint(dark_color2 + 1, 255)
    center_0 = random.randint(0, input_size)
    center_1 = random.randint(0, input_size)
    r1 = random.randint(10, 56)
    r2 = random.randint(10, 56)
    cv2.ellipse(img, (center_0, center_1), (r1, r2), 0, 0, 360, (light_color0, light_color1, light_color2), -1)
    cv2.ellipse(mask, (center_0, center_1), (r1, r2), 0, 0, 360, 255, -1)

    # White noise
    density = random.uniform(0, 0.1)
    for i in range(input_size):
        for j in range(input_size):
            if random.random() < density:
                img[i, j, 0] = random.randint(0, 255)
                img[i, j, 1] = random.randint(0, 255)
                img[i, j, 2] = random.randint(0, 255)

    return img, mask


def train_unet(mode: str, input_size: int):
    if mode == 'new':
        model_creator = get_unet
    elif mode == 'classic':
        model_creator = get_classic_unet
    else:
        raise ValueError('invalid mode')

    epochs = 200
    patience = 20
    batch_size = 16
    optim_type = 'SGD'
    learning_rate = 0.001

    out_model_path = 'pretrained/{}_weights{}.h5'.format(mode, input_size)
    model = model_creator(input_size=input_size)  # My unet

    if os.path.isfile(out_model_path):
        model.load_weights(out_model_path)

    if optim_type == 'SGD':
        optim = SGD(lr=learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    else:
        optim = Adam(lr=learning_rate)
    model.compile(optimizer=optim, loss=dice_coef_loss, metrics=[dice_coef])

    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        # EarlyStopping(monitor='val_loss', patience=patience, verbose=0),
        ModelCheckpoint('weights/tmp/pretrained{}_temp.h5'.format(input_size), monitor='val_loss', save_best_only=True,
                        verbose=1),
    ]

    print('Start training...')
    history = model.fit_generator(
        generator=batch_generator(batch_size, gen_random_image),
        epochs=epochs,
        steps_per_epoch=200,
        validation_data=batch_generator(batch_size, gen_random_image),
        validation_steps=200,
        verbose=1,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('out/pretrained{}.csv'.format(input_size), index=False)
    print('Training is finished (weights and logs are generated )...')


if __name__ == '__main__':
    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            '',
            ['input_size=', 'classic'])
    except Exception as e:
        print(e)
        sys.exit(2)

    opts = dict(opts)
    opts.setdefault('--input_size', 224)

    mode = 'new'
    input_size = 224
    for o in opts:
        if o == '--classic':
            mode = 'classic'
        elif o == '--input_size':
            input_size = int(opts[o])

    if K.backend() == 'tensorflow':
        try:
            from tensorflow import __version__ as __tensorflow_version__

            print('Tensorflow version: {}'.format(__tensorflow_version__))
        except:
            print('Tensorflow is unavailable...')
    else:
        try:
            from theano.version import version as __theano_version__

            print('Theano version: {}'.format(__theano_version__))
        except:
            print('Theano is unavailable...')
    print('Keras version {}'.format(__version__))
    print('Dim ordering:', K.image_dim_ordering())

    train_unet(mode, input_size=input_size)
