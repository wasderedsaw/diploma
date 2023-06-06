import ntpath
import time
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import re
from keras import Model
from keras import backend as K
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard
from keras.optimizers import Adam
import getopt
import sys
from batch_generator import DatasetSequence, preprocess_batch
from mask_converters import VALID_CLASSES
from metrics import jacard_coef_loss
from utils import prepare_environment, view_images
from unet_model import get_unet, get_classic_unet


def prepare_model(
        model_creator,
        input_size: int = 224,
        input_channels: int = 3,
        dropout_val: float = 0.2,
        batch_norm: bool = True,
        weights_path: Optional[str] = None) -> Model:
    model = model_creator(
        input_size=input_size,
        input_channels=input_channels,
        dropout_val=dropout_val,
        batch_norm=batch_norm,
    )
    if weights_path is not None:
        model.load_weights(weights_path)
    optimizer = Adam()
    model.compile(optimizer=optimizer, loss=jacard_coef_loss, metrics=['accuracy'])

    return model


def fit(model: Model,
        out_model_path,
        out_logs_path,
        out_tmp_weights_path,  # TODO
        train_path,
        test_path,
        input_size: int,
        epochs=10,
        batch_size=2,
        resume=False,
        last_epoch=-1):
    patience = 20

    train_generator = DatasetSequence(train_path, batch_size, input_size)
    test_generator = DatasetSequence(test_path, batch_size, input_size)

    steps_per_epoch = len(train_generator)

    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=5, min_lr=1e-9, epsilon=0.00001, verbose=1,
                          mode='min'),
        EarlyStopping(monitor='loss', patience=patience, verbose=1),
        ModelCheckpoint('%s/temp{epoch:02d}-{loss:.2f}.h5' % out_tmp_weights_path,
                        monitor='loss',
                        save_best_only=True,
                        verbose=1),
        CSVLogger(out_logs_path, append=resume),
        TensorBoard(log_dir='./logs',
                    histogram_freq=0,
                    batch_size=batch_size,
                    write_graph=True,
                    write_grads=False,
                    write_images=True,
                    embeddings_freq=0,
                    embeddings_layer_names=None,
                    embeddings_metadata=None)

    ]

    print('Start training...')
    history = model.fit_generator(
        generator=train_generator,
        validation_data=test_generator,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        initial_epoch=last_epoch + 1 if resume else 0,
        verbose=1,
        callbacks=callbacks)

    model.save_weights(out_model_path)
    pd.DataFrame(history.history).to_csv('out/zf_unet_224_train.csv', index=False)
    print('Training is finished...')


def check_model(model: Model,
                test_path,
                cnt=10):
    # imgs = [cv2.imread('data/imgs/ex1.png'), cv2.imread('data/imgs/ex2.png')]
    imgs = []
    for i in range(cnt):
        print('{}/{}_img.jpg'.format(test_path, i))
        img = cv2.imread('{}/{}_img.jpg'.format(test_path, i))
        imgs.append(img)

    imgs_preprocessed = np.array(imgs, dtype=np.float32)
    if K.image_dim_ordering() == 'th':
        imgs_preprocessed = imgs_preprocessed.transpose((0, 3, 1, 2))
    imgs_preprocessed = preprocess_batch(imgs_preprocessed)

    predicted_masks = model.predict(np.array(imgs_preprocessed))
    print(predicted_masks.shape)

    view_images([list(imgs), list(predicted_masks)], ['origin', 'res'])


def make_plots(source: str):
    df = pd.read_csv(source)

    class_name = ntpath.basename(source)
    class_name = re.sub('\.csv', '', class_name)

    plt.xlabel('epoch')
    plt.ylabel('acc')
    plt.plot(df['epoch'], df['acc'], label='train')
    plt.plot(df['epoch'], df['val_acc'], label='test')
    plt.legend(loc=0)
    plt.savefig('out/{}_acc_plot.png'.format(class_name))
    plt.gcf().clear()

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.plot(df['epoch'], df['loss'], label='train')
    plt.plot(df['epoch'], df['val_loss'], label='test')
    plt.legend(loc=0)
    plt.savefig('out/{}_loss_plot.png'.format(class_name))
    plt.gcf().clear()


def main():
    prepare_environment()
    start_time = time.time()

    try:
        opts, args = getopt.getopt(
            sys.argv[1:],
            '',
            ['batch_size=', 'epochs=', 'weights=', 'pretrained=',
             'train_data=', 'test_data=', 'cnt=',
             'logs=', 'class=', 'input_size=',
             'classic', 'train', 'apply'])
    except Exception as e:
        print(e)
        sys.exit(2)

    opts = dict(opts)
    opts.setdefault('--batch_size', 2)
    opts.setdefault('--epochs', 50)
    opts.setdefault('--cnt', 10)

    train_mode, apply_mode, batch_size, epochs, cnt, input_size = False, False, 2, 50, 10, 224
    logs_path, train_data_path, test_data_path, class_name = None, None, None, None
    out_weights_path, pretrained_weights_path = None, None
    model_creator, mode = get_unet, 'new'
    for o in opts:
        if o == '--class':
            class_name = opts[o]
        elif o == '--input_size':
            input_size = int(opts[o])
        elif o == '--batch_size':
            batch_size = int(opts[o])
        elif o == '--epochs':
            epochs = int(opts[o])
        elif o == '--cnt':
            cnt = int(opts[o])
        elif o == '--weights':
            out_weights_path = opts[o]
        elif o == '--pretrained':
            pretrained_weights_path = opts[o]
        elif o == '--logs':
            logs_path = opts[o]
        elif o == '--train_data':
            train_data_path = opts[o]
        elif o == '--test_data':
            test_data_path = opts[o]
        elif o == '--train':
            train_mode = True
        elif o == '--apply':
            apply_mode = True
        elif o == '--classic':
            mode = 'classic'
            model_creator = get_classic_unet

    if pretrained_weights_path is None:
        pretrained_weights_path = 'pretrained/{}_weights{}.h5'.format(mode, input_size)

    assert train_mode != apply_mode, "please specify exactly one running mode"

    if class_name is None:
        assert out_weights_path is not None, "please specify weights_path"
        assert apply_mode or train_data_path is not None, "please specify train_data_path"
        assert test_data_path is not None, "please specify test data path"
        assert logs_path is not None, "please specify logs path"
    else:
        assert class_name in VALID_CLASSES, "invalid class"

        out_weights_path = 'weights/{}_{}.h5'.format(class_name, mode)
        train_data_path = 'data/{}_train'.format(class_name)
        test_data_path = 'data/{}_test'.format(class_name)
        logs_path = 'out/{}_{}.csv'.format(class_name, mode)

    if train_mode:
        print('loading initial weights from {}'.format(pretrained_weights_path))
        model = prepare_model(
            model_creator=model_creator,
            weights_path=pretrained_weights_path,
            input_size=input_size)
        print('training model on {}, validating on {}'.format(train_data_path, test_data_path))
        fit(model,
            out_model_path=out_weights_path,
            out_logs_path=logs_path,
            out_tmp_weights_path='weights/tmp',
            train_path=train_data_path,
            test_path=test_data_path,
            epochs=epochs,
            batch_size=batch_size,
            input_size=input_size)
        make_plots(logs_path)
    else:
        model = prepare_model(
            model_creator=model_creator,
            weights_path=out_weights_path,
            input_size=input_size)
        check_model(model, test_data_path, cnt=cnt)

    if train_mode:
        print('Weights saved to {}'.format(out_weights_path))
        print('Logs saved to {}'.format(logs_path))
    print('total time: {}h'.format((time.time() - start_time) / 3600.))


if __name__ == '__main__':
    main()
