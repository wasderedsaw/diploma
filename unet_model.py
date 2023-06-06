# coding: utf-8

from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.core import SpatialDropout2D, Activation
from keras import backend as K
from keras.layers.merge import concatenate

OUTPUT_MASK_CHANNELS = 1


def double_conv_layer(x, size, dropout, batch_norm):
    if K.image_dim_ordering() == 'th':
        axis = 1
    else:
        axis = 3
    conv = Conv2D(size, (3, 3), padding='same')(x)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    conv = Conv2D(size, (3, 3), padding='same')(conv)
    if batch_norm is True:
        conv = BatchNormalization(axis=axis)(conv)
    conv = Activation('relu')(conv)
    if dropout > 0:
        conv = SpatialDropout2D(dropout)(conv)
    return conv


def get_classic_unet(input_size: int = 224,
                     input_channels: int = 3,
                     dropout_val: float = 0,  # Not actually used, present for compatibility with modified u-net
                     batch_norm: bool = False
                     ):
    if K.image_dim_ordering() == 'th':
        inputs = Input((input_channels, input_size, input_size))
        axis = 1
    else:
        inputs = Input((input_size, input_size, input_channels))
        axis = 3
    filters = 32
    batch_norm = False

    conv_16x = double_conv_layer(inputs, 2 * filters, 0, batch_norm)
    pool_8x = MaxPooling2D(pool_size=(2, 2))(conv_16x)

    conv_8x = double_conv_layer(pool_8x, 4 * filters, 0, batch_norm)
    pool_4x = MaxPooling2D(pool_size=(2, 2))(conv_8x)

    conv_4x = double_conv_layer(pool_4x, 8 * filters, 0, batch_norm)
    pool_2x = MaxPooling2D(pool_size=(2, 2))(conv_4x)

    conv_2x = double_conv_layer(pool_2x, 16 * filters, 0, batch_norm)
    pool_1x = MaxPooling2D(pool_size=(2, 2))(conv_2x)

    conv_1x = double_conv_layer(pool_1x, 32 * filters, 0, batch_norm)

    up_2x = concatenate([UpSampling2D(size=(2, 2))(conv_1x), conv_2x], axis=axis)
    up_conv_2x = double_conv_layer(up_2x, 16 * filters, 0, batch_norm)

    up_4x = concatenate([UpSampling2D(size=(2, 2))(up_conv_2x), conv_4x], axis=axis)
    up_conv_4x = double_conv_layer(up_4x, 8 * filters, 0, batch_norm)

    up_8x = concatenate([UpSampling2D(size=(2, 2))(up_conv_4x), conv_8x], axis=axis)
    up_conv_8x = double_conv_layer(up_8x, 4 * filters, 0, batch_norm)

    up_16x = concatenate([UpSampling2D(size=(2, 2))(up_conv_8x), conv_16x], axis=axis)
    up_conv_16x = double_conv_layer(up_16x, 2 * filters, 0, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_16x)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="CLASSIC_UNET")
    return model


def get_unet(input_size: int = 224,
             input_channels: int = 3,
             dropout_val: float = 0.2,
             batch_norm: bool = True) -> Model:
    if K.image_dim_ordering() == 'th':
        inputs = Input((input_channels, input_size, input_size))
        axis = 1
    else:
        inputs = Input((input_size, input_size, input_channels))
        axis = 3
    filters = 32

    conv_32x = double_conv_layer(inputs, filters, 0, batch_norm)
    pool_16x = MaxPooling2D(pool_size=(2, 2))(conv_32x)

    conv_16x = double_conv_layer(pool_16x, 2 * filters, 0, batch_norm)
    pool_8x = MaxPooling2D(pool_size=(2, 2))(conv_16x)

    conv_8x = double_conv_layer(pool_8x, 4 * filters, 0, batch_norm)
    pool_4x = MaxPooling2D(pool_size=(2, 2))(conv_8x)

    conv_4x = double_conv_layer(pool_4x, 8 * filters, 0, batch_norm)
    pool_2x = MaxPooling2D(pool_size=(2, 2))(conv_4x)

    conv_2x = double_conv_layer(pool_2x, 16 * filters, 0, batch_norm)
    pool_1x = MaxPooling2D(pool_size=(2, 2))(conv_2x)

    conv_1x = double_conv_layer(pool_1x, 32 * filters, 0, batch_norm)

    up_2x = concatenate([UpSampling2D(size=(2, 2))(conv_1x), conv_2x], axis=axis)
    up_conv_2x = double_conv_layer(up_2x, 16 * filters, 0, batch_norm)

    up_4x = concatenate([UpSampling2D(size=(2, 2))(up_conv_2x), conv_4x], axis=axis)
    up_conv_4x = double_conv_layer(up_4x, 8 * filters, 0, batch_norm)

    up_8x = concatenate([UpSampling2D(size=(2, 2))(up_conv_4x), conv_8x], axis=axis)
    up_conv_8x = double_conv_layer(up_8x, 4 * filters, 0, batch_norm)

    up_16x = concatenate([UpSampling2D(size=(2, 2))(up_conv_8x), conv_16x], axis=axis)
    up_conv_16x = double_conv_layer(up_16x, 2 * filters, 0, batch_norm)

    up_32x = concatenate([UpSampling2D(size=(2, 2))(up_conv_16x), conv_32x], axis=axis)
    up_conv_32x = double_conv_layer(up_32x, filters, dropout_val, batch_norm)

    conv_final = Conv2D(OUTPUT_MASK_CHANNELS, (1, 1))(up_conv_32x)
    conv_final = Activation('sigmoid')(conv_final)

    model = Model(inputs, conv_final, name="UNET_MOD")
    return model
