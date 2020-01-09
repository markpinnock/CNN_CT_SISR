import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def dnResNetBlock(nc, inputlayer, upsamp):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputlayer)
    # BN1 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(conv1)
    # BN2 = keras.layers.BatchNormalization()(conv2)
    pool = keras.layers.Conv3D(nc, (2, 2, 1), strides=(2, 2, 1), padding='same', activation='relu')(conv2)
    
    if upsamp == 2:
        skip = keras.layers.Conv3DTranspose(nc, (1, 1, 3), (1, 1, 2), padding='same', activation='relu')(BN2)
        skip = keras.layers.Conv3DTranspose(nc, (1, 1, 3), (1, 1, 2), padding='same', activation='relu')(skip)
    elif upsamp == 1:
        skip = keras.layers.Conv3DTranspose(nc, (1, 1, 3), (1, 1, 2), padding='same', activation='relu')(BN2)
    else:
        skip = conv2

    return skip, pool


def upResNetBlock(nc, inputlayer, skip, tconv_strides):
    tconv = keras.layers.Conv3DTranspose(nc, (2, 2, 2), strides=tconv_strides, padding='same', activation='relu')(inputlayer)
    # BN1 = keras.layers.BatchNormalization()(tconv)
    concat = keras.layers.concatenate([tconv, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(concat)
    # BN2 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(conv1)
    # BN3 = keras.layers.BatchNormalization()(conv2)
    
    return conv2


def UNetGen(input_shape, starting_channels):
    inputlayer = keras.layers.Input(shape=input_shape)
    nc = starting_channels

    skip1, dnres1 = dnResNetBlock(nc, inputlayer, upsamp=2)
    skip2, dnres2 = dnResNetBlock(nc * 2, dnres1, upsamp=1)
    skip3, dnres3 = dnResNetBlock(nc * 4, dnres2, upsamp=0)
    skip4, dnres4 = dnResNetBlock(nc * 8, dnres3, upsamp=0)

    dn5 = keras.layers.Conv3D(nc * 16, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(dnres4)
    # BN5 = keras.layers.BatchNormalization()(dn5)

    upres4 = upResNetBlock(nc * 8, dn5, skip4, (2, 2, 1))
    upres3 = upResNetBlock(nc * 4, upres4, skip3, (2, 2, 1))
    upres2 = upResNetBlock(nc * 2, upres3, skip2, (2, 2, 2))
    upres1 = upResNetBlock(nc, upres2, skip1, (2, 2, 2))

    outputlayer = keras.layers.Conv3D(1, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='sigmoid')(upres1)

    return keras.Model(inputs=inputlayer, outputs=outputlayer)
