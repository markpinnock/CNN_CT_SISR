import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


def dnResNetBlock(nc, inputlayer):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputlayer)
    BN1 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(BN1)
    BN2 = keras.layers.BatchNormalization()(conv2)
    pool = keras.layers.Conv3D(nc, (2, 2, 2), strides=(2, 2, 2), padding='same', activation='relu')(conv2)
    return BN2, pool


def upResNetBlock(nc, inputlayer, skip, tconv_strides):
    tconv = keras.layers.Conv3DTranspose(nc, (2, 2, 2), strides=tconv_strides, padding='same', activation='relu')(inputlayer)
    BN1 = keras.layers.BatchNormalization()(tconv)
    concat = keras.layers.concatenate([BN1, skip], axis=4)
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(concat)
    BN2 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(BN2)
    BN3 = keras.layers.BatchNormalization()(conv2)
    return BN3


def UNetGen():
    inputlayer = keras.layers.Input(shape=(64, 64, 3, 1, ))

    skip1, dnres1 = dnResNetBlock(32, inputlayer)
    skip2, dnres2 = dnResNetBlock(64, dnres1)
    skip3, dnres3 = dnResNetBlock(128, dnres2)
    skip4, dnres4 = dnResNetBlock(256, dnres3)

    dn5 = keras.layers.Conv3D(512, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='relu')(dnres4)
    BN = keras.layers.BatchNormalization()(dn5)

    upres4 = upResNetBlock(256, BN, skip4, (2, 2, 1))
    upres3 = upResNetBlock(128, upres4, skip3, (2, 2, 2))
    upres2 = upResNetBlock(64, upres3, skip2, (2, 2, 2))
    upres1 = upResNetBlock(32, upres2, skip1, (2, 2, 2))

    outputlayer = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='sigmoid')(upres1)

    return keras.Model(inputs=inputlayer, outputs=outputlayer)
