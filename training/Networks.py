import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


""" Generates 3D UNet without batch-normalisation, linear output """


# Encoding block, anisotropic strides used as volume only 3 slices thick"
def dnResNetBlock(nc, inputlayer):
    conv1 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(inputlayer)
    # BN1 = keras.layers.BatchNormalization()(conv1)
    conv2 = keras.layers.Conv3D(nc, (3, 3, 3), strides=(2, 2, 1), padding='same', activation='relu')(conv1)
    # BN2 = keras.layers.BatchNormalization()(conv2)
    
    return conv1, conv2


# Decoding block
def upResNetBlock(nc, inputlayer, skip, tconv_strides):
    tconv = keras.layers.Conv3DTranspose(nc, (2, 2, 2), strides=tconv_strides, padding='same', activation='relu')(inputlayer)
    # BN1 = keras.layers.BatchNormalization()(tconv)
    concat = keras.layers.concatenate([tconv, skip], axis=4)
    conv = keras.layers.Conv3D(nc, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(concat)
    # BN2 = keras.layers.BatchNormalization()(conv1)
    
    return conv


# Volume thickness needs upsampling from 3 (input) to 12 (output)
def upSampModule(skip_in, nc, up_factor):
    if up_factor == 2:
        skip_out = keras.layers.Conv3DTranspose(nc, (1, 1, 3), (1, 1, 2), padding='same', activation='relu')(skip_in)
        skip_out = keras.layers.Conv3DTranspose(nc, (1, 1, 3), (1, 1, 2), padding='same', activation='relu')(skip_out)
    elif up_factor == 1:
        skip_out = keras.layers.Conv3DTranspose(nc, (1, 1, 3), (1, 1, 2), padding='same', activation='relu')(skip_in)
    elif up_factor == 0:
        skip_out = skip_in
    else:
        raise ValueError("WRONG UPSAMP FACTOR")

    return skip_out


# Generate UNet
def UNetGen(input_shape, starting_channels):
    inputlayer = keras.layers.Input(shape=input_shape)
    nc = starting_channels
    # factor = np.ones((2, 128, 128, 3, 8)).astype(np.float32)
    # factor[1, :, :, :, 4] *= 100000
    # factor = tf.convert_to_tensor(factor)

    skip1, dnres1 = dnResNetBlock(nc, inputlayer)
    skip2, dnres2 = dnResNetBlock(nc * 2, dnres1)
    # dnres2_mod = tf.multiply(dnres2, factor)
    skip3, dnres3 = dnResNetBlock(nc * 4, dnres2)
    skip4, dnres4 = dnResNetBlock(nc * 8, dnres3)

    dn5 = keras.layers.Conv3D(nc * 16, (3, 3, 3), strides=(1, 1, 1), padding='same', activation='relu')(dnres4)
    # BN5 = keras.layers.BatchNormalization()(dn5)

    # Up-sampling module needed in last two decoding blocks to generate high-res output
    upres4 = upResNetBlock(nc * 8, dn5, skip4, (2, 2, 1))
    upres3 = upResNetBlock(nc * 4, upres4, skip3, (2, 2, 1))
    upres2 = upResNetBlock(nc * 2, upres3, upSampModule(skip2, nc * 2, 1), (2, 2, 2))
    upres1 = upResNetBlock(nc, upres2, upSampModule(skip1, nc, 2), (2, 2, 2))

    outputlayer = keras.layers.Conv3D(1, (1, 1, 1), strides=(1, 1, 1), padding='same', activation='linear')(upres1)
    output_list = [dnres1, dnres2, dnres3, dnres4, dn5, upres4, upres3, upres2, upres1, outputlayer]

    return keras.Model(inputs=inputlayer, outputs=output_list)
