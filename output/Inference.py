import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('..')

from Networks import UNetGen


TOSHIBA_MIN = -2917
TOSHIBA_MAX = 16297
LO_RES_DIMS = (512, 512, 3, 1, )

img1, _ = nrrd.read("5 4.0 Cryo CTF  CE.nrrd")
img2, _ = nrrd.read("53 4.0 Cryo CTF  CE.nrrd")

img1 = (img1 - TOSHIBA_MIN) / (TOSHIBA_MAX - TOSHIBA_MIN)
img2 = (img2 - TOSHIBA_MIN) / (TOSHIBA_MAX - TOSHIBA_MIN)

UNet = UNetGen(LO_RES_DIMS, 4)
print(UNet.summary())
UNet.load_weights("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/models/nc4_ep10_eta0.001/nc4_ep10_eta0.001.ckpt")

print([var.shape for var in UNet.trainable_variables])
print(UNet.trainable_variables[8].shape)
