import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append('..')
sys.path.append('/home/mpinnock/SISR/010_CNN_SISR/')

from training.Networks import UNetGen
from utils.DataLoader import imgLoader


# Generate file path and data path

FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/"
DATA_PATH = "Z:/SISR_Data/Toshiba/"

# Generate experiment name and save paths
EXPT_NAME = "nc8_ep500_eta0.001"
MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"
IMG_SAVE_PATH = f"{FILE_PATH}output_npy/{EXPT_NAME}/"

if not os.path.exists(IMG_SAVE_PATH):
    os.mkdir(IMG_SAVE_PATH)

MB_SIZE = 1
NC = 8

# Find data and check hi and lo pair numbers match
lo_path = f"{DATA_PATH}Real_Lo/"
hi_path = f"{DATA_PATH}Real_Hi/"
lo_imgs = os.listdir(lo_path)
hi_imgs = os.listdir(hi_path)
lo_imgs.sort()
hi_imgs.sort()

N = len(lo_imgs)
assert N == len(hi_imgs), "HI/LO IMG PAIRS UNEVEN LENGTHS"
assert MB_SIZE == 1, "MUST USE MINIBATCH SIZE 1"

LO_VOL_SIZE = (512, 512, 3, 1, )

# Create dataset
test_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[hi_path, lo_path, hi_imgs, lo_imgs, False, True], output_types=(tf.float32, tf.float32))

# Initialise model
UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)
UNet.load_weights(f"{MODEL_SAVE_PATH}nc8_ep500_eta0.001.ckpt")

idx = 0
lo_range = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

for hi, lo in test_ds.batch(MB_SIZE):
    file_stem = f"{IMG_SAVE_PATH}{lo_imgs[idx][:-6]}_O.npy"
    print(file_stem[-30:])

    pred = UNet(lo, training=False).numpy()[0, :, :, :, 0]
    hi_np = hi.numpy()[0, :, :, :, 0]
    lo_np = lo.numpy()[0, :, :, :, 0]

    # for i in range(12):
    #     plt.figure(figsize=(10, 10))
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(np.fliplr(lo_np[:, :, lo_range[i]].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(np.fliplr(hi_np[:, :, i].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(np.fliplr(pred[:, :, i].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(np.fliplr(pred[:, :, i].T) - np.fliplr(lo_np[:, :, lo_range[i]].T), origin='lower', cmap='gray')
    #     plt.show()

    np.save(f"{file_stem}", pred)

    idx += 1
