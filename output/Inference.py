import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

sys.path.append('..')

from Networks import UNetGen


# Min and max voxel intensities, input img dimensions
TOSHIBA_MIN = -2917
TOSHIBA_MAX = 16297
LO_RES_DIMS = (512, 512, 3, 1, )

# Read images and normalise
img1, _ = nrrd.read("5 4.0 Cryo CTF  CE.nrrd")
img2, _ = nrrd.read("53 4.0 Cryo CTF  CE.nrrd")

img1 = (img1 - TOSHIBA_MIN) / (TOSHIBA_MAX - TOSHIBA_MIN)
img2 = (img2 - TOSHIBA_MIN) / (TOSHIBA_MAX - TOSHIBA_MIN)

# Generate U-Net with adjusted feature maps
UNet = UNetGen(LO_RES_DIMS, 4)
print(UNet.summary())
UNet.load_weights("C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/models/nc4_ep10_eta0.001/nc4_ep10_eta0.001.ckpt")

print([var.shape for var in UNet.trainable_variables])
print(UNet.trainable_variables[8].shape)

# Test image forward pass
mb = np.concatenate([img1[np.newaxis, :, :, :, np.newaxis], img2[np.newaxis, :, :, :, np.newaxis]], axis=0).astype(np.float32)
pred = UNet(mb)
pred_img1 = np.squeeze(pred[-1][0, ...].numpy())
pred_img2 = np.squeeze(pred[-1][1, ...].numpy())

# Plot output
fig, axs = plt.subplots(2, 2, figsize=(15, 15))
axs[0, 0].imshow(np.fliplr(img1[:, :, 2].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
axs[0, 1].imshow(np.fliplr(img2[:, :, 1].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
axs[1, 0].imshow(np.fliplr(pred_img1[:, :, 10].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
axs[1, 1].imshow(np.fliplr(pred_img2[:, :, 6].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
plt.axis('off')

layer = pred[1].numpy()

fig = plt.figure(figsize=(15, 15))

for i in range(layer.shape[4]):
    plt.subplot(2, 4, i+1)
    plt.imshow(np.fliplr(layer[0, :, :, 2, i].T), origin='lower', cmap='gray')
    plt.axis('off')

fig = plt.figure(figsize=(15, 15))

for i in range(layer.shape[4]):
    plt.subplot(2, 4, i+1)
    plt.imshow(np.fliplr(layer[1, :, :, 1, i].T), origin='lower', cmap='gray')
    plt.axis('off')

plt.show()