import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import sys
import tensorflow as tf

from NetworkLayers import UNetGen

sys.path.append('..')


# Specify input dims and number of channels
LO_VOL_SIZE = (512, 512, 3, 1, )
NC = 4

# Normalise input image
input_image, _ = nrrd.read('53 4.0 Cryo CTF  CE.nrrd')
input_image = ((input_image - input_image.min()) / (input_image.max() - input_image.min())).astype(np.float32)
input_tensor = tf.convert_to_tensor(input_image[np.newaxis, :, :, :, np.newaxis])

# Generate U-Net
UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)
print(UNet.summary())
UNet.load_weights("C:/Users/roybo/OneDrive - University College London/Collaborations/RobotNeedleSeg/Code/001_CNN_Robotic_Needle_Seg/models/nc4_mb4_ep100_eta0.001/nc4_mb4_ep100_eta0.001.ckpt")
print([var.shape for var in UNet.trainable_variables])

# Forward pass with image
layers = UNet(input_tensor)
print([layer.numpy().shape for layer in layers])
# 256, 128, 64, 32, 32, 64, 128, 256, 512, 512
# Needle tip (230, 276)

# Generate saliency maps
with tf.GradientTape() as tape:
    tape.watch(input_tensor)
    pred = UNet(input_tensor)[9]
    gradients = tape.gradient(pred, input_tensor)

gradients_np = np.squeeze(gradients.numpy())
gradients_np = (gradients_np - gradients_np.min()) / (gradients_np.max() - gradients_np.min())

# Display
fig, axs = plt.subplots(1, 3)
axs[0].imshow(np.fliplr(input_image[:, :, 0].T), cmap='gray', origin='lower')
axs[0].axis('off')
axs[1].imshow(np.fliplr(gradients_np[:, :, 0].T), cmap='gray', origin='lower')
axs[1].axis('off')
axs[2].imshow(np.fliplr(pred[0, :, :, 0, 0].numpy().T), cmap='gray', origin='lower')
axs[2].axis('off')
plt.show()