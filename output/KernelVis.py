import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import tensorflow.keras as keras

from NetworkLayers import UNetGen

sys.path.append('..')

""" Visualises images that maximally activate CNN kernels """

# Determine input size, number of channels, which layer and which filter to visualise
LO_VOL_SIZE = (512, 512, 3, 1,)
NC = 4
LAYER = 9
FILTER = 0
ETA = 1

# Generate U-Net
UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)
print(UNet.summary())
UNet.load_weights(
    "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/models/nc4_ep10_eta0.001/nc4_ep10_eta0.001.ckpt")
print([var.shape for var in UNet.trainable_variables])
input_image = tf.random.uniform([1, 512, 512, 3, 1], -1, 1)
layers = UNet(input_image)
print([layer.numpy().shape for layer in layers])

# Perform gradient ascent on input noise
for i in range(1):
    input_image = tf.random.uniform([1, 512, 512, 3, 1], -1, 1)

    for j in range(10):
        with tf.GradientTape() as tape:
            tape.watch(input_image)
            output = UNet(input_image)[LAYER]
            loss = tf.reduce_mean(output[:, :, :, :, i])
            grad = tape.gradient(loss, input_image)
            grad_norm = grad / (tf.sqrt(tf.reduce_mean(tf.square(grad))) + 1e-5)
            input_image += grad_norm * ETA
            print(i, j)

    kernel = np.squeeze(input_image.numpy())
    
    # Visualise image that maximally activates kernel
    plt.subplot(1, 1, i + 1)
    plt.imshow(kernel[:, :, 1], cmap='plasma')
    plt.axis('off')

plt.show()
