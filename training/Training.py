import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow.keras as keras
import tensorflow as tf

sys.path.append('..')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import trainStep


FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/promise12-data/"
SAVE_PATH = ""
MB_SIZE = 8
EPOCHS = 1

file_list = os.listdir(FILE_PATH)
lo_imgs = [img for img in file_list if "_L.nrrd" in img]
hi_imgs = [img for img in file_list if "_H.nrrd" in img]

N = len(lo_imgs)
assert N == len(hi_imgs), "HI/LO IMG PAIRS UNEVEN LENGTHS"

train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[FILE_PATH, lo_imgs, True], output_types=tf.float32)

# test_ds = tf.data.Dataset.from_generator(
#     imgLoader, args=[FILE_PATH, test_imgs, False], output_types=tf.float32)

UNet = UNetGen()
print(UNet.summary())

model_loss = keras.losses.MAE()
model_metric = keras.metrics.MeanAbsoluteError()
Optimiser = keras.optimizers.Adam(1e-4)

for epoch in range(EPOCHS):
    for data in train_ds.batch(MB_SIZE):
        lo_vol = np.expand_dims(np.squeeze(data[:, 0, ::2, ::2, ::2]), axis=4)
        hi_vol = np.expand_dims(np.squeeze(data[:, 1, ::2, ::2, ::2]), axis=4)
        trainStep(lo_vol, hi_vol, UNet, Optimiser, model_loss, model_metric)

    print(f"Epoch {epoch}, Loss {model_metric.result():.4f}")
    model_metric.reset_states()

# for imgs in test_ds.batch(MB_SIZE):
#     imgs = np.expand_dims(imgs[:, ::2, ::2, ::2], axis=4)
#     pred = UNet(imgs, training=False)

#     fig = plt.figure(figsize=(3, MB_SIZE))

#     for i in range(pred.shape[0]):    
#         plt.subplot(3, MB_SIZE, i + 1)
#         plt.imshow(imgs[i, :, :, 8, 0], cmap='gray')
#         plt.subplot(3, MB_SIZE, i + 1 + MB_SIZE)
#         plt.imshow(pred[i, :, :, 8, 0], cmap='hot')
#         plt.subplot(3, MB_SIZE, i + 1 + MB_SIZE * 2)
#         plt.imshow(imgs[i, :, :, 8, 0] * pred[i, :, :, 8, 0], cmap='gray')
#         plt.axis('off')
    
#     plt.show()
