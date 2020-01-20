from argparse import ArgumentParser
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import tensorflow.keras as keras
import tensorflow as tf
import time

sys.path.append('..')
sys.path.append('/home/mpinnock/SISR/010_CNN_SISR/')

from Networks import UNetGen
from utils.DataLoader import imgLoader
from utils.TrainFuncs import trainStep, valStep


# Handle arguments
parser = ArgumentParser()
parser.add_argument('--file_path', '-fp', help="File path", type=str)
parser.add_argument('--data_path', '-dp', help="Data path", type=str)
# parser.add_argument('--data_aug', '-da', help="Data augmentation", action='store_true')
parser.add_argument('--minibatch_size', '-mb', help="Minibatch size", type=int, nargs='?', const=4, default=4)
parser.add_argument('--num_chans', '-nc', help="Starting number of channels", type=int, nargs='?', const=32, default=32)
parser.add_argument('--epochs', '-ep', help="Number of epochs", type=int, nargs='?', const=5, default=5)
parser.add_argument('--folds', '-f', help="Number of cross-validation folds", type=int, nargs='?', const=0, default=0)
parser.add_argument('--crossval', '-c', help="Fold number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--gpu', '-g', help="GPU number", type=int, nargs='?', const=0, default=0)
parser.add_argument('--eta', '-e', help="Learning rate", type=float, nargs='?', const=0.001, default=0.001)
arguments = parser.parse_args()

# Generate file path and data path
if arguments.file_path == None:
    FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/"
else:
    FILE_PATH = arguments.file_path

if arguments.data_path == None:
    DATA_PATH = "Z:/SISR_Data/Toshiba/"
else:
    DATA_PATH = arguments.data_path

# Set hyperparameters
MB_SIZE = arguments.minibatch_size
NC = arguments.num_chans # Number of feature maps in first conv layer
EPOCHS = arguments.epochs
NUM_FOLDS = arguments.folds
FOLD = arguments.crossval
ETA = arguments.eta # Learning rate
NUM_EX = 4 # Number of example images to display

if FOLD >= NUM_FOLDS and NUM_FOLDS != 0:
   raise ValueError("Fold number cannot be greater or equal to number of folds")

GPU = arguments.gpu

# Generate experiment name and save paths
EXPT_NAME = f"nc{NC}_ep{EPOCHS}_eta{ETA}"

if NUM_FOLDS > 0:
    EXPT_NAME += f"_cv{FOLD}"

MODEL_SAVE_PATH = f"{FILE_PATH}models/{EXPT_NAME}/"

if not os.path.exists(MODEL_SAVE_PATH) and NUM_FOLDS == 0:
    os.mkdir(MODEL_SAVE_PATH)

IMAGE_SAVE_PATH = f"{FILE_PATH}images/{EXPT_NAME}/"

if not os.path.exists(IMAGE_SAVE_PATH):
    os.mkdir(IMAGE_SAVE_PATH)

# Open log file
if arguments.file_path == None:
    LOG_SAVE_PATH = f"{FILE_PATH}/"
else:
    LOG_SAVE_PATH = f"{FILE_PATH}reports/"

LOG_SAVE_NAME = f"{LOG_SAVE_PATH}{EXPT_NAME}.txt"

if not os.path.exists(LOG_SAVE_PATH):
    os.mkdir(LOG_SAVE_PATH)

log_file = open(LOG_SAVE_NAME, 'w')

# Find data and check hi and lo pair numbers match, then shuffle
lo_path = f"{DATA_PATH}Sim_Lo/"
hi_path = f"{DATA_PATH}Sim_Hi/"
lo_imgs = os.listdir(lo_path)
hi_imgs = os.listdir(hi_path)
lo_imgs.sort()
hi_imgs.sort()

N = len(lo_imgs)
assert N == len(hi_imgs), "HI/LO IMG PAIRS UNEVEN LENGTHS"

LO_VOL_SIZE = (512, 512, 3, 1, )

random.seed(10)
temp_list = list(zip(hi_imgs, lo_imgs))
random.shuffle(temp_list)
hi_imgs, lo_imgs = zip(*temp_list)

# Set cross validation folds and example images
if NUM_FOLDS == 0:
    hi_train = hi_imgs
    lo_train = lo_imgs
    ex_indices = np.random.choice(len(hi_train), NUM_EX)
    hi_examples = np.array(hi_train)[ex_indices]
    lo_examples = np.array(lo_train)[ex_indices]
    hi_examples = [s.encode("utf-8") for s in hi_examples]
    lo_examples = [s.encode("utf-8") for s in lo_examples]
else:
    num_in_fold = int(N / NUM_FOLDS)
    hi_val = hi_imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    lo_val = lo_imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    hi_train = hi_imgs[0:FOLD * num_in_fold] + hi_imgs[(FOLD + 1) * num_in_fold:]
    lo_train = lo_imgs[0:FOLD * num_in_fold] + lo_imgs[(FOLD + 1) * num_in_fold:]
    ex_indices = np.random.choice(len(hi_val), NUM_EX)
    hi_examples = np.array(hi_val)[ex_indices]
    lo_examples = np.array(lo_val)[ex_indices]
    hi_examples = [s.encode("utf-8") for s in hi_examples]
    lo_examples = [s.encode("utf-8") for s in lo_examples]

# Create dataset
train_ds = tf.data.Dataset.from_generator(
    imgLoader, args=[hi_path, lo_path, hi_train, lo_train, True], output_types=(tf.float32, tf.float32))

if NUM_FOLDS > 0:
    val_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[hi_path, lo_path, hi_val, lo_val, False], output_types=(tf.float32, tf.float32))

# Initialise model
UNet = UNetGen(input_shape=LO_VOL_SIZE, starting_channels=NC)

if arguments.file_path == None:
    print(UNet.summary())

# Create losses
loss = keras.losses.MeanSquaredError()
train_metric = keras.metrics.MeanSquaredError()
val_metric = keras.metrics.MeanSquaredError()
Optimiser = keras.optimizers.Adam(ETA)

# Set start time
start_time = time.time()

# Training
for epoch in range(EPOCHS):
    for hi_vol, lo_vol in train_ds.batch(MB_SIZE):
        trainStep(lo_vol, hi_vol, UNet, Optimiser, loss, train_metric)
    
    # Validation step if required
    if NUM_FOLDS > 0:
        for hi_vol, lo_vol in val_ds.batch(MB_SIZE):
            valStep(lo_vol, hi_vol, UNet, val_metric)

    # Print losses every epoch
    print(f"Epoch: {epoch + 1}, Train Loss: {train_metric.result()}, Val Loss: {val_metric.result()}")
    log_file.write(f"Epoch: {epoch + 1}, Train Loss: {train_metric.result()}, Val Loss: {val_metric.result()}\n")
    train_metric.reset_states()
    val_metric.reset_states()

    # Generate example images and save
    fig, axs = plt.subplots(4, NUM_EX)
    
    for i in range(4):
        for j in range(NUM_EX):
            for data in imgLoader(hi_path.encode("utf-8"), lo_path.encode("utf-8"), [hi_examples[j]], [lo_examples[j]], False):
                hi_vol = data[0]
                lo_vol = data[1]

            pred = UNet(lo_vol[np.newaxis, ...])

            axs[0, j].imshow(np.fliplr(lo_vol[:, :, 1, 0].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
            axs[0, j].axis('off')
            axs[1, j].imshow(np.fliplr(pred[0, :, :, 5, 0].numpy().T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
            axs[1, j].axis('off')
            axs[2, j].imshow(np.fliplr(hi_vol[:, :, 5, 0].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
            axs[2, j].axis('off')
            axs[3, j].imshow(np.fliplr(hi_vol[:, :, 5, 0].T - pred[0, :, :, 5, 0].numpy().T), cmap='hot', origin='lower')
            axs[3, j].axis('off')

    fig.subplots_adjust(wspace=0.025, hspace=0.1)
    plt.savefig(f"{IMAGE_SAVE_PATH}/Epoch_{epoch + 1}.png", dpi=250)
    plt.close()

UNet.save_weights(f"{MODEL_SAVE_PATH}{EXPT_NAME}.ckpt")

log_file.write(f"Time: {(time.time() - start_time) / 60:.2f} min\n")
log_file.close()
