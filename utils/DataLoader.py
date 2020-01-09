import numpy as np
import random
import os
import tensorflow as tf


def imgLoader(hi_path, lo_path, hi_list, lo_list, shuffle_flag):
    hi_path = hi_path.decode("utf-8")
    lo_path = lo_path.decode("utf-8")

    if shuffle_flag == True:
        temp_list = list(zip(hi_list, lo_list))
        np.random.shuffle(temp_list)
        hi_list, lo_list = zip(*temp_list)

    N = len(hi_list)
    i = 0 

    while i < N:
        try:
            lo_name = lo_list[i].decode("utf-8")
            lo_vol = np.load(lo_path + lo_name).astype(np.float32)
            # lo_vol = lo_vol[::8, ::8, :]

            hi_name = lo_name[:-5] + "H.npy"
            hi_vol = np.load(hi_path  + hi_name).astype(np.float32)
            # hi_vol = hi_vol[::8, ::8, :]

        except Exception as e:
            print(f"IMAGE LOAD FAILURE: {lo_name} {hi_name} ({e})")

        else:
            yield (hi_vol[:, :, :, np.newaxis], lo_vol[:, :, :, np.newaxis])

        finally:
            i += 1

# Data aug

if __name__ == "__main__":

    FILE_PATH = "Z:/SISR_Data/Toshiba/"
    hi_path = f"{FILE_PATH}/Sim_Hi/"
    lo_path = f"{FILE_PATH}/Sim_Lo/"
    hi_imgs = os.listdir(hi_path)
    lo_imgs = os.listdir(lo_path)
    hi_imgs.sort()
    lo_imgs.sort()

    N = len(hi_imgs)
    NUM_FOLDS = 5
    FOLD = 0
    MB_SIZE = 8
    random.seed(10)

    for i in range(N):
        # print(hi_imgs[i], lo_imgs[i])
        assert hi_imgs[i][:-5] == lo_imgs[i][:-5], "HI/LO PAIRS DON'T MATCH"

    temp_list = list(zip(hi_imgs, lo_imgs))
    random.shuffle(temp_list)
    hi_imgs, lo_imgs = zip(*temp_list)

    for i in range(N):
        # print(hi_imgs[i], lo_imgs[i])
        assert hi_imgs[i][:-5] == lo_imgs[i][:-5], "HI/LO PAIRS DON'T MATCH"

    num_in_fold = int(N / NUM_FOLDS)
    hi_val = hi_imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    lo_val = lo_imgs[FOLD * num_in_fold:(FOLD + 1) * num_in_fold]
    hi_train = hi_imgs[0:FOLD * num_in_fold] + hi_imgs[(FOLD + 1) * num_in_fold:]
    lo_train = lo_imgs[0:FOLD * num_in_fold] + lo_imgs[(FOLD + 1) * num_in_fold:]

    for i in range(len(hi_val)):
        # print(hi_val[i], lo_val[i])
        assert hi_val[i][:-5] == lo_val[i][:-5], "HI/LO PAIRS DON'T MATCH"
    
    for i in range(len(hi_train)):
        # print(hi_train[i], lo_train[i])
        assert hi_train[i][:-5] == lo_train[i][:-5], "HI/LO PAIRS DON'T MATCH"

    print(f"N: {N}, val: {len(hi_val)}, train: {len(hi_train)}, val + train: {len(hi_val) + len(hi_train)}")
    
    train_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[hi_path, lo_path, hi_train, lo_train, True], output_types=(tf.float32, tf.float32))

    val_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[hi_path, lo_path, hi_val, lo_val, False], output_types=(tf.float32, tf.float32))
    
    for data in train_ds.batch(MB_SIZE):
        print(data[0].shape, data[1].shape)
        # pass
    
    for data in val_ds.batch(MB_SIZE):
        print(data[0].shape, data[1].shape)
        # pass
