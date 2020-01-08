import numpy as np
import os
import tensorflow as tf


def imgLoader(file_path, img_list, shuffle_flag):
    file_path = file_path.decode("utf-8")
    img_list.sort()

    if shuffle_flag == True:
        np.random.shuffle(img_list)

    N = len(img_list)
    i = 0 

    while i < N:
        try:
            lo_name = img_list[i].decode("utf-8")
            lo_vol = np.load(file_path + lo_name)

            hi_name = lo_name[:-6] + "H.nrrd"
            hi_vol = np.load(file_path  + hi_name)

        except Exception as e:
            print(f"IMAGE LOAD FAILURE: {lo_name} {hi_name} ({e})")

        else:
            yield (lo_vol, hi_vol)

        finally:
            i += 1

# Data aug

if __name__ == "__main__":

    FILE_PATH = "C:/Users/rmappin/OneDrive - University College London/PhD/PhD_Prog/promise12-data/"
    file_list = os.listdir(FILE_PATH)
    train_imgs = [img for img in file_list if "image_train" in img]
    train_labs = [lab for lab in file_list if "label_train" in lab]
    test_imgs = [img for img in file_list if "image_test" in img]

    N = len(train_imgs)
    assert N == len(train_labs)

    MB_SIZE = 8

    train_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[FILE_PATH, train_imgs, True], output_types=tf.float32)

    test_ds = tf.data.Dataset.from_generator(
        imgLoader, args=[FILE_PATH, test_imgs, False], output_types=tf.float32)
    
    for data in train_ds.batch(MB_SIZE):
        print(data.shape)
    
    for data in test_ds.batch(MB_SIZE):
        print(data.shape)