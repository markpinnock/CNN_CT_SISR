import json
import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os


COORD_PATH = "Z:/SISR_Data/Toshiba/"
FILE_PATH = "Z:/SISR_Data/Toshiba/Real_Raw/"
SAVE_HI_PATH =f"{COORD_PATH}Real_Hi/"
SAVE_LO_PATH =f"{COORD_PATH}Real_Lo/"
file_list = os.listdir(FILE_PATH)

TOSHIBA_RANGE = [-2917, 16297]
min_val = TOSHIBA_RANGE[0]
max_val = TOSHIBA_RANGE[1]

with open(f"{COORD_PATH}coords.json", 'r') as infile:
    coords = json.load(infile)

for hi_name, coords in coords.items():

    if len(hi_name) == 25:
        lo_name = f"{hi_name[:-9]}{coords[0]}_L.nrrd"
    elif len(hi_name) == 26:
        lo_name = f"{hi_name[:-10]}{coords[0]}_L.nrrd"

    hi = nrrd.read(f"{FILE_PATH}{hi_name}")[0]
    lo = nrrd.read(f"{FILE_PATH}{lo_name}")[0]
    print((hi.min(), hi.max()), (lo.min(), lo.max()))
    hi = (hi - min_val) / (max_val - min_val)
    lo = (lo - min_val) / (max_val - min_val)
    print((hi.min(), hi.max()), (lo.min(), lo.max()))
    sub_hi = hi[:, :, coords[1][0] - 2:coords[1][1] + 2]
    lo_range = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

    print(hi_name)
    assert sub_hi.shape == (512, 512, 12)

    # for i in range(12):
    #     plt.figure(figsize=(18, 6))
    #     plt.subplot(1, 3, 1)
    #     plt.imshow(np.fliplr(sub_hi[:, :, i].T), origin='lower', cmap='gray')
    #     plt.subplot(1, 3, 2)
    #     plt.imshow(np.fliplr(lo[:, :, lo_range[i]].T), origin='lower', cmap='gray')
    #     plt.subplot(1, 3, 3)
    #     plt.imshow(np.fliplr(sub_hi[:, :, i].T) - np.fliplr(lo[:, :, lo_range[i]].T), origin='lower', cmap='gray')
    #     plt.show()

    # np.save(f"{SAVE_HI_PATH}{hi_name[:-5]}.npy", sub_hi)
    # np.save(f"{SAVE_LO_PATH}{lo_name[:-5]}.npy", lo)
