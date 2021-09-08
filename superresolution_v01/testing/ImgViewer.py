import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os


DATA_PATH = "Z:/SISR_Data/Toshiba/"
FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/"
PRED_PATH = f"{FILE_PATH}output_npy/nc8_ep500_eta0.001/"
INT_PATH = f"{FILE_PATH}output_int/"
RF_PATH = f"{FILE_PATH}output_rf/"
SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/images/Comparisons/"
hi_list = os.listdir(f"{DATA_PATH}Real_Hi/")
lo_list = os.listdir(f"{DATA_PATH}Real_Lo/")
pred_list = os.listdir(PRED_PATH)
int_list = os.listdir(INT_PATH)
rf_list = os.listdir(RF_PATH)
cu_list = [vol for vol in int_list]
rf_list = [vol for vol in rf_list]
hi_list.sort()
lo_list.sort()
pred_list.sort()
cu_list.sort()
rf_list.sort()

lo_range = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

for i in range(35):
    hi = np.load(f"{DATA_PATH}Real_Hi/{hi_list[i]}")
    lo = np.load(f"{DATA_PATH}Real_Lo/{lo_list[i]}")
    pred = np.load(f"{PRED_PATH}{pred_list[i]}")
    cu, _ = nrrd.read(f"{INT_PATH}{cu_list[i]}")
    rf, _ = nrrd.read(f"{RF_PATH}{rf_list[i]}")
    
    if i in [0, 1, 2]:
        offset_x, offset_y = 210, 220
    elif i in [3, 4, 5]:
        offset_x, offset_y = 320, 350
    elif i in [6, 7, 8]:
        offset_x, offset_y = 180, 210
    elif i in [9, 10, 11, 12]:
        offset_x, offset_y = 330, 290
    elif i in [13, 14, 15, 16]:
        offset_x, offset_y = 150, 260
    elif i in [17, 18, 19, 20]:
        offset_x, offset_y = 200, 270
    elif i in [21, 22, 23, 24]:
        offset_x, offset_y = 210, 280
    elif i in [25, 26, 27, 28, 29, 30]:
        offset_x, offset_y = 260, 270
    elif i in [31, 32, 33, 34]:
        offset_x, offset_y = 130, 250
    else:
        raise ValueError

    for j in range(12):
        temp_x = 512 - offset_x
        temp_y = offset_y
        new_indices_x = [temp_x - 96, temp_x + 96]
        new_indices_y = [temp_y - 96, temp_y + 96]
#        new_indices_x = [0, 512]
#        new_indices_y = [0, 512]

        fig, axs = plt.subplots(2, 4, figsize=(18, 9))

        axs[0, 0].imshow(np.fliplr(lo[:, :, lo_range[j]].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[0, 0].hlines(y=new_indices_y[0], xmin=512-new_indices_x[0], xmax=512-new_indices_x[1], linewidth=0.75, color='w')
        axs[0, 0].hlines(y=new_indices_y[1], xmin=512-new_indices_x[0], xmax=512-new_indices_x[1], linewidth=0.75, color='w')
        axs[0, 0].vlines(x=512-new_indices_x[0], ymin=new_indices_y[0], ymax=new_indices_y[1], linewidth=0.75, color='w')
        axs[0, 0].vlines(x=512-new_indices_x[1], ymin=new_indices_y[0], ymax=new_indices_y[1], linewidth=0.75, color='w')
        axs[0, 0].axis('off')

        axs[0, 1].imshow(
            np.fliplr(cu[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[0, 1].axis('off')

        axs[0, 2].imshow(
            np.fliplr(rf[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[0, 2].axis('off')

        axs[0, 3].imshow(
            np.fliplr(pred[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[0, 3].axis('off')

        axs[1, 0].imshow(
            np.fliplr(hi[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T), cmap='gray', vmin=0.12, vmax=0.18, origin='lower')
        axs[1, 0].axis('off')

        axs[1, 1].imshow(
            np.abs(np.fliplr(cu[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T) - np.fliplr(hi[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T)), origin='lower', cmap='hot')
        axs[1, 1].axis('off')

        axs[1, 2].imshow(
            np.abs(np.fliplr(rf[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T) - np.fliplr(hi[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T)), origin='lower', cmap='hot')
        axs[1, 2].axis('off')

        axs[1, 3].imshow(np.abs(np.fliplr(pred[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T) - np.fliplr(hi[new_indices_x[0]:new_indices_x[1], new_indices_y[0]:new_indices_y[1], j].T)), origin='lower', cmap='hot')
        axs[1, 3].axis('off')

        img_stem = hi_list[i][:-5]

#        plt.show()
        plt.savefig(f"{SAVE_PATH}{img_stem}{j}.png", dpi=300)
        plt.close()
