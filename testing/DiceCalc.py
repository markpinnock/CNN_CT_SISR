import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import scipy.stats as stat


def diceLoss(pred, mask):
    numer = np.sum(pred * mask) * 2
    denom = np.sum(pred) + np.sum(mask) + 1e-6
    dice = numer / denom

    return dice


FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/output_seg/"
hi_list = os.listdir(f"{FILE_PATH}Hi/")
cu_list = os.listdir(f"{FILE_PATH}CU/")
rf_list = os.listdir(f"{FILE_PATH}RF/")
pred_list = os.listdir(f"{FILE_PATH}O/")
hi_list.sort()
cu_list.sort()
rf_list.sort()
pred_list.sort()

cu_dice = []
rf_dice = []
pred_dice = []

for i in range(len(hi_list)):
    hi = nrrd.read(f"{FILE_PATH}Hi/{hi_list[i]}")[0][0]
    cu = nrrd.read(f"{FILE_PATH}CU/{cu_list[i]}")[0][0]
    rf = nrrd.read(f"{FILE_PATH}RF/{rf_list[i]}")[0][0]
    pred = nrrd.read(f"{FILE_PATH}O/{pred_list[i]}")[0][0]

    cu_dice.append(diceLoss(cu, hi))
    rf_dice.append(diceLoss(rf, hi))
    pred_dice.append(diceLoss(pred, hi))

    # for j in range(12):
    #     plt.subplot(2, 2, 1)
    #     plt.imshow(hi[:, :, j], cmap='gray')
    #     plt.subplot(2, 2, 2)
    #     plt.imshow(hi[:, :, j] - nn[:, :, j], cmap='gray')
    #     plt.subplot(2, 2, 3)
    #     plt.imshow(hi[:, :, j] - ln[:, :, j], cmap='gray')
    #     plt.subplot(2, 2, 4)
    #     plt.imshow(hi[:, :, j] - pred[:, :, j], cmap='gray')
    #     print(i)
    #     plt.pause(0.5)
    #     plt.close()

print(stat.shapiro(cu_dice), stat.shapiro(rf_dice), stat.shapiro(pred_dice))
print(stat.mannwhitneyu(cu_dice, pred_dice))
print(stat.mannwhitneyu(rf_dice, pred_dice))

print("Mean, CI")
print(f"CU: {np.mean(cu_dice)} {[np.mean(cu_dice) - 1 * np.std(cu_dice), np.mean(cu_dice) + 1 * np.std(cu_dice)]}")
print(f"RF: {np.mean(rf_dice)} {[np.mean(rf_dice) - 1 * np.std(rf_dice), np.mean(rf_dice) + 1 * np.std(rf_dice)]}")
print(f"NT: {np.mean(pred_dice)} {[np.mean(pred_dice) - 1 * np.std(pred_dice), np.mean(pred_dice) + 1 * np.std(pred_dice)]}")

print("Median, q")
print(f"CU: Dice {np.median(cu_dice)} {np.quantile(cu_dice, (0.05, 0.95))}")
print(f"RF: Dice {np.median(rf_dice)} {np.quantile(rf_dice, (0.05, 0.95))}")
print(f"NT: Dice {np.median(pred_dice)} {np.quantile(pred_dice, (0.05, 0.95))}")

plt.figure()

plt.subplot(1, 3, 1)
plt.hist(cu_dice)
plt.subplot(1, 3, 2)
plt.hist(rf_dice)
plt.subplot(1, 3, 3)
plt.hist(pred_dice)

plt.show()