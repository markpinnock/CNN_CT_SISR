import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.interpolate as scii
import scipy.stats as stat
import scipy.ndimage as nd
import skimage.metrics as sk


FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/"
EXPT_NAME = "nc8_ep500_eta0.001"
LO_PATH = "Z:/SISR_Data/Toshiba/Sim_Lo/"
HI_PATH = "Z:/SISR_Data/Toshiba/Sim_Hi/"
IMG_SAVE_PATH = f"{FILE_PATH}output_int/"
PRED_PATH = f"{FILE_PATH}output_npy/{EXPT_NAME}/"
lo_imgs = os.listdir(LO_PATH)
lo_imgs.sort()
hi_imgs = os.listdir(HI_PATH)
hi_imgs.sort()
pred_imgs = os.listdir(PRED_PATH)
pred_imgs.sort()

assert len(lo_imgs) == len(hi_imgs)
assert len(hi_imgs) == len(pred_imgs)

x = np.linspace(0, 511, 512)
y = np.linspace(0, 511, 512)
z = np.array([2, 6, 10])
Y, X, Z = np.meshgrid(np.linspace(0, 511, 512), np.linspace(0, 511, 512), np.linspace(0, 11, 12))
q_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
HI_VOL_SHAPE = (512, 512, 12)

lo_range = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]

MSE_ln = []
MSE_nt = []
pSNR_ln = []
pSNR_nt = []
SSIM_ln = []
SSIM_nt = []

for i in range(len(lo_imgs)):

    lin_stem = f"{IMG_SAVE_PATH}{lo_imgs[i][:-6]}_LN.npy"
    lo_vol = nd.gaussian_filter(np.load(f"{LO_PATH}{lo_imgs[i]}"), [0.5, 0.5, 1])
    hi_vol = np.load(f"{HI_PATH}{hi_imgs[i]}")
#    pred_vol = np.load(f"{PRED_PATH}{pred_imgs[i]}")

    rgi_ln = scii.RegularGridInterpolator((x, y, z), lo_vol, method='linear', bounds_error=False, fill_value=None)
    int_ln = rgi_ln(q_points).reshape(HI_VOL_SHAPE)

    MSE_ln.append(sk.mean_squared_error(hi_vol, int_ln))
#    MSE_nt.append(sk.mean_squared_error(hi_vol, pred_vol))
    pSNR_ln.append(sk.peak_signal_noise_ratio(hi_vol, int_ln, data_range=hi_vol.max() - hi_vol.min()))
#    pSNR_nt.append(sk.peak_signal_noise_ratio(hi_vol, pred_vol, data_range=hi_vol.max() - hi_vol.min()))
    SSIM_ln.append(sk.structural_similarity(hi_vol, int_ln))
#    SSIM_nt.append(sk.structural_similarity(hi_vol, pred_vol))

    print(lin_stem[-25:])

    # for j in range(12):
    #     plt.figure(figsize=(18, 9))
    #     plt.subplot(2, 4, 1)
    #     plt.imshow(np.fliplr(lo_vol[:, :, lo_range[j]].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 4, 2)
    #     plt.imshow(np.fliplr(pred_vol[:, :, j].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 4, 3)
    #     plt.imshow(np.fliplr(int_nn[:, :, j].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 4, 4)
    #     plt.imshow(np.fliplr(int_ln[:, :, j].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 4, 5)
    #     plt.imshow(np.fliplr(hi_vol[:, :, j].T), origin='lower', cmap='gray', vmin=0.12, vmax=0.18)
    #     plt.subplot(2, 4, 6)
    #     plt.imshow(np.fliplr(pred_vol[:, :, j].T) - np.fliplr(hi_vol[:, :, j].T), origin='lower', cmap='gray')
    #     plt.subplot(2, 4, 7)
    #     plt.imshow(np.fliplr(int_nn[:, :, j].T) - np.fliplr(hi_vol[:, :, j].T), origin='lower', cmap='gray')
    #     plt.subplot(2, 4, 8)
    #     plt.imshow(np.fliplr(int_ln[:, :, j].T) - np.fliplr(hi_vol[:, :, j].T), origin='lower', cmap='gray')
    #     plt.show()

    # io.savemat(f"Z:/SISR_Data/Toshiba/Mat_Lo/{lo[:-4]}.mat", {'lo': lo_vol})
    # np.save(f"{nn_stem}", int_nn)
    # np.save(f"{lin_stem}", int_ln)

#print(stat.shapiro(MSE_ln))
#print(stat.shapiro(pSNR_ln))
#print(stat.shapiro(SSIM_ln))

#print("Mean, CI")
#print(f"NN: MSE {np.mean(MSE_nn)}, pSNR {np.mean(pSNR_nn)}, SSIM {np.mean(SSIM_nn)}")
print(f"LN: MSE {np.mean(MSE_ln)}, pSNR {np.mean(pSNR_ln)}, SSIM {np.mean(SSIM_ln)}")
#print(f"NT: MSE {np.mean(MSE_nt)}, pSNR {np.mean(pSNR_nt)}, SSIM {np.mean(SSIM_nt)}")

#print(f"NN: MSE {(np.mean(MSE_nn) - 1 * np.std(MSE_nn), np.mean(MSE_nn) + 1 * np.std(MSE_nn))}, "\
#      f"pSNR {(np.mean(pSNR_nn) - 1 * np.std(pSNR_nn), np.mean(pSNR_nn) + 1 * np.std(pSNR_nn))}, "\
#      f"SSIM {(np.mean(SSIM_nn) - 1 * np.std(SSIM_nn), np.mean(SSIM_nn) + 1 * np.std(SSIM_nn))}")
#print(f"LN: MSE {(np.mean(MSE_ln) - 1 * np.std(MSE_ln), np.mean(MSE_ln) + 1 * np.std(MSE_ln))}, "\
#      f"pSNR {(np.mean(pSNR_ln) - 1 * np.std(pSNR_ln), np.mean(pSNR_ln) + 1 * np.std(pSNR_ln))}, "\
#      f"SSIM {(np.mean(SSIM_ln) - 1 * np.std(SSIM_ln), np.mean(SSIM_ln) + 1 * np.std(SSIM_ln))}")
#print(f"NT: MSE {(np.mean(MSE_nt) - 1 * np.std(MSE_nt), np.mean(MSE_nt) + 1 * np.std(MSE_nt))}, "\
#      f"pSNR {(np.mean(pSNR_nt) - 1 * np.std(pSNR_nt), np.mean(pSNR_nt) + 1 * np.std(pSNR_nt))}, "\
#      f"SSIM {(np.mean(SSIM_nt) - 1 * np.std(SSIM_nt), np.mean(SSIM_nt) + 1 * np.std(SSIM_nt))}")
#
#print("Median, q")
#print(f"NN: MSE {np.median(MSE_nn)} {np.quantile(MSE_nn, (0.1, 0.9))}, "\
#      f"pSNR {np.median(pSNR_nn)} {np.quantile(pSNR_nn, (0.1, 0.9))} "\
#      f"SSIM {np.median(SSIM_nn)} {np.quantile(SSIM_nn, (0.1, 0.9))}")
#print(f"LN: MSE {np.median(MSE_ln)} {np.quantile(MSE_ln, (0.1, 0.9))}, "\
#      f"pSNR {np.median(pSNR_ln)} {np.quantile(pSNR_ln, (0.1, 0.9))} "\
#      f"SSIM {np.median(SSIM_ln)} {np.quantile(SSIM_ln, (0.1, 0.9))}")
#print(f"NT: MSE {np.median(MSE_nt)} {np.quantile(MSE_nt, (0.1, 0.9))}, "\
#      f"pSNR {np.median(pSNR_nt)} {np.quantile(pSNR_nt, (0.1, 0.9))} "\
#      f"SSIM {np.median(SSIM_nt)} {np.quantile(SSIM_nt, (0.1, 0.9))}")

# plt.figure()
#
# plt.subplot(3, 3, 1)
# plt.hist(MSE_nn)
# plt.subplot(3, 3, 2)
# plt.hist(MSE_ln)
# plt.subplot(3, 3, 3)
# # plt.hist(MSE_nt)
#
# plt.subplot(3, 3, 4)
# plt.hist(pSNR_nn)
# plt.subplot(3, 3, 5)
# plt.hist(pSNR_ln)
# plt.subplot(3, 3, 6)
# # plt.hist(pSNR_nt)
#
# plt.subplot(3, 3, 7)
# plt.hist(SSIM_nn)
# plt.subplot(3, 3, 8)
# plt.hist(SSIM_ln)
# plt.subplot(3, 3, 9)
# # plt.hist(SSIM_nt)
#
# plt.show()
