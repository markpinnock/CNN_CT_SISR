import matplotlib.pyplot as plt
import nrrd
import numpy as np
import os
import pickle
import scipy.io as io
import skimage.metrics as skim
import sklearn.ensemble as skl

from rf_utils import interpolate, get_vec_sample_grid, extract_patches


""" Adapted from https://github.com/asindel/VSRF

    Sindel et al. Learning from a handful volumes: MRI resolution enhancement
    with volumetric super-resolution forests. 25th IEEE International Conference
    on Image Processing. 2018 pp 1453-1457 """


def main():
#    print("!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("?????????????????????????")
    SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/scripts/rf_models/"
    OUT_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/"
    LO_PATH = "D:/SISR_Data/Toshiba/Mat_Real_Int/"
#    HI_PATH = "D:/SISR_Data/Toshiba/Sim_Hi/"
#    LO_PATH = "D:/SISR_Data/Toshiba/Real_Lo/"
    HI_PATH = "D:/SISR_Data/Toshiba/Real_Hi/"
    SUBJECTS = sorted(list({sub[0:15] for sub in os.listdir(HI_PATH)}))

    # Hyper-params
    IMG_SIZE = [512, 512, 12]
    PATCH_SIZE = [3, 3, 3]
    N_FEATURES = 10
    N_PRIN_COMP = None
    PCA = False
    OVERLAP = [2, 2, 2]
    SIGMA = [1, 1, 1]

    # Set up arrays needed for interpolation
#    x = np.linspace(0, 511, 512)
#    y = np.linspace(0, 511, 512)
#    z = np.array([2, 6, 10])
#    Y, X, Z = np.meshgrid(np.linspace(0, 511, 512), np.linspace(0, 511, 512), np.linspace(0, 11, 12))
#    q_points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()]).T
    
    # Set up filters
    Dx = np.tile(np.reshape([1, 0, -1], [3, 1, 1]), [1, 3, 3])
    Dy = np.tile(np.reshape([1, 0, -1], [1, 3, 1]), [3, 1, 3])
    Dz = np.tile(np.reshape([1, 0, 0, 0, -1], [1, 1, 5]), [3, 3, 1]) / 4 # Slice thickness
    D2x = np.tile(np.reshape([-1, 0, 2, 0, -1], [5, 1, 1]), [1, 3, 3]) / 2
    D2y = np.tile(np.reshape([-1, 0, 2, 0, -1], [1, 5, 1]), [3, 1, 3]) / 2
    D2z = np.tile(np.reshape([-1, 0, 0, 0, 2, 0, 0, 0, -1], [1, 1, 9]), [3, 3, 1]) / 8 # Slice thickness
    
    if N_FEATURES == 10:
        filter_dict = {"Dx": Dx, "Dy": Dy, "Dz": Dz, "D2x": D2x, "D2y": D2y, "D2z": D2z, "M": True, "phi_xy": True, "phi_zx": True, "phi_zy": True }
    elif N_FEATURES == 7:
        filter_dict = filter_dict = {"Dx": Dx, "Dy": Dy, "Dz": Dz, "D2x": D2x, "D2y": D2y, "D2z": D2z, "M": True, "phi_xy": False, "phi_zx": False, "phi_zy": False }
    elif N_FEATURES == 6:
        filter_dict = filter_dict = {"Dx": Dx, "Dy": Dy, "Dz": Dz, "D2x": D2x, "D2y": D2y, "D2z": D2z, "M": False, "phi_xy": False, "phi_zx": False, "phi_zy": False }
    elif N_FEATURES == 1:
        filter_dict = None
    else:
        raise ValueError

    # Set up sample grid
    g = get_vec_sample_grid(IMG_SIZE, PATCH_SIZE, overlap_step=OVERLAP)
    rf = pickle.load(open(f"{SAVE_PATH}rf_model.sav", 'rb'))
    if PCA: N_PRIN_COMP = rf.n_features_ # TEST
    
    base_MSE = []
    base_pSNR = []
    base_SSIM = []
    test_MSE = []
    test_pSNR = []
    test_SSIM = []
    
    for subject in SUBJECTS:

        # Get list of images for this subject
        print(f"TEST: {subject}")
        lo_list = [f for f in os.listdir(LO_PATH) if subject in f]
        hi_list = [f for f in os.listdir(HI_PATH) if subject in f]
        lo_list.sort()
        hi_list.sort() 
        N = len(lo_list)
        assert N == len(hi_list)
#
#        feat_arr = np.zeros((N * g.shape[1], np.prod(patch_size) * N_FEATURES))        
#        lo_arr = np.zeros((N * g.shape[1], np.prod(patch_size)))
#        hi_arr = np.zeros((N * g.shape[1], np.prod(patch_size)))
        
        # Perform inference on each image in turn
        for i in range(N):
            file_stem = hi_list[i][:-5]
            hi_img = np.load(f"{HI_PATH}{hi_list[i]}")
            lo_img = io.loadmat(f"{LO_PATH}{lo_list[i]}")["int"]
#            lo_img = interpolate(x, y, z, q_points, lo_img)
    
            feat_patches = extract_patches(lo_img, g, gauss_sigma=SIGMA, filters=filter_dict)
            lo_patches = extract_patches(lo_img, g, gauss_sigma=SIGMA, filters=None)
            hi_patches = extract_patches(hi_img, g, gauss_sigma=SIGMA, filters=None)

            # Perform PCA if required - can this be used for single images?
            if PCA:
                C = feat_arr.T @ feat_arr
                D, V = np.linalg.eig(C)
                assert np.isclose(D.imag.sum(), 0.0)
                assert np.isclose(V.imag.sum(), 0.0)
                V = V[:, np.argsort(D)].real
                V = V[:, -N_PRIN_COMP:]
                feat_arr = feat_arr @ V

            pred = rf.predict(feat_patches) + lo_patches
#            pred = np.zeros(lo_patches.shape)
#            q = feat_arr.shape[0] // 4
#
#        for i in range(3):
#            pred[i * q:(i + 1) * q, :] = rf.predict(feat_arr[i * q:(i + 1) * q, :]) + lo_arr[i * q:(i + 1) * q, :]
#
#        pred[3 * q:, :] = rf.predict(feat_arr[3 * q:, :]) + lo_arr[3 * q:, :]
            result = np.zeros(np.prod(IMG_SIZE))
            weights = np.zeros(np.prod(IMG_SIZE))
        
            for i in range(g.shape[1]):
                result[g[:, i]] += pred[i, :]
                weights[g[:, i]] += 1
            
            result[weights > 0] /= weights[weights > 0]
            pred_img = np.reshape(result, IMG_SIZE)

            # Account for missing slices if patch doesn't fit
            for k in range(3):
                n_patch = np.floor((IMG_SIZE[k] - PATCH_SIZE[k]) / OVERLAP[k])
                pos_end = int(n_patch * OVERLAP[k] + PATCH_SIZE[k])

                if n_patch <= IMG_SIZE[k]:

                    if k == 0:
                        pred_img[pos_end:IMG_SIZE[k], :, :] = lo_img[pos_end:IMG_SIZE[k], :, :]
                    elif k == 1:
                        pred_img[:, pos_end:IMG_SIZE[k], :] = lo_img[:, pos_end:IMG_SIZE[k], :]
                    elif k == 2:
                        pred_img[:, :, pos_end:IMG_SIZE[k]] = lo_img[:, :, pos_end:IMG_SIZE[k]]
                    else:
                        raise ValueError

#            nrrd.write(f"{OUT_PATH}output_int/{file_stem}CU.nrrd", lo_img)
            nrrd.write(f"{OUT_PATH}output_rf/{file_stem}RF.nrrd", pred_img)

            base_MSE.append(skim.mean_squared_error(hi_patches, lo_patches))
            base_pSNR.append(skim.peak_signal_noise_ratio(hi_patches, lo_patches, data_range=hi_patches.max() - hi_patches.min()))
            base_SSIM.append(skim.structural_similarity(hi_patches, lo_patches))        
            test_MSE.append(skim.mean_squared_error(hi_patches, pred))
            test_pSNR.append(skim.peak_signal_noise_ratio(hi_patches, pred, data_range=hi_patches.max() - hi_patches.min()))
            test_SSIM.append(skim.structural_similarity(hi_patches, pred))

    print(f"Baseline: {np.median(base_MSE), np.median(base_pSNR), np.median(base_SSIM)}")
    print(f"Baseline: {np.quantile(base_MSE, [0.1, 0.9]), np.quantile(base_pSNR, [0.1, 0.9]), np.quantile(base_SSIM, [0.1, 0.9])}")
    print(f"Random forest: {np.median(test_MSE), np.median(test_pSNR), np.median(test_SSIM)}")
    print(f"Random forest: {np.quantile(test_MSE, [0.1, 0.9]), np.quantile(test_pSNR, [0.1, 0.9]), np.quantile(test_SSIM, [0.1, 0.9])}")


if __name__ == "__main__":

    main()
