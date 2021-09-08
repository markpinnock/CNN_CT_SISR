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
    
    SAVE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/scripts/rf_models/"
    LO_PATH = "D:/SISR_Data/Toshiba/Mat_Int/"
    HI_PATH = "D:/SISR_Data/Toshiba/Sim_Hi/"
#    LO_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/output_int/"
#    HI_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/output_npy/nc8_ep500_eta0.001/"
    SUBJECTS = sorted(list({sub[0:15] for sub in os.listdir(HI_PATH)}))
    TRAIN_SUBJECTS = SUBJECTS[0:10]
    VAL_SUBJECTS = SUBJECTS[10:]

    # Hyper-params
    IMG_SIZE = [512, 512, 12]
    PATCH_SIZE = [3, 3, 3]
    N_FEATURES = 10
    N_PRIN_COMP = None
    PCA = False
    OVERLAP = [16, 16, 2]
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

    rf = skl.RandomForestRegressor(
            n_estimators=3,
            max_depth=None,
            min_samples_split=2,
            n_jobs=-1,
            verbose=0,
            warm_start=True
            )

    for subject in TRAIN_SUBJECTS:

        # Get list of images for this subject
        print(f"TRAIN: {subject}")
        lo_list = [f for f in os.listdir(LO_PATH) if subject in f]
        hi_list = [f for f in os.listdir(HI_PATH) if subject in f]
        lo_list.sort()
        hi_list.sort()
        N = len(lo_list)
        assert N == len(hi_list)
      
        feat_arr = np.zeros((N * g.shape[1], np.prod(PATCH_SIZE) * N_FEATURES))
        diff_arr = np.zeros((N * g.shape[1], np.prod(PATCH_SIZE)))

        # Add each image to array in turn
        for i in range(N):
    
            hi_img = np.load(f"{HI_PATH}{hi_list[i]}")
            lo_img = io.loadmat(f"{LO_PATH}{lo_list[i]}")["int"]
#            lo_img = interpolate(x, y, z, q_points, lo_img)
            diff_img = hi_img - lo_img
    
            feat_patches = extract_patches(lo_img, g, gauss_sigma=SIGMA, filters=filter_dict)
            diff_patches = extract_patches(diff_img, g, gauss_sigma=SIGMA, filters=None)
            feat_arr[i * g.shape[1]:(i + 1) * g.shape[1], :] = feat_patches
            diff_arr[i * g.shape[1]:(i + 1) * g.shape[1], :] = diff_patches

        # Perform PCA if required
        if PCA:
            C = feat_arr.T @ feat_arr
            D, V = np.linalg.eig(C)
            assert np.isclose(D.imag.sum(), 0.0)
            assert np.isclose(V.imag.sum(), 0.0)
            V = V[:, np.argsort(D)].real

            if not N_PRIN_COMP:
                D = D[np.argsort(D)].real
                D = np.cumsum(D) / np.sum(D)
                N_PRIN_COMP = (D > 1e-3).sum()

            V = V[:, -N_PRIN_COMP:]
            feat_arr = feat_arr @ V

        # Fit Random Forest for this subject and expand forest if not last subject
        rf.fit(feat_arr, diff_arr)
        if TRAIN_SUBJECTS.index(subject) < len(TRAIN_SUBJECTS) - 1: rf.n_estimators += 3
    
    pickle.dump(rf, open(f"{SAVE_PATH}{subject}.sav", 'wb'))
    
    base_MSE = []
    base_pSNR = []
    base_SSIM = []
    val_MSE = []
    val_pSNR = []
    val_SSIM = []
    
    for subject in VAL_SUBJECTS:
        print(f"VALIDATE: {subject}")
        lo_list = [f for f in os.listdir(LO_PATH) if subject in f]
        hi_list = [f for f in os.listdir(HI_PATH) if subject in f]
        lo_list.sort()
        hi_list.sort() 
        N = len(lo_list)
        assert N == len(hi_list)

        feat_arr = np.zeros((N * g.shape[1], np.prod(patch_size) * N_FEATURES))        
        lo_arr = np.zeros((N * g.shape[1], np.prod(patch_size)))
        hi_arr = np.zeros((N * g.shape[1], np.prod(patch_size)))
        
        for i in range(N):
    
            hi_img = np.load(f"{HI_PATH}{hi_list[i]}")
            lo_img = io.loadmat(f"{LO_PATH}{lo_list[i]}")["int"]
#            lo_img = interpolate(x, y, z, q_points, lo_img)
    
            feat_patches = extract_patches(lo_img, g, gauss_sigma=SIGMA, filters=filter_dict)
            lo_patches = extract_patches(lo_img, g, gauss_sigma=SIGMA, filters=None)
            hi_patches = extract_patches(hi_img, g, gauss_sigma=SIGMA, filters=None)
    
            feat_arr[i * g.shape[1]:(i + 1) * g.shape[1], :] = feat_patches
            lo_arr[i * g.shape[1]:(i + 1) * g.shape[1], :] = lo_patches
            hi_arr[i * g.shape[1]:(i + 1) * g.shape[1], :] = hi_patches

        # Perform PCA if required
        if PCA:
            C = feat_arr.T @ feat_arr
            D, V = np.linalg.eig(C)
            assert np.isclose(D.imag.sum(), 0.0)
            assert np.isclose(V.imag.sum(), 0.0)
            V = V[:, np.argsort(D)].real
            V = V[:, -N_PRIN_COMP:]
            feat_arr = feat_arr @ V

        # Fit Random Forest for this subject
        pred = rf.predict(feat_arr) + lo_arr

        base_MSE.append(skim.mean_squared_error(hi_arr, lo_arr))
        base_pSNR.append(skim.peak_signal_noise_ratio(hi_arr, lo_arr, data_range=hi_arr.max() - hi_arr.min()))
        base_SSIM.append(skim.structural_similarity(hi_arr, lo_arr))        
        val_MSE.append(skim.mean_squared_error(hi_arr, pred))
        val_pSNR.append(skim.peak_signal_noise_ratio(hi_arr, pred, data_range=hi_arr.max() - hi_arr.min()))
        val_SSIM.append(skim.structural_similarity(hi_arr, pred))

    print(np.mean(base_MSE), np.mean(base_pSNR), np.mean(base_SSIM))
    print(np.mean(val_MSE), np.mean(val_pSNR), np.mean(val_SSIM))


if __name__ == "__main__":

    main()
