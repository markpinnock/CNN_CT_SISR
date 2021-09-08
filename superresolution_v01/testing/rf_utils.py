import numpy as np
import scipy.interpolate as scii
import scipy.ndimage as nd


""" Adapted from https://github.com/asindel/VSRF

    Sindel et al. Learning from a handful volumes: MRI resolution enhancement
    with volumetric super-resolution forests. 25th IEEE International Conference
    on Image Processing. 2018 pp 1453-1457 """


def interpolate(x, y, z, q_points, lo_vol):

    HI_VOL_SHAPE = (512, 512, 12)
    lo_vol = nd.gaussian_filter(lo_vol, [1, 1, 1])
    rgi_ln = scii.RegularGridInterpolator((x, y, z), lo_vol, method='linear', bounds_error=False, fill_value=None)
    int_ln = rgi_ln(q_points).reshape(HI_VOL_SHAPE)
    
    return int_ln


def get_vec_sample_grid(img_size, patch_size, overlap_step=[2, 2, 2], border=[0, 0, 0]):
    
    index = np.reshape(np.linspace(0, np.prod(img_size) - 1, np.prod(img_size)), img_size)
    grid = index[0:patch_size[0], 0:patch_size[1], 0:patch_size[2]]

    offset = index[
            border[0]:img_size[0]-patch_size[0]-border[0]:overlap_step[0],
            border[1]:img_size[1]-patch_size[1]-border[1]:overlap_step[1],
            border[2]:img_size[2]-patch_size[2]-border[2]:overlap_step[2]
            ]

    offset = np.reshape(offset, [1, 1, 1, np.prod(offset.shape)])
    grid = np.tile(grid.reshape(patch_size + [1]), [1, 1, 1, np.prod(offset.shape)]) + np.tile(offset, patch_size + [1])
    grid = np.reshape(grid, [np.prod(grid.shape[0:3]), grid.shape[3]])

    return grid.astype(np.int32)


def extract_patches(img, grid, gauss_sigma, filters=None):
    
    eps = 1e-8
    features = []
    
    if filters:

        dx = nd.convolve(img, filters["Dx"])
        dy = nd.convolve(img, filters["Dy"])
        dz = nd.convolve(img, filters["Dz"]) # Adjust for upscaling factor
        features.append(dx)
        features.append(dy)
        features.append(dz)

        features.append(nd.convolve(img, filters["D2x"]))
        features.append(nd.convolve(img, filters["D2y"]))
        features.append(nd.convolve(img, filters["D2z"]))

        if filters["phi_xy"]:

            img = nd.gaussian_filter(img, gauss_sigma)

            dx = nd.convolve(img, filters["Dx"])
            dy = nd.convolve(img, filters["Dy"])
            dz = nd.convolve(img, filters["Dz"]) # Adjust for upscaling factor
    
            features.append(np.arctan(dy / (dx + eps))) # phi_xy
            features.append(np.arctan(dx / (dz + eps))) # phi_zx
            features.append(np.arctan(dy / (dz + eps))) # phi_zy
        
        if filters["M"]:
            
            features.append(np.sqrt(np.square(dx) + np.square(dx) + np.square(dx))) # M

    else:
        features.append(img)

    return np.vstack(patches).T
