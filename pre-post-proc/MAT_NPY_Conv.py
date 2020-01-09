from argparse import ArgumentParser
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy.io as sio


parser = ArgumentParser()
parser.add_argument('--file_path', '-f', help="File path", type=str)
parser.add_argument('--save_path', '-s', help='Save path', type=str)
parser.add_argument('--review', '-r', help="Review vols y/n", type=str, nargs='?', const='n', default='n')
arguments = parser.parse_args()

if arguments.file_path == None:
    raise ValueError("Must provide file path")
else:
    file_path = arguments.file_path + '/'

if arguments.save_path == None:
    raise ValueError("Must provide save path")
else:
    save_path = arguments.save_path + '/'

if arguments.review == 'y':
    review_flag = True
else:
    review_flag = False

hi_list = [vol for vol in os.listdir(file_path) if "H.mat" in vol]
lo_list = [vol for vol in os.listdir(file_path) if "L.mat" in vol]
hi_list.sort()
lo_list.sort()

N = len(lo_list)

if len(hi_list) != N:
    raise ValueError(f"Unequal numbers hi and lo vols: {len(hi_list)}, {N}")

for i in range(N):
    hi_mat = sio.loadmat(file_path + hi_list[i])
    hvol = hi_mat['hvol']

    try:
        lo_mat = sio.loadmat(file_path + hi_list[i][:-5] + "L.mat")
    except:
        raise ValueError("Lo vol not found")
    else:
        lvol = lo_mat['lvol']

        if review_flag:
            for i in range(hvol.shape[2]):
                fig, axs = plt.subplots(1, 2)
                axs[0].imshow(np.fliplr(hvol[:, :, i].T), vmin=0.12, vmax=0.18, cmap='gray', origin='lower')
                
                if i < 4:
                    axs[1].imshow(np.fliplr(lvol[:, :, 0].T), vmin=0.12, vmax=0.18, cmap='gray', origin='lower')
                elif i >= 4 and i < 8:
                    axs[1].imshow(np.fliplr(lvol[:, :, 1].T), vmin=0.12, vmax=0.18, cmap='gray', origin='lower')
                else:
                    axs[1].imshow(np.fliplr(lvol[:, :, 2].T), vmin=0.12, vmax=0.18, cmap='gray', origin='lower')

                plt.show()
            
                plt.close()
        
        np.save(save_path + "Sim_Hi/" + hi_list[i][:-4] + ".npy", hvol)
        np.save(save_path + "Sim_Lo/" + lo_list[i][:-4] + ".npy", lvol)
        print(f"{hi_list[i][:-4]}.npy, {lo_list[i][:-4]}.npy CONVERTED")

if len(os.listdir(save_path + "Sim_Hi/")) != len(os.listdir(save_path + "Sim_Lo/")):
    raise ValueError("Unequal number of converted vols")