import nrrd
import numpy as np
import os


DATA_PATH = "Z:/SISR_Data/Toshiba/"
FILE_PATH = "C:/Users/roybo/OneDrive - University College London/PhD/PhD_Prog/010_CNN_SISR/"
PRED_PATH = f"{FILE_PATH}output_npy/nc8_ep500_eta0.001/"
INT_PATH = f"{FILE_PATH}output_int/"
hi_list = os.listdir(f"{DATA_PATH}Real_Hi/")
pred_list = os.listdir(PRED_PATH)
int_list = os.listdir(INT_PATH)
nn_list = [vol for vol in int_list if 'NN' in vol]
ln_list = [vol for vol in int_list if 'LN' in vol]

for vol in hi_list:
    img = np.load(f"{DATA_PATH}Real_Hi/{vol}")
    nrrd.write(f"{DATA_PATH}NRRD_Hi/{vol[:-4]}.nrrd", img)

for vol in pred_list:
    img = np.load(f"{PRED_PATH}{vol}")
    nrrd.write(f"{DATA_PATH}NRRD_Pred/{vol[:-4]}.nrrd", img)

for vol in nn_list:
    img = np.load(f"{INT_PATH}{vol}")
    nrrd.write(f"{DATA_PATH}NRRD_NN/{vol[:-4]}.nrrd", img)

for vol in ln_list:
    img = np.load(f"{INT_PATH}{vol}")
    nrrd.write(f"{DATA_PATH}NRRD_LN/{vol[:-4]}.nrrd", img)



