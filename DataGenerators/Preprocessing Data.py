import numpy as np
from DataPreprocessing import DataPreProcessing
import torch
import random


np.random.seed(1)
random.seed(1)

l_max = 8
merge_num = 1
N_set = (40, 20)
lightcurve_unit_len = 100
data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_total.npz"

dataPP = DataPreProcessing(data_path=data_path, ell_approx=True)
dataPP.X_total = torch.concat((dataPP.X_total[:, :100], dataPP.X_total[:, -(9+5):]), dim=-1)
dataPP.Y_total = dataPP.Y_total[:, 0:(l_max+1)**2]
dataPP.coef2R(dataPP.Y_total, l_max=l_max, N_set=N_set)
dataPP.merge(merge_num=merge_num, ast_repeat_num=20, lc_len=lightcurve_unit_len, dupl_ratio=1)
dataPP.X_total = dataPP.X_total.numpy()
dataPP.Y_total = dataPP.Y_total.numpy()


X_total = dataPP.X_total[:, :-5]
Y_total = dataPP.Y_total
ell_approx = dataPP.X_total[:, -5:]

print("lc_arr shape:", X_total.shape)
print("r_arr shape:", Y_total.shape)
print("ell_arr shape:", ell_approx.shape)

# Save the processed data
np.savez("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_total_preprocessed.npz",
            lc_arr=X_total, r_arr=Y_total, ell_arr=ell_approx)
