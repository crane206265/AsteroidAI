
import os

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from Environment import AstEnv
from DataPreprocessing import DataPreProcessing


np.random.seed(1)
random.seed(1)

l_max = 8
merge_num = 3
N_set = (32, 16)
lightcurve_unit_len = 100
data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_total.npz"
model_save_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/checkpoints/"

dataPP = DataPreProcessing(data_path=data_path)
dataPP.X_total = torch.concat((dataPP.X_total[:, :100], dataPP.X_total[:, -9:]), dim=-1)
dataPP.Y_total = dataPP.Y_total[:, 0:(l_max+1)**2]
dataPP.coef2R(dataPP.Y_total, l_max=l_max, N_set=N_set)
dataPP.merge(merge_num=merge_num, ast_repeat_num=10, lc_len=lightcurve_unit_len, dupl_ratio=0.01)
dataPP.X_total = dataPP.X_total.numpy()
dataPP.Y_total = dataPP.Y_total.numpy()
X_total, _, y_total, _ = dataPP.train_test_split(trainset_ratio=0.1)

reward_list = []
max_reward_list = []


reward_domain = [-100, 50]
X_total = X_total[1500:2000, :] #not done


#lc_arr = None
#ell_arr = None
#reward_arr = None
first = False
total_data = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/ell_upto_"+str(reward_domain[0])+".npz")
lc_arr = total_data["lc_arr"]
ell_arr = total_data["ell_arr"]
reward_arr = total_data["reward_arr"]
for i in tqdm(range(X_total.shape[0])):
    if i%30 == 0 and i != 0:
        np.savez("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/ell_upto_"+str(reward_domain[0])+".npz",
                  lc_arr=lc_arr, ell_arr=ell_arr, reward_arr=reward_arr)
    
    env = AstEnv(X_total[i, :-9*merge_num], X_total[i, -9*merge_num:], merge_num, reward_domain, N_set, lightcurve_unit_len)
    
    if env.ell_err:
        continue

    else:
        if first:
            lc_arr = X_total[i, :].copy()
            ell_arr = np.array([env.R_set[0], env.R_set[1], env.R_set[2], env.tilt[0], env.tilt[1]])
            reward_arr = np.array([env.reward0])
            first = False
        else:
            lc_arr = np.vstack((lc_arr, X_total[i, :]))
            ell_arr_temp = np.array([env.R_set[0], env.R_set[1], env.R_set[2], env.tilt[0], env.tilt[1]])
            ell_arr = np.vstack((ell_arr, ell_arr_temp))
            reward_arr = np.concatenate((reward_arr, np.array([env.reward0])), axis=0)
        print("size :", lc_arr.shape, ell_arr.shape, reward_arr.shape)

np.savez("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/ell_upto_"+str(reward_arr[0])+".npz",
                  lc_arr=lc_arr, ell_arr=ell_arr, reward_arr=reward_arr)

    