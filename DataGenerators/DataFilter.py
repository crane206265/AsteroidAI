import numpy as np
from tqdm import tqdm
import gc
import os


# -------------------- Test Data Loading --------------------

data0 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_0.npy")#[1:]
data1 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_1.npy")#[1:]
data2 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_2.npy")#[1:]

print("[Data shapes]")
print("data0 shape : ", data0.shape)
print("data1 shape : ", data1.shape)
print("data2 shape : ", data2.shape)
print("-"*20)

# -------------------- Main --------------------

filtered_data_list = []

for data in [data0, data1, data2]:
    filtered_data_temp = np.zeros((1, data.shape[1]))
    for i in tqdm(range((data.shape[0]-1)//800)):
        lc_arr = data[i*800, 800:900]
        fft_coef_zip = np.abs(np.fft.fft(lc_arr))[:lc_arr.shape[0]//2+1]
        fft_coef_zip = np.log10(fft_coef_zip+1e-8)
        log_thr = np.log10(4)#4
        if np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:]):
            filtered_data_temp = np.concatenate((filtered_data_temp, data[i*800:(i+1)*800, :]), axis=0)
    filtered_data_temp[0, :] = data[0, :]
    filtered_data_list.append(filtered_data_temp)
    print("Filtered Dataset Shape : "+str(filtered_data_temp.shape))

np.save("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_filtered_0.npy"
        , filtered_data_list[0])
np.save("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_filtered_1.npy"
        , filtered_data_list[1])
np.save("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_filtered_2.npy"
        , filtered_data_list[2])
