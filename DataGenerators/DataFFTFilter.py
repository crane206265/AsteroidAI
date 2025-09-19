import numpy as np
from tqdm import tqdm
import gc
import os


# -------------------- Data Loading --------------------

batch_num = 5
data_list = []
data_path_default = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_"
for i in range(batch_num):
    data_list.append(np.load(data_path_default+str(i)+".npy"))

print("[Data shapes]")
for data in data_list:
    print("Data Size :"+str(data.shape))
print("-"*20)

# -------------------- Main --------------------

filtered_data_list = []

for data in data_list:
    filtered_data_temp = np.zeros((1, data.shape[1]))
    for i in tqdm(range((data.shape[0]-1)//800)):
        lc_arr = data[i*800, 800:900]
        fft_coef_zip = np.abs(np.fft.fft(lc_arr))[:lc_arr.shape[0]//2+1]
        fft_coef_zip = np.log10(fft_coef_zip+1e-8)
        log_thr = np.log10(4)#4
        if np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:]):
            filtered_data_temp = np.concatenate((filtered_data_temp, data[i*800:(i+1)*800, :]), axis=0)
    filtered_data_temp[0, 0] = filtered_data_temp.shape[0]
    filtered_data_list.append(filtered_data_temp)
    print("Filtered Dataset Shape : "+str(filtered_data_temp.shape))

for i, data in enumerate(filtered_data_list):
    np.save("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_filtered_"+str(i)+".npy", data)
