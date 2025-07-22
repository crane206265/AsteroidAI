import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import gc

folder_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/"
save_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_preset_rolled/"

file_list = os.listdir(folder_path)
data_list = [f for f in file_list if f.__contains__('data_pole_axis_RL_preset_') and not f.__contains__('info')]


# Roll the data by lc_max point

for i in tqdm(range(len(data_list))):
    preset_temp = np.load(folder_path + data_list[i])
    obs_set_num = (preset_temp.shape[0]-1)//800
    gc.collect()

    for j in range(obs_set_num):
        data_temp = preset_temp[j*800+1, :]
        lc_max_idx = np.argmax(data_temp[800:900])

        if lc_max_idx != 0:
            preset_temp[j*800+1:(j+1)*800+1, 800:900] = np.roll(preset_temp[j*800+1:(j+1)*800+1, 800:900], -lc_max_idx, axis=1)

            r_arr = preset_temp[j*800+1:(j+1)*800+1, :800].reshape(-1, 40, 20)
            r_arr = np.roll(r_arr, -lc_max_idx, axis=1)
            preset_temp[j*800+1:(j+1)*800+1, :800] = r_arr.reshape(-1, 800)
        
    save_file_name = save_path + "data_pole_axis_RL_preset_rolled_" + str(i) + ".npy"
    np.save(save_file_name, preset_temp)
    
