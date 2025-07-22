import numpy as np
from tqdm import tqdm
import os
import gc

data_folder_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_preset_rolled/"
info_folder_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/"
save_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/"
batch_size = 3 #GB
batch_size = int(batch_size * 1e+9) # Convert GB to bytes

data_list = [f for f in os.listdir(data_folder_path) if f.__contains__('data_pole_axis_RL_preset_') and not f.__contains__('info')]
info_list = [f for f in os.listdir(info_folder_path) if f.__contains__('data_pole_axis_RL_preset_') and f.__contains__('info')]

data_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
info_list.sort(key=lambda x: int(x.split('_')[-1].split('.')[0].removeprefix('info')))

#print(data_list)
#print(info_list)

RL_preset_data = np.zeros((1, 911), dtype=np.float32)
RL_preset_info_reward0 = np.zeros((1), dtype=np.float32)
RL_preset_info_passed_idx = np.zeros((1))

batch_idx = 0
data_shape_list = []
total_data_memory_size = 0

print("Concatenating data files...")
for i in tqdm(range(len(data_list))):
    data = np.load(data_folder_path + data_list[i])
    info = np.load(info_folder_path + info_list[i])
    gc.collect()
    
    RL_preset_data = np.concatenate((RL_preset_data, data[1:, :].astype(np.float32)), axis=0)
    RL_preset_info_reward0 = np.concatenate((RL_preset_info_reward0, info['reward0'].astype(np.float32)))
    RL_preset_info_passed_idx = np.concatenate((RL_preset_info_passed_idx, info['passed_idx']))

    RL_preset_data[0, 0] = RL_preset_data.shape[0]
    data_memory_size = RL_preset_data.size * RL_preset_data.itemsize
    if data_memory_size >= batch_size:
        save_file_name = save_path + "data_pole_axis_RL_preset_batch_" + str(batch_idx) + ".npy"
        batch_idx += 1
        np.save(save_file_name, RL_preset_data)
        
        print(f"Saved {save_file_name} with shape {RL_preset_data.shape}")
        
        data_shape_list.append(RL_preset_data.shape)
        total_data_memory_size += data_memory_size
        RL_preset_data = np.zeros((1, 911), dtype=np.float32)


RL_preset_info_reward0 = RL_preset_info_reward0[1:]  # Remove the initial zero entry
RL_preset_info_passed_idx = RL_preset_info_passed_idx[1:]   # Remove the initial zero entry

if RL_preset_data.shape[0] > 1:  # If there is remaining data to save
    save_file_name = save_path + "data_pole_axis_RL_preset_batch_" + str(batch_idx) + ".npy"
    np.save(save_file_name, RL_preset_data)

    print(f"Saved {save_file_name} with shape {RL_preset_data.shape}")

    data_shape_list.append(RL_preset_data.shape)
    total_data_memory_size += RL_preset_data.size * RL_preset_data.itemsize

np.save(save_path + "data_pole_axis_RL_preset_passed_idx.npy", RL_preset_info_passed_idx)
np.save(save_path + "data_pole_axis_RL_preset_reward0.npy", RL_preset_info_reward0)

print("Data Shape List :", data_shape_list)
print("Reward0 Shape :", RL_preset_info_reward0.shape)
print("Passed Index Shape :", RL_preset_info_passed_idx.shape)
print("Total Data Memory Size : "+str(int(100*total_data_memory_size/1e+9)/100)+"GB")


