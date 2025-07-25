import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import gc


save_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data_analysis/similarity/"

data0 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_0.npy")
data1 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_1.npy")
data_train = np.concatenate((data0[1:], data1[1:]), axis=0)
del data0, data1
gc.collect()
data2 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_2.npy")[1:]

print("[Data shapes]")
print("data_train shape : ", data_train.shape)
print("data2 shape : ", data2.shape)
print("-"*20)

sample_idx = [508, 620, 639, 466, 862, 970, 582, 789, 828, 309]
test_img_idx = [2, 4, 5, 1, 8, 9, 3, 6, 7, 0]


def MSE_Roll_r_sim(target_img, ref_img):
    sims = np.zeros((40))
    for i in range(40):
        target_img_roll = np.roll(target_img, i, axis=0)
        sims[i] = np.sqrt(np.mean((target_img_roll - ref_img) ** 2))
    sims = sims * 100

    return np.min(sims), np.argmin(sims)

def plotter(r_arr, lc_arr, idx):
    fig = plt.figure(figsize=(9, 7))
    ax11 = fig.add_subplot(2, 1, 1)
    ax12 = fig.add_subplot(2, 1, 2)

    r_arr_img = ax11.imshow(r_arr.T, vmax=12, vmin=8)
    ax11.set_title("R_arr at idx " + str(idx))
    plt.colorbar(r_arr_img, ax=ax11)
    ax12.plot(lc_arr, label="Lightcurve", color='royalblue')
    ax12.set_title("Lightcurve at idx " + str(idx))

    plt.show()

def r_sim_hist(r_arr, dataset, params, self=False, save=False):
    similarity_arr = np.zeros((dataset.shape[0] // 800))
    for j in tqdm(range(0, dataset.shape[0], 800)):
        if self and j // 800 == params[1]:
            continue
        r_arr_dataset = dataset[j, :800].reshape(40, 20)
        lc_arr_dataset = dataset[j, 800:900]

        mean0 = 5
        r_arr = r_arr * mean0 / np.mean(r_arr)
        r_arr_dataset = r_arr_dataset * mean0 / np.mean(r_arr_dataset)

        similarity, roll_idx = MSE_Roll_r_sim(r_arr, r_arr_dataset)
        similarity_arr[j // 800] = similarity

        #print(f"Sample Set {i}, Train Set {j//800}, Similarity: {similarity:.2f}, Roll Index: {roll_idx}")
    
    sim_hist = plt.hist(similarity_arr, bins=np.linspace(0, 100, 51), color='orange', alpha=0.5, label='Similarity Histogram')
    plt.xticks(np.linspace(0, 100, 11))
    plt.xlabel('Similarity (MSE)')
    plt.ylabel('Frequency')
    if self:
        title = "Self_Similarity_Histogram_for_Test_Image"+str(params[0])
    else:
        title = "Similarity_Histogram_for_Test_Image"+str(params[0])
    plt.title(title + " (Sample Set "+str(params[1])+")")
    
    if save:
        plt.savefig(save_path + title+'.png')
    else:
        plt.show()
    plt.close()

    return sim_hist

for num, i in zip(test_img_idx[:], sample_idx[:]):
    r_arr = data2[i*800, :800].reshape(40, 20)
    lc_arr = data2[i*800, 800:900]

    #plotter(r_arr, lc_arr, i)

    r_sim_hist(r_arr, data_train, (num, i), self=False, save=True)
    r_sim_hist(r_arr, data2, (num, i), self=True, save=True)
    
    
    


