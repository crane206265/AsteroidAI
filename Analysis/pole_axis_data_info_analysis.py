import matplotlib.pyplot as plt
import numpy as np
import os

"""
data0 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_RL_preset_20.npy")
data = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_preset_rolled/data_pole_axis_RL_preset_rolled_0.npy")
#data = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_2.npy")

print(data0.shape, data.shape)


fig = plt.figure(figsize=(10, 6))
ax11 = fig.add_subplot(4, 2, 1)
ax12 = fig.add_subplot(4, 2, 2)
ax21 = fig.add_subplot(4, 2, 3)
ax22 = fig.add_subplot(4, 2, 4)
ax31 = fig.add_subplot(4, 2, 5)
ax32 = fig.add_subplot(4, 2, 6)
ax41 = fig.add_subplot(4, 2, 7)
ax42 = fig.add_subplot(4, 2, 8)

ax11.plot(data0[1, 800:900], label="original lightcurve", color='royalblue')
ax11.plot(data[1, 800:900], label="rolled lightcurve", color='orange', alpha=0.35)
ax11.legend()
ax12.plot(data[1, 800:900], label="rolled lightcurve", color='orange')
ax21.imshow(data0[1, :800].reshape(40, 20).T)
ax22.imshow(data[1, :800].reshape(40, 20).T)
ax31.imshow(data0[1:801, -1].reshape(40, 20).T)
ax32.imshow(data[1:801, -1].reshape(40, 20).T)
ax41.imshow(data0[1:801, -5].reshape(40, 20).T)
ax42.imshow(data[1:801, -5].reshape(40, 20).T)

'''
ax11.set_title("Lightcurve Comparison")
ax12.set_title("Rolled Lightcurve")
ax21.set_title("Original R_arr")
ax22.set_title("Rolled R_arr")
ax31.set_title("Original Reward Map")
ax32.set_title("Rolled Reward Map")
ax41.set_title("Original $\phi$ Action Map")
ax42.set_title("Rolled $\phi$ Action Map")
'''

plt.show()
"""

passed_idx = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_passed_idx.npy")
reward0 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_reward0.npy")

final_idx = 997

full_idx = np.arange(0, 997)
generated_idx = np.setdiff1d(full_idx, passed_idx)

print("Generated indicies : ")
print("[ ", end="")
ast_idx_temp = 0
for i in generated_idx:
    if i // 25 != ast_idx_temp:
        print("\n"+str(i)+", ", end="")
    else:
        print(str(i)+", ", end="")
    ast_idx_temp = i // 25
print("]")


generated_idx_hist, _, _ = plt.hist(generated_idx, bins=np.arange(0, final_idx, 25), color='orange', alpha=0.5, label='Map Generated Indices')
plt.title('Map-Generated Indices')
plt.plot([0-5, final_idx+5], np.mean(generated_idx_hist)*np.ones((2)), color='black', linestyle='--', linewidth=1)
#plt.show()

print("--------------------")
print("Total number of Generated Indices : "+str(len(generated_idx))+" / "+str(final_idx))
print("Rate (Used/Total) = "+str(int(100*100*len(generated_idx)/final_idx)/100)+"%")

file_list = [f for f in os.listdir("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/") if f.__contains__('data_pole_axis_RL_preset_batch_')]
total_data_byte = sum([os.path.getsize("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/"+f) for f in file_list])
print("Total data size : "+str(int(100*total_data_byte*1e-9)/100)+"GB")

plt.show()
