import numpy as np
import matplotlib.pyplot as plt
import cv2


data2 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_2.npy")[1:]

print("[Data shapes]")
print("data2 shape : ", data2.shape)
print("-"*20)

sample_idx = [508, 620, 639, 466, 862, 970, 582, 789, 828, 309]
test_img_idx = [2, 4, 5, 1, 8, 9, 3, 6, 7, 0]


class RewardMapModifier():
    def __init__(self, extends=(0, 1), blur_coef=(5, 3)):
        self.extends = extends
        self.blur_coef = blur_coef
    
    def extend_hori(self, reward_map, action_maps):
        left_reward = reward_map[..., :, -int(reward_map.shape[-2]*self.extends[1]/2):, :]
        right_reward = reward_map[..., :, :int(reward_map.shape[-2]*self.extends[1]/2), :]
    
        if action_maps is not None:
            left_actions = action_maps[..., :, -int(action_maps.shape[-2]*self.extends[1]/2):, :].copy()
            right_actions = action_maps[..., :, :int(action_maps.shape[-2]*self.extends[1]/2), :].copy()
            left_actions[..., :, :, 0] = left_actions[..., :, :, 0] - 1
            right_actions[..., :, :, 0] = right_actions[..., :, :, 0] + 1
    
        if self.extends[1] != 0:
            extended_reward = np.concatenate((left_reward, reward_map, right_reward), axis=-2)
            extended_actions = np.concatenate((left_actions, action_maps, right_actions), axis=-2) if action_maps is not None else action_maps
        else:
            extended_reward = reward_map
            extended_actions = action_maps
    
        return extended_reward, extended_actions
    
    def extend_vert(self, reward_map, action_maps):
        top_reward = np.flip(reward_map[..., :int(reward_map.shape[-3]*self.extends[0]/2), :, :], -3)
        bottom_reward = np.flip(reward_map[..., -int(reward_map.shape[-3]*self.extends[0]/2):, :, :], -3)
        top_reward = np.roll(top_reward, 20, axis=-2)
        bottom_reward = np.roll(bottom_reward, 20, axis=-2)
        
        if action_maps is not None:
            top_actions = np.flip(action_maps[..., 1:int(action_maps.shape[-3]*self.extends[0]/2), :, :].copy(), -3)
            bottom_actions = np.flip(action_maps[..., -int(action_maps.shape[-3]*self.extends[0]/2):-1, :, :].copy(), -3)
            top_actions[..., :, :, 1] = 2*0 - top_actions[..., :, :, 1]
            bottom_actions[..., :, :, 1] = 2*1 - bottom_actions[..., :, :, 1]
            
        if self.extends[0] != 0:
            extended_reward = np.concatenate((top_reward, reward_map, bottom_reward), axis=-3)
            extended_actions = np.concatenate((top_actions, action_maps, bottom_actions), axis=-3) if action_maps is not None else action_maps
        else:
            extended_reward = reward_map
            extended_actions = action_maps
    
        return extended_reward, extended_actions
    
    def blur(self, reward_map):
        #reward_map = 2.5 * np.tan( reward_map * (np.pi/2) / 6 )\n",
        if len(reward_map.shape) == 3:
            reward_map[:, :, 0] = cv2.GaussianBlur(reward_map[:, :, 0], (self.blur_coef[0], self.blur_coef[0]), self.blur_coef[1])
        elif len(reward_map.shape) == 4:
            for i in range(reward_map.shape[0]):
                reward_map[i, :, :, 0] = cv2.GaussianBlur(reward_map[i, :, :, 0], (self.blur_coef[0], self.blur_coef[0]), self.blur_coef[1])
        reward_map = 6 * (2/np.pi) * np.arctan(reward_map/2)
        return reward_map
    
    def operation(self, reward_map, action_maps, order=['extend_hori', 'extend_vert', 'blur']):
        result_reward = reward_map
        result_action = action_maps
        for op in order:
            if op == 'extend_hori':
                result_reward, result_action = self.extend_hori(result_reward, result_action)
            elif op == 'extend_vert':
                result_reward, result_action = self.extend_vert(result_reward, result_action)
            elif op == 'blur':
                result_reward = self.blur(result_reward)
            else:
                raise NotImplementedError()
        return result_reward, result_action
    
    def ext_N_set(self, N_set):
        return (N_set[0]+2*int(N_set[0]*self.extends[1]/2), N_set[1]+2*int(N_set[1]*self.extends[0]/2))
    

def plotter(r_arr, lc_arr, lc_info, reward_map0, reward_map, idx):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    Sdir = lc_info[0:3]
    Edir = lc_info[3:6]
    Stheta = np.arccos(Sdir[-1]) * 20 / np.pi
    Etheta = np.arccos(Edir[-1]) * 20 / np.pi
    #reward_map = 6*2/np.pi * np.arctan(reward_map * np.pi/2)

    r_arr_img = ax1.imshow(r_arr.T, vmax=8, vmin=12)
    ax1.set_title("R_arr at idx " + str(idx))
    plt.colorbar(r_arr_img, ax=ax1, shrink=0.75)
    ax1.plot([0, 39], [Stheta, Stheta], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax1.plot([0, 39], [Etheta, Etheta], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax1.legend()

    ax2.plot(lc_arr, label="Lightcurve", color='royalblue')
    ax2.set_title("Lightcurve at idx " + str(idx))

    reward_map0_img = ax3.imshow(reward_map0.T, vmax=-6, vmin=6)
    ax3.set_title("Reward_Map0 at idx " + str(idx))
    plt.colorbar(reward_map0_img, ax=ax3, shrink=0.75)
    ax3.plot([0, 39], [Stheta, Stheta], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax3.plot([0, 39], [Etheta, Etheta], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax3.legend()

    reward_map_img = ax4.imshow(reward_map.T, vmax=-6, vmin=6)
    ax4.set_title("Reward_Map_Extended at idx " + str(idx))
    plt.colorbar(reward_map_img, ax=ax4, shrink=0.75)
    ax4.plot([0, 40*(modifier1.extends[1]+1)-1], [Stheta*(modifier1.extends[0]+1), Stheta*(modifier1.extends[0]+1)], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax4.plot([0, 40*(modifier1.extends[1]+1)-1], [Etheta*(modifier1.extends[0]+1), Etheta*(modifier1.extends[0]+1)], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax4.plot([20*modifier1.extends[1], 20*modifier1.extends[1]+40], [10*modifier1.extends[0], 10*modifier1.extends[0]], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.plot([20*modifier1.extends[1], 20*modifier1.extends[1]+40], [10*modifier1.extends[0]+20, 10*modifier1.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.plot([20*modifier1.extends[1], 20*modifier1.extends[1]], [10*modifier1.extends[0], 10*modifier1.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.plot([20*modifier1.extends[1]+40, 20*modifier1.extends[1]+40], [10*modifier1.extends[0], 10*modifier1.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.legend()

    plt.tight_layout()
    plt.show()


sample_idx = np.array(sample_idx)
test_img_idx = np.array(test_img_idx)
sample_idx = sample_idx[np.argsort(test_img_idx)]
#raise NotImplementedError

modifier0 = RewardMapModifier((0, 0), (3, 2))
modifier1 = RewardMapModifier((1, 1), (3, 2))


for num, idx in enumerate(sample_idx):
    r_arr = data2[idx*800, :800].reshape(40, 20)
    lc_arr = data2[idx*800, 800:900]
    lc_info = data2[idx*800, 900:906]
    reward_map0 = data2[idx*800:(idx+1)*800, -1].reshape(40, 20)

    reward_map, _ = modifier0.operation(np.expand_dims(reward_map0.T, axis=-1), None, order=['extend_hori', 'extend_vert', 'blur'])
    reward_map_ext, _ = modifier1.operation(np.expand_dims(reward_map0.T, axis=-1), None, order=['extend_hori', 'extend_vert', 'blur'])
    reward_map = reward_map[:, :, 0].T
    reward_map_ext = reward_map_ext[:, :, 0].T

    plotter(r_arr, lc_arr, lc_info, reward_map, reward_map_ext, num)
