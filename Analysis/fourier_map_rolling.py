import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch


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
        top_reward = np.roll(reward_map[..., :int(reward_map.shape[-3]*self.extends[0]/2), :, :], 20, axis=-2)
        bottom_reward = np.roll(reward_map[..., -int(reward_map.shape[-3]*self.extends[0]/2):, :, :], 20, axis=-2)
        top_reward = np.flip(top_reward, axis=-3)
        bottom_reward = np.flip(bottom_reward, axis=-3)

        if action_maps is not None:
            top_actions = np.flip(action_maps[..., :int(action_maps.shape[-3]*self.extends[0]/2), :, :].copy(), -3)
            bottom_actions = np.flip(action_maps[..., -int(action_maps.shape[-3]*self.extends[0]/2):, :, :].copy(), -3)
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
        #reward_map = 6 * 2*(1/(1+np.exp(-reward_map/7)) - 0.5)
        if len(reward_map.shape) == 3:
            reward_map[:, :, 0] = cv2.GaussianBlur(reward_map[:, :, 0], (self.blur_coef[0], self.blur_coef[0]), self.blur_coef[1])
        elif len(reward_map.shape) == 4:
            for i in range(reward_map.shape[0]):
                reward_map[i, :, :, 0] = cv2.GaussianBlur(reward_map[i, :, :, 0], (self.blur_coef[0], self.blur_coef[0]), self.blur_coef[1])
                #max_val = np.max(np.abs(reward_map[i, :, :, 0]))
                #reward_map[i, :, :, 0] = 6 * (2/np.pi) * np.arctan(reward_map[i, :, :, 0]/2) / ((2/np.pi) * np.arctan(max_val/2))
        reward_map = 6 * (2/np.pi) * np.arctan(reward_map/8)
        #reward_map = 6 * 2*(1/(1+np.exp(-reward_map/7)) - 0.5)
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
                if self.blur_coef == (0, 0):
                    reward_map = 6 * 2*(1/(1+np.exp(-reward_map/7)) - 0.5)
                else:
                    result_reward = self.blur(result_reward)
            else:
                raise NotImplementedError()
        return result_reward, result_action

    def ext_N_set(self, N_set):
        return (N_set[0]+2*int(N_set[0]*self.extends[1]/2), N_set[1]+2*int(N_set[1]*self.extends[0]/2))
    
def shifter(img, dx=0, dy=0):
    img_F = np.fft.fft2(img)
    N, M = img.shape
    ky = np.fft.fftfreq(N)[:, None]
    kx = np.fft.fftfreq(M)[None, :]
    phase = np.exp(-2j*np.pi*(kx*dx + ky*dy))
    new_img = np.fft.ifft2(img_F*phase)
    return new_img.real


def plotter(r_arr, reward_map, idx):
    fig = plt.figure(figsize=(10, 6))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    img1 = ax1.imshow(reward_map, vmin=-6, vmax=6)
    plt.colorbar(img1, ax=ax1, fraction=0.026, pad=0.04)
    ax1.set_title("Shift = "+str(0)+" (Original)")

    reward_map2 = shifter(img=reward_map, dx=0, dy=0)
    img2 = ax2.imshow(reward_map2, vmin=-6, vmax=6)
    plt.colorbar(img2, ax=ax2, fraction=0.026, pad=0.04)
    ax2.set_title("Shift = "+str(0)+f" (MSE={np.mean((reward_map2-reward_map)**2):.3g})")

    dx3 = 5
    reward_map3 = shifter(img=reward_map, dx=dx3, dy=0)
    img3 = ax3.imshow(reward_map3, vmin=-6, vmax=6)
    plt.colorbar(img3, ax=ax3, fraction=0.026, pad=0.04)
    ax3.set_title("Shift = "+str(dx3)+f" (MSE={np.mean((reward_map3-np.roll(reward_map, dx3, axis=1))**2):.3g})")

    dx4 = 2.5
    reward_map4 = shifter(img=reward_map, dx=dx4, dy=0)
    img4 = ax4.imshow(reward_map4, vmin=-6, vmax=6)
    plt.colorbar(img4, ax=ax4, fraction=0.026, pad=0.04)
    ax4.set_title("Shift = "+str(dx4))

    plt.show()


sample_idx = np.array(sample_idx)
test_img_idx = np.array(test_img_idx)
sample_idx = sample_idx[np.argsort(test_img_idx)]
#raise NotImplementedError

modifier = RewardMapModifier((0, 0), (3, 2))


for num, idx in enumerate(sample_idx):
    r_arr = data2[idx*800, :800].reshape(40, 20)
    lc_arr = data2[idx*800, 800:900]
    lc_info = data2[idx*800, 900:906]
    reward_map0 = data2[idx*800:(idx+1)*800, -1].reshape(40, 20)

    reward_map, _ = modifier.operation(np.expand_dims(reward_map0.T, axis=-1), None, order=['extend_vert', 'extend_hori', 'blur'])
    reward_map = reward_map[:, :, 0].T
    
    plotter(r_arr, reward_map.T, num)
    break
