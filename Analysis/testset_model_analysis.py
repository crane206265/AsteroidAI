import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
from torch import nn
from torchsummary import summary
from tqdm import tqdm
import gc
import os



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Model Setting --------------------

class QValueNet_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, activation=nn.ReLU, dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.x_shift = nn.Parameter(torch.tensor([0], dtype=torch.float32))
        self.y_shift = nn.Parameter(torch.tensor([0], dtype=torch.float32))

        # R_arr encoders (input: [B, C, 40, 20])
        self.r_arr_encoder1 = nn.Sequential(
            nn.Conv2d(1, 8, (9, 5)),  # 1 channel / assumed input is already done padding=1 #(1, 16, 3)
            self.activation(),
            nn.MaxPool2d(2)  # -> 20×10
        )

        self.r_arr_encoder2 = nn.Sequential(
            nn.Conv2d(8, 16, (5, 3)),  # assumed input is already done padding=1 #(16, 32, 3)
            self.activation(),
            nn.Flatten(),     # -> 16×20×10 = 3200  # -> 32×20×10 = 6400
            nn.Linear(3200, 1024)
        )

        # Info encoder (input: [B, 1, 6])
        self.info_encoder = nn.Sequential(
            nn.Linear(6, 32),
            self.activation(),
            nn.Linear(32, 64)
        )

        # RL encoder (input: [B, 1, 4])
        self.rl_encoder = nn.Sequential(
            nn.Linear(4, 32),
            self.activation(),
            nn.Linear(32, 64)
        )

        # Lightcurves encoder (input: [B, 1, 100])
        self.lc_encoder1 = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=15),
            self.activation(),
            nn.MaxPool1d(2),   # → 50
        )

        self.lc_encoder2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=9),
            self.activation(),
            nn.Flatten(),      # → 32×50
            nn.Linear(32*50, 256)
        )

        # Fusion & Head
        self.head = nn.Sequential(
            nn.Linear(1024 + 256 + 64 + 64, 1024),
            self.activation(),
            nn.Dropout(dropout),

            #nn.Linear(1024, 1024),
            #self.activation(),
            #nn.Dropout(dropout), #//// 새로 추가

            nn.Linear(1024, 256),
            self.activation(),
            nn.Dropout(dropout),

            nn.Linear(256, 1)  # e.g., class count or regression value
        )

    def r_padding(self, x, pad=(1, 1)):
        N, C, H, W = x.shape
        pad_H = pad[0]
        pad_W = pad[1]

        out = torch.full((N, C, H + 2*pad_H, W + 2*pad_W), fill_value=0.0, dtype=x.dtype, device=x.device)
        out[:, :, pad_H:pad_H+H, pad_W:pad_W+W] = x
        out[:, :, :, :pad_W] = torch.roll(torch.flip(out[:, :, :, pad_W:pad_W+pad_W], (-2,)), 20, -1)
        out[:, :, :, -pad_W:] = torch.roll(torch.flip(out[:, :, :, -pad_W-pad_W:-pad_W], (-2,)), 20, -1)
        out[:, :, :pad_H, pad_W:pad_W+W] = x[:, :, -pad_H:, :]
        out[:, :, -pad_H:, pad_W:pad_W+W] = x[:, :, :pad_H, :]
        return out

    def lc_padding(self, x, pad=1):
        N, C, W = x.shape

        out = torch.full((N, C, W + 2*pad), fill_value=0.0, dtype=x.dtype, device=x.device)
        out[:, :, pad:pad+W] = x
        out[:, :, :pad] = x[:, :, -pad:]
        out[:, :, -pad:] = x[:, :, :pad]
        return out

    def shifter(self, img, dx=0, dy=0):
        PI = 3.14159265358979
        img_F = torch.fft.fft2(img)
        N, M = img.shape
        ky = torch.fft.fftfreq(N)[:, None].to(device)
        kx = torch.fft.fftfreq(M)[None, :].to(device)
        phase = torch.exp(-2j*PI*(kx*dx + ky*dy))
        new_img = torch.fft.ifft2(img_F*phase)
        return new_img.real

    def forward(self, X):
        r_arr = X[..., :800].reshape((X.shape[0], 1, 40, 20))
        lc_arr = X[..., 800:900].reshape((X.shape[0], 1, 100))
        lc_info = X[..., 900:906]
        rl_info = X[..., 906:]

        r_arr_feat = torch.transpose(r_arr, -2, -1)
        r_arr_feat = self.r_padding(r_arr_feat, pad=(4, 2))
        r_arr_feat = self.r_arr_encoder1(r_arr_feat)
        r_arr_feat = self.r_padding(r_arr_feat, pad=(2, 1))
        r_arr_feat = self.r_arr_encoder2(r_arr_feat)

        lc_feat = self.lc_padding(lc_arr, pad=7)
        lc_feat = self.lc_encoder1(lc_feat)
        lc_feat = self.lc_padding(lc_feat, pad=4)
        lc_feat = self.lc_encoder2(lc_feat)

        info_feat = self.info_encoder(lc_info)
        info_feat = torch.squeeze(info_feat, dim=1)

        rl_feat = self.rl_encoder(rl_info)
        rl_feat = torch.squeeze(rl_feat, dim=1)

        fusion_feat = torch.cat((r_arr_feat, lc_feat, info_feat, rl_feat), dim=1)
        out = self.head(fusion_feat)

        out = self.shifter(out, dx=20*self.x_shift, dy=10*self.y_shift)

        PI = 3.14159265358979
        out = 6 * 2 / PI * torch.atan(out/0.8) #out/0.8
        #out = 7 * 2 / PI * torch.atan(1.5 * out)

        return out

# -------------------- Test Data Loading --------------------

data2 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_2.npy")[1:]

print("[Data shapes]")
print("data2 shape : ", data2.shape)
print("-"*20)

#sample_idx = [508, 620, 639, 466, 862, 970, 582, 789, 828, 309]
#test_img_idx = [2, 4, 5, 1, 8, 9, 3, 6, 7, 0]
sample_idx = [867, 206, 998, 73, 1032, 740, 411, 624, 670, 204]
test_img_idx = [7, 2, 8, 0, 9, 6, 3, 4, 5, 1]

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

# -------------------- Definitions of Funtions --------------------

def input_data(state):
    input_list = []
    for idx in range(800):
        i = idx//int(20)
        j = idx%int(20)
        phi_action = (i/40)%1
        theta_action = (j/20)%1
        actions = np.array([phi_action, theta_action, 0.1, 0.1])
        input = torch.tensor(np.concatenate((state, actions))).float().to(device)
        input_list.append(torch.unsqueeze(input, 0))
    total_input = torch.concat(input_list, dim=0)
    return total_input
    
def load_model(model_path):
    model = QValueNet_CNN(input_dim=910, hidden_dim=1024, activation=nn.ELU, dropout=0.15).to(device)
    #summary(model, (1, model.input_dim))
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def loss_curve(reward_map, losses, test_img_idx):
    fig = plt.figure(figsize=(9, 4))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    im1 = ax1.imshow(reward_map, vmax=6, vmin=-6)
    plt.colorbar(im1, ax=ax1, fraction=0.026, pad=0.04)
    ax1.set_title("Test_img_idx : "+str(test_img_idx))

    for i in range(losses.shape[1]): ax2.plot(losses[:, i], label="loss"+str(i))
    plt.legend()
    ax2.set_title("Loss Curves")

    plt.show()

def display_all(target_maps, pred_maps, losses):
    dpi = 250
    fig = plt.figure(figsize=(60, 50))
    axes_total = []

    for i in tqdm(range(pred_maps.shape[1])):
        ax_list_temp = []
        for j in range(pred_maps.shape[0]+1+1):
            gc.collect()
            ax_temp = fig.add_subplot(target_maps.shape[1], target_maps.shape[0]+1+1, i*(target_maps.shape[0]+1+1)+j+1)
            if j == pred_maps.shape[0]:
                img_temp = ax_temp.imshow(target_maps[i, :, :], vmax=6, vmin=-6)
                plt.colorbar(img_temp, ax=ax_temp, fraction=0.026, pad=0.04)
                ax_temp.set_title("Target Test Img "+str(i))
            elif j >= 0 and j <= pred_maps.shape[0]-1:
                img_temp = ax_temp.imshow(pred_maps[j, i, :, :], vmax=np.max(np.abs(pred_maps[j, i, :, :])), vmin=-np.max(np.abs(pred_maps[j-1, i, :, :])))
                plt.colorbar(img_temp, ax=ax_temp, fraction=0.026, pad=0.04)
                ax_temp.set_title("Predicted Test Img "+str((j+1)*10+20))
            elif j == pred_maps.shape[0]+1:
                for k in range(losses.shape[-1]):
                    ax_temp.plot(losses[:, i, k], label='loss'+str(k))
                plt.legend()
            ax_list_temp.append(ax_temp)
        axes_total.append(ax_list_temp)
    #plt.show()
    plt.tight_layout()
    plt.savefig("C:/Users/dlgkr/Downloads/display_all_2.png", dpi=dpi)
# -------------------- Loss Functions --------------------

def processer(reward_map, propagation=(3, 1)):
    hori_prop, vert_prop = propagation
    reward_map_pos = np.where(reward_map > 0, reward_map, 0)
    reward_map_neg = np.where(reward_map < 0, reward_map, 0)

    exp = 2
    div = hori_prop + vert_prop - 0.5
    reward_map_prop = reward_map_pos**exp
    for i in range(1, hori_prop+1):
        reward_map_prop[:, :-i] = reward_map_prop[:, :-i] + reward_map_pos[:, i:]**exp
        reward_map_prop[:,  i:] = reward_map_prop[:,  i:] + reward_map_pos[:, :-i]**exp
    for j in range(1, vert_prop+1):
        reward_map_prop[:-j, :] = reward_map_prop[:-j, :] + reward_map_pos[j:, :]**exp
        reward_map_prop[ j:, :] = reward_map_prop[ j:, :] + reward_map_pos[:-j, :]**exp
    reward_map_prop = (reward_map_prop / div)**(1/exp)
    reward_map_prop += reward_map_neg

    return reward_map_prop


# Smoothed Propagation Loss
def loss_fn(x, y):
    # propagation of positive values
    hori_prop, vert_prop = 1, 1
    beta = 0.3
    x_prop = processer(x, propagation=(hori_prop, vert_prop))
    y_prop = processer(y, propagation=(hori_prop, vert_prop))

    x_prop_pos = np.where(x_prop > 0, x_prop, 0)
    x_prop_neg = np.where(x_prop < 0, x_prop, 0)
    x_prop_final = x_prop_pos*np.max(y_prop)/(np.max(x_prop)+beta) + x_prop_neg*np.min(y_prop)/(np.min(x_prop)+beta)

    loss = np.sqrt(np.mean((x_prop_final - y_prop) ** 2))
    return loss


def plotter0(r_arr, lc_arr, lc_info, reward_map0, reward_map, loss, roll_loss, idx):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    Sdir = lc_info[0:3]
    Edir = lc_info[3:6]
    Stheta = np.arccos(Sdir[-1]) * 20 / np.pi
    Etheta = np.arccos(Edir[-1]) * 20 / np.pi
    #reward_map = 6*2/np.pi * np.arctan(reward_map * np.pi/2)

    r_arr_img = ax1.imshow(r_arr, vmax=8, vmin=12)
    ax1.set_title("R_arr at idx " + str(idx))
    plt.colorbar(r_arr_img, ax=ax1, shrink=0.75)
    ax1.plot([0, 39], [Stheta, Stheta], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax1.plot([0, 39], [Etheta, Etheta], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax1.legend()

    ax2.plot(lc_arr, label="Lightcurve", color='royalblue')
    ax2.plot(np.roll(lc_arr, 50), label="Lightcurve_Rolled", color='orangered', linestyle='dotted', alpha=0.7)
    ax2.set_title("Lightcurve at idx " + str(idx))

    reward_map0_img = ax4.imshow(reward_map0, vmax=np.max(np.abs(reward_map0)), vmin=-np.max(np.abs(reward_map0)))#, vmax=6, vmin=-6)
    ax4.set_title("Reward_Map at idx " + str(idx) + "(loss="+str(int(loss*1000)/1000)+")")
    plt.colorbar(reward_map0_img, ax=ax4, shrink=0.75)
    ax4.plot([0, 40*(modifier0.extends[1]+1)-1], [Etheta*(modifier0.extends[0]+1), Etheta*(modifier0.extends[0]+1)], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax4.plot([0, 40*(modifier0.extends[1]+1)-1], [Stheta*(modifier0.extends[0]+1), Stheta*(modifier0.extends[0]+1)], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax4.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0]+20, 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.plot([20*modifier0.extends[1], 20*modifier0.extends[1]], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.plot([20*modifier0.extends[1]+40, 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax4.set_xlim([0-0.5, 40-0.5])
    ax4.set_ylim([20-0.5, 0-0.5])

    reward_map_img = ax5.imshow(reward_map, vmax=np.max(np.abs(reward_map)), vmin=-np.max(np.abs(reward_map)))#, vmax=6, vmin=-6)
    ax5.set_title("Reward_Map at idx " + str(idx) + "(loss="+str(int(loss*1000)/1000)+")")
    plt.colorbar(reward_map_img, ax=ax5, shrink=0.75)
    ax5.plot([0, 40*(modifier0.extends[1]+1)-1], [Etheta*(modifier0.extends[0]+1), Etheta*(modifier0.extends[0]+1)], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax5.plot([0, 40*(modifier0.extends[1]+1)-1], [Stheta*(modifier0.extends[0]+1), Stheta*(modifier0.extends[0]+1)], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax5.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0]+20, 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.plot([20*modifier0.extends[1], 20*modifier0.extends[1]], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.plot([20*modifier0.extends[1]+40, 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.set_xlim([0-0.5, 40-0.5])
    ax5.set_ylim([20-0.5, 0-0.5])

    roll_loss_img = ax6.imshow(roll_loss)
    ax6.plot(np.argmin(roll_loss)%40, np.argmin(roll_loss)//40, color='red', marker='x')  
    ax6.set_title("Roll Loss at idx " + str(idx))
    plt.colorbar(roll_loss_img, ax=ax6, shrink=0.75)  

    plt.tight_layout()
    plt.show()

def plotter1(r_arr, lc_arr, lc_info, reward_map0, reward_map, loss, roll_loss, idx):
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    Sdir = lc_info[0:3]
    Edir = lc_info[3:6]
    Stheta = np.arccos(Sdir[-1]) * 20 / np.pi
    Etheta = np.arccos(Edir[-1]) * 20 / np.pi

    ax1.plot(lc_arr, label="Lightcurve", color='royalblue')
    ax1.plot(np.roll(lc_arr, 50), label="Lightcurve_Rolled", color='orangered', linestyle='dotted', alpha=0.7)
    ax1.set_title("Lightcurve at idx " + str(idx))

    fft_coef_zip = np.abs(np.fft.fft(lc_arr))[:lc_arr.shape[0]//2+1]
    fft_coef_zip = np.log10(fft_coef_zip)
    log_thr = np.log10(4)#0.6
    global filtered_num, total_num, filtered_percents
    if not np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:]):
        print("Filtered Percent {:.2f}%".format(100*filtered_num/total_num))
        filtered_percents.append(100*filtered_num/total_num)
        plt.close()
        return
    filtered_num += 1
    lim = (np.min(fft_coef_zip)-0.3, np.max(fft_coef_zip)+0.3)

    ax2.plot(fft_coef_zip)
    ax2.plot([np.argmax(fft_coef_zip), np.argmax(fft_coef_zip)], [lim[0], lim[1]], linestyle='dotted')
    ax2.set_title("FFT of LC at idx " + str(idx))
    ax2.set_ylim(lim[0], lim[1])

    r_arr_img = ax4.imshow(r_arr, vmax=8, vmin=12)
    ax4.set_title("R_arr at idx " + str(idx))
    plt.colorbar(r_arr_img, ax=ax4, shrink=0.75)
    ax4.plot([0, 39], [Stheta, Stheta], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax4.plot([0, 39], [Etheta, Etheta], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax4.legend()

    reward_map0_img = ax5.imshow(reward_map0, vmax=np.max(np.abs(reward_map0)), vmin=-np.max(np.abs(reward_map0)))#, vmax=6, vmin=-6)
    ax5.set_title("Reward_Map at idx " + str(idx) + "(loss="+str(int(loss*1000)/1000)+")")
    plt.colorbar(reward_map0_img, ax=ax5, shrink=0.75)
    ax5.plot([0, 40*(modifier0.extends[1]+1)-1], [Etheta*(modifier0.extends[0]+1), Etheta*(modifier0.extends[0]+1)], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax5.plot([0, 40*(modifier0.extends[1]+1)-1], [Stheta*(modifier0.extends[0]+1), Stheta*(modifier0.extends[0]+1)], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax5.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0]+20, 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.plot([20*modifier0.extends[1], 20*modifier0.extends[1]], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.plot([20*modifier0.extends[1]+40, 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax5.set_xlim([0-0.5, 40-0.5])
    ax5.set_ylim([20-0.5, 0-0.5])

    reward_map_img = ax6.imshow(reward_map, vmax=np.max(np.abs(reward_map)), vmin=-np.max(np.abs(reward_map)))#, vmax=6, vmin=-6)
    ax6.set_title("Reward_Map at idx " + str(idx) + "(loss="+str(int(loss*1000)/1000)+")")
    plt.colorbar(reward_map_img, ax=ax6, shrink=0.75)
    ax6.plot([0, 40*(modifier0.extends[1]+1)-1], [Etheta*(modifier0.extends[0]+1), Etheta*(modifier0.extends[0]+1)], color='royalblue', label='Earth Direction', linewidth=2, linestyle='dashed')
    ax6.plot([0, 40*(modifier0.extends[1]+1)-1], [Stheta*(modifier0.extends[0]+1), Stheta*(modifier0.extends[0]+1)], color='orangered', label='Sun Direction', linewidth=2, linestyle='dashed')
    ax6.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]], color='gold', linewidth=0.8, linestyle='dotted')
    ax6.plot([20*modifier0.extends[1], 20*modifier0.extends[1]+40], [10*modifier0.extends[0]+20, 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax6.plot([20*modifier0.extends[1], 20*modifier0.extends[1]], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax6.plot([20*modifier0.extends[1]+40, 20*modifier0.extends[1]+40], [10*modifier0.extends[0], 10*modifier0.extends[0]+20], color='gold', linewidth=0.8, linestyle='dotted')
    ax6.set_xlim([0-0.5, 40-0.5])
    ax6.set_ylim([20-0.5, 0-0.5])

    plt.tight_layout()
    #plt.show()
    plt.savefig("./analysis/testset_model_analysis_imgs/img{:03d}.png".format(idx))
    plt.close()
    print("Filtered Percent {:.2f}%".format(100*filtered_num/total_num))
    filtered_percents.append(100*filtered_num/total_num)

# -------------------- Main Analysis --------------------

model_path = "C:/Users/dlgkr/Downloads/train0831_2/120model.pt"

#np.random.seed(206265)
sample_idx = list(np.random.randint(0, data2.shape[0]//800, 200))
print("sample idx : [", end='')
for idx in sample_idx:
    print(idx, end=' ')
print("]")
test_img_idx = list(range(len(sample_idx)))

modifier0 = RewardMapModifier((0, 0), (3, 2))

losses = np.zeros((len(test_img_idx)))
pred_maps = np.zeros((len(test_img_idx), 20, 40))
target_maps = np.zeros((len(test_img_idx), 20, 40))

model = load_model(model_path)
gc.collect()

filtered_num = 0
total_num = 0
filtered_percents = []

for num, i in zip(test_img_idx[:], sample_idx[:]):
    # num : test_img_idx
    # i : idx in data2
    state = data2[i*800, :906]
    target0 = data2[i*800:(i+1)*800, -1].reshape(40, 20).T
    target, _ = modifier0.operation(np.expand_dims(target0, axis=-1), None, order=['extend_vert', 'extend_hori', 'blur'])
    target = target[:, :, 0]
    
    pred = np.zeros((20, 40))
    model.eval()
    with torch.no_grad():
        input = input_data(state)
        rewards = model(input)
        pred = rewards.cpu().numpy().reshape(40, 20).T

    pred_maps[num, :, :] = pred.copy()
    losses[num] = loss_fn(pred, target)
    target_maps[num, :, :] = target.copy()

    roll_loss = np.zeros((20, 40))
    for j in range(20):
        for k in range(40):
            roll_loss[j, k] = loss_fn(np.roll(pred, (j, k), axis=(0, 1)), target)

    total_num += 1
    #plotter0(state[:800].reshape(40, 20).T, state[800:900], state[900:906], target_maps[num, :, :], pred_maps[num, :, :], losses[num], roll_loss, i)
    plotter1(state[:800].reshape(40, 20).T, state[800:900], state[900:906], target_maps[num, :, :], pred_maps[num, :, :], losses[num], roll_loss, i)




"""
reward0_path = "c:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_reward0.npy"
passed_idx_path = "c:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_passed_idx.npy"
reward0 = np.load(reward0_path)
passed_idx = np.load(passed_idx_path)
print(reward0)
print(passed_idx)
print(data2.shape[0]//800)  
"""
plt.plot(filtered_percents)
plt.plot([0-10, len(filtered_percents)+10], [filtered_percents[-1], filtered_percents[-1]], color='lightgray', linestyle='-.')
plt.ylabel("%")
plt.ylim([15, 40])
plt.show()      
