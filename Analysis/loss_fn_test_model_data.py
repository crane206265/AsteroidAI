import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torchsummary import summary
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------- Model Setting --------------------

class QValueNet_CNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, activation=nn.ReLU, dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

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

        PI = 3.14159265358979
        out = 6 * 2 / PI * torch.atan(2 * out) #out/0.8
        #out = 7 * 2 / PI * torch.atan(1.5 * out)

        return out

# -------------------- Test Data Loading --------------------

data2 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_2.npy")[1:]

print("[Data shapes]")
print("data2 shape : ", data2.shape)
print("-"*20)

sample_idx = [508, 620, 639, 466, 862, 970, 582, 789, 828, 309]
test_img_idx = [2, 4, 5, 1, 8, 9, 3, 6, 7, 0]

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
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

# -------------------- Loss Functions --------------------

# MSE Loss
def loss0(pred, target): return np.mean((pred-target)**2)

# Smoothed Propagation Loss
def loss1(x, y):
    # propagation of positive values
    hori_prop, vert_prop = 1, 1
    eps = 1
    x_prop = processer(x, propagation=(hori_prop, vert_prop))
    y_prop = processer(y, propagation=(hori_prop, vert_prop))

    x_prop_pos = np.where(x_prop > 0, x_prop, 0)
    x_prop_neg = np.where(x_prop < 0, x_prop, 0)
    x_prop_final = x_prop_pos*np.max(y_prop)/(np.max(x_prop)+eps) + x_prop_neg*np.min(y_prop)/(np.min(x_prop)+eps)

    loss = np.sqrt(np.mean((x_prop_final - y_prop) ** 2))
    return loss

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

# -------------------- Main Analysis --------------------

model_paths = []
loss_fns = [loss0, loss1]

for model_path in model_paths:
    model = load_model(model_path)
    gc.collect()

    losses = np.zeros((len(test_img_idx), len(loss_fns)))
    for num, i in zip(test_img_idx[:], sample_idx[:]):
        # num : test_img_idx
        # i : idx in data2
        state = data2[i*800, :906]
        target = data2[i*800:(i+1)*800, -1].reshape(40, 20).T

        pred = np.zeros((20, 40))
        model.eval()
        with torch.no_grad():
            input = input_data(state)
            rewards = model(input)
            pred = rewards.cpu().numpy().reshape(40, 20).T
        
        for j, loss_fn in enumerate(loss_fns):
            losses[num, j] = loss_fn(pred, target)




    
        