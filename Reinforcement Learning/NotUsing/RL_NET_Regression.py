import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_RL_preset_510.npy"
data_RL_preset0 = np.load(data_path)



class QValueNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=512, activation=nn.ReLU, dropout=0.3):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.activation = activation

        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            activation(),
            nn.Dropout(dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Dropout(dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            activation(),
            nn.Dropout(dropout),

            nn.Linear(self.hidden_dim, 1)
        )

    def forward(self, X):
        return self.model(X)

class Dataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super(Dataset, self).__init__()

        if not torch.is_tensor(x_tensor):
            self.x = torch.tensor(x_tensor).float()
            self.y = torch.tensor(y_tensor).float()
        else:
            self.x = x_tensor.float()
            self.y = y_tensor.float()

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.x.shape[0]

class EarlyStopping():
    def __init__(self, patience, delta, mode='min'):
        """
        patience : max number of waiting
        delta : min boundary of "change"
        mode :
        verbose :
        """

        self.patience = patience
        self.delta = delta
        self.mode = mode
        self.best_score = np.inf if mode == 'min' else 0
        self.count = 0
        self.early_stop = False

    def __call__(self, score):
        if self.mode == 'min':
            if (self.best_score - score) < self.delta:
                self.count += 1
            else:
                self.best_score = score
                self.count = 0
        elif self.mode == 'max':
            if (score - self.best_score) < self.delta:
                self.count += 1
            else:
                self.best_score = score
                self.count = 0

        if self.count >= self.patience:
            self.early_stop = True

def data_split(datasets, train_ratio=0.7, shuffle=True):
    idx = np.arange(0, datasets[0].shape[0])
    np.random.shuffle(idx)
    train_list = []
    test_list = []
    for dataset in datasets:
        if shuffle: dataset = dataset[idx]
        train_list.append(dataset[:int(train_ratio*dataset.shape[0])])
        test_list.append(dataset[int(train_ratio*dataset.shape[0]):])
    return train_list, test_list


class CustomLoss(nn.Module):
    def __init__(self, relative, percent):
      super().__init__()
      self.relative = relative
      self.percent = percent

    def forward(self, input, target):
      torch_MSE = nn.MSELoss()
      if self.relative:
          loss = torch_MSE(input/(target+1e-6), target/(target+1e-6))
          loss = torch.sqrt(loss + 1e-6)
      else:
          loss = torch.sqrt(torch_MSE(input, target))
          #weight = 0.5 + 0.5*torch.abs(target)
          #loss = torch.sqrt(torch.sum(weight*(input-target)**2)/torch.sum(weight) + 1e-6)
      if self.percent:
          loss = 100 * loss
      return loss

def train_loop(dataloader, model, loss_fn, optimizer, train_loss, es:EarlyStopping):
    epoch_loss = 0
    n_train = 0

    model.train()
    #with torch.autograd.detect_anomaly(True):
    for X_train, y_train in dataloader:
        X_train = X_train.to(device)
        y_train = y_train.to(device)
        pred = model(X_train)

        loss = loss_fn(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()*X_train.size(0)
        n_train += X_train.size(0)

    epoch_loss /= n_train
    train_loss.append(epoch_loss)

    es(epoch_loss)
    #print("train_loss : {:9.4g}".format(epoch_loss), end=' ')

def test_loop(dataloader, model, loss_fn, test_loss, epoch):
    epoch_loss = 0
    n_test = 0

    model.eval()
    with torch.no_grad():
        for X_test, y_test in dataloader:
            X_test = X_test.to(device)
            y_test = y_test.to(device)
            pred = model(X_test)

            epoch_loss += loss_fn(pred, y_test).item()*X_test.size(0)
            n_test += X_test.size(0)

    epoch_loss /= n_test
    test_loss.append(epoch_loss)

    print("train_loss : {:9.4g}".format(train_loss[-1]), end=' ')
    print("| test_loss : {:9.4g}".format(epoch_loss), end=' ')
    print("\n", end=' ')

def test_img_show(i_img):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)

    if i_img == 0 or True:
        ax1.clear()
        im1 = ax1.imshow(test_img_list[i_img])
        ax1.set_title("TEST_IMAGE_"+str(i_img))
        plt.colorbar(im1, ax=ax1, fraction=0.026, pad=0.04)

    reward_map_temp = np.zeros((resol*N_set[0], resol*N_set[1]))
    model.eval()
    with torch.no_grad():
        for idx in range(N_set[0]*N_set[1]*resol*resol):
            i = idx//int(resol*N_set[1])
            j = idx%int(resol*N_set[1])
            phi_action = (i/(resol*N_set[0]))%1
            theta_action = (j/(resol*N_set[1]))%1

            state = test_img_data[i_img*resol*N_set[0]*N_set[1], :295]
            actions = np.array([phi_action, theta_action, 0.1, 0.1])

            input = torch.tensor(np.concatenate((state, actions))).float().to(device)
            reward = model(input)
            reward_map_temp[i, j] = reward
    ax2.clear()
    im2 = ax2.imshow(reward_map_temp.T)
    ax2.set_title("MODEL_OUTPUT_"+str(i_img))
    plt.colorbar(im2, ax=ax2, fraction=0.026, pad=0.04)

    plt.show()


# seed
seed = 722
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# hyperparameters
batch_size = 256
learning_rate = 6e-5
max_epoch = 1000

# other parameters
N_set = (40, 20)
resol = 1


data_len = data_RL_preset0[0, 0]
test_img_data = data_RL_preset0[-int(data_RL_preset0[0, 1]):, :]
test_img_num = int(data_RL_preset0[0, 1]/(resol*N_set[0]*N_set[1]))
test_img_list = []
for i in range(test_img_num):
    test_img_list.append(test_img_data[i*resol*N_set[0]*N_set[1]:(i+1)*resol*N_set[0]*N_set[1], -1].reshape((N_set[0], N_set[1])).T)

data_RL_preset = data_RL_preset0[1:-int(data_RL_preset0[0, 1]), :]
state_data = data_RL_preset[:, :-5]
action_data = data_RL_preset[:, -5:-1]
reward_data = data_RL_preset[:, -1:]

train_list, test_list = data_split((state_data, action_data, reward_data), train_ratio=0.7, shuffle=True)
train_state_data, train_action_data, train_reward_data = train_list
test_state_data, test_action_data, test_reward_data = test_list

train_dataset = Dataset(np.concatenate((train_state_data, train_action_data), axis=1), train_reward_data)
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = Dataset(np.concatenate((test_state_data, test_action_data), axis=1), test_reward_data)
test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)



model = QValueNet(input_dim=299, hidden_dim=512, activation=nn.ReLU, dropout=0.15).to(device)
summary(model, (1, model.input_dim))

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = CustomLoss(relative=False, percent=False)

train_loss = []
test_loss = []

es = EarlyStopping(patience=2000, delta=0.1)
for epoch in tqdm(range(max_epoch)):
    #print("EPOCH "+str(epoch)+" TRAINING...")
    train_loop(train_dataloader, model, loss_fn, optimizer, train_loss, es)
    #print("EPOCH "+str(epoch)+" TESTING...")
    test_loop(test_dataloader, model, loss_fn, test_loss, epoch)
    #print("")

    if es.early_stop:
        print("EarlyStop Triggered : Bestscore = {:7.4g}".format(es.best_score))
        break

    if (epoch+1)%10 == 0 and epoch != 0:
        plt.plot(train_loss[2:], label='train_loss')
        plt.plot(test_loss[2:], label='test_loss')
        plt.legend()
        plt.title("Train/Test Loss (MSE)")
        plt.show()

        for i in range(test_img_num):
            test_img_show(i)

    print("[epochs:{:2}]".format(epoch+2), end='')

print("DONE")

plt.plot(train_loss[2:], label='train_loss')
plt.plot(test_loss[2:], label='test_loss')
plt.legend()
plt.title("Train/Test Loss (MSE)")
plt.show()
