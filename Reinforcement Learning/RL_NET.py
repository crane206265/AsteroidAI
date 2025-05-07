import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torchsummary import summary
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
        self.best_score = np.Inf if mode == 'min' else 0
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


def train_loop(dataloader, model, loss_fn, optimizer, train_loss, es:EarlyStopping):
    epoch_loss = 0
    n_train = 0
    
    model.train()
    #with torch.autograd.detect_anomaly(True):
    for X_train, y_train in dataloader:
        #print(X_train.shape, y_train.shape)
        #X_train = torch.unsqueeze(X_train, 1)
        #y_train = torch.unsqueeze(y_train, 1)
        #print(X_train.shape, y_train.shape, "..")
        
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
            #print(X_test.shape, y_test.shape)
            #X_test = torch.unsqueeze(X_test, 1)
            #y_test = torch.unsqueeze(y_test, 1)
            #print(X_test.shape, y_test.shape, "..")

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
    

# seed
seed = 722
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# hyperparameters
batch_size = 256
learning_rate = 1e-3
max_epoch = 100

data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_RL_preset_510.npy"
data_RL_preset = np.load(data_path)
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

model = QValueNet(input_dim=299, hidden_dim=512, activation=nn.LeakyReLU, dropout=0.15).to(device)
summary(model, (1, model.input_dim))

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = nn.MSELoss()

train_loss = []
test_loss = []

es = EarlyStopping(patience=20, delta=0.1)
for epoch in tqdm(range(max_epoch)):
    print("[epochs:{:2}]".format(epoch), end=' ')
    #print("EPOCH "+str(epoch)+" TRAINING...")
    train_loop(train_dataloader, model, loss_fn, optimizer, train_loss, es)
    #print("EPOCH "+str(epoch)+" TESTING...")
    test_loop(test_dataloader, model, loss_fn, test_loss, epoch)
    #print("")
    
    if es.early_stop:
        print("EarlyStop Triggered : Bestscore = {:7.4g}".format(es.best_score))
        break
print("DONE")

plt.plot(train_loss[:], label='train_loss')
plt.plot(test_loss[:], label='test_loss')
plt.legend()
plt.title("Train/Test Loss (MSE)")
plt.show()


