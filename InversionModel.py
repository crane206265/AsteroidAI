import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torch.utils import data


# Hyperparameters
max_epoch = 100
learning_rate = 0.001
batch_size = 32
testset_ratio = 0.3

# Dataset Preparation
total_data = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_total.npz")
X_total = torch.tensor(total_data['X_total'].astype(np.float32))
Y_total = torch.tensor(total_data['Y_total'].astype(np.complex64))

# complex to real
Y_total = torch.view_as_real(Y_total).type(torch.float32)
Y_total = torch.flatten(Y_total, 1)
Y_total = torch.cat((Y_total[:, :-5], torch.unsqueeze(Y_total[:, -4], dim=1), torch.unsqueeze(Y_total[:, -2], dim=1)), dim=1)

dataset_len = X_total.shape[0]
shuffle_idx = np.arange(0, dataset_len)
np.random.shuffle(shuffle_idx)
X_total = X_total[shuffle_idx, :]
Y_total = Y_total[shuffle_idx, :]

X_train0 = X_total[:int(dataset_len*testset_ratio)]
X_test0 = X_total[int(dataset_len*testset_ratio):]
y_train0 = Y_total[:int(dataset_len*testset_ratio)]
y_test0 = Y_total[int(dataset_len*testset_ratio):]


class LC_dataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super().__init__()

        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    

train_dataset = LC_dataset(X_train0, y_train0)
train_dataloader = data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_dataset = LC_dataset(X_test0, y_test0)
test_dataloader = data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True)



# Choose device to be used for learning
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")



# Model Class
class LC_1dCNN(nn.Module):
    def __init__(self, lightcurve_len = 200, output_dim = 165):
        """
        lightcurve_len : length of lightcurve
        output_dim : dimension of output = 2*(coef_l+1)^2 + 3
        """
        super().__init__()
        self.lc_len = lightcurve_len

        # layers for lightcurve data
        self.ConvLayer1 = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout()
        )

        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout()
        )

        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout()
        )

        len_after_cnn = ((lightcurve_len//2)//2)//2

        self.LinearLayer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*len_after_cnn, 1024),
            nn.ReLU(),
            nn.Dropout()
        )

        # layers for concatenated data
        self.LinearLayer2 = nn.Sequential(
            nn.Linear(1030, 512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.LinearLayer3 = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout()
        )

        self.OutLayer = nn.Linear(256, output_dim)

    def forward(self, x):
        # x : [[lc_arr], [lc_info]] (BatchSize X 1 X lc_len+6)
        lc_info = x[:, :, self.lc_len:]
        x = x[:, :, :self.lc_len] #lc_arr
        # 1D CNN
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        x = self.ConvLayer3(x)
        x = self.LinearLayer1(x)

        # concatenate data
        x = torch.cat((x, torch.squeeze(lc_info)), dim=1)

        x = self.LinearLayer2(x)
        x = self.LinearLayer3(x)
        output = self.OutLayer(x)

        return output

model = LC_1dCNN().to(device)
#print(model)



class EarlyStopping():
    def __init__(self, patience=5, delta=0.0005, mode='min'):
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
    for X_train, y_train in dataloader:
        X_train = torch.unsqueeze(X_train, 1)
        y_train = torch.unsqueeze(y_train, 1)
        
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

    print("train_loss : {:.4g}".format(epoch_loss), end=' ')

def test_loop(dataloader, model, loss_fn, test_loss):
    epoch_loss = 0
    epoch_coef_loss = 0
    epoch_axis_loss = 0
    coef_std = 0
    n_test = 0

    model.eval()
    with torch.no_grad():
        for X_test, y_test in dataloader:
            X_test = torch.unsqueeze(X_test, 1)
            y_test = torch.unsqueeze(y_test, 1)

            X_test = X_test.to(device)
            y_test = y_test.to(device)

            pred = model(X_test)

            epoch_loss += loss_fn(pred, y_test).item()*X_test.size(0)
            epoch_coef_loss += CoefAxisLoss.CoefLoss(pred, y_test).item()*X_test.size(0)
            epoch_axis_loss += CoefAxisLoss.AxisLoss(pred, y_test).item()*X_test.size(0)
            coef_std += np.std((y_test.numpy())[:, :, :-3])*X_test.size(0) # coef_std
            n_test += X_test.size(0)
        
    epoch_loss /= n_test
    epoch_coef_loss /= n_test
    epoch_axis_loss /= n_test
    coef_std /= n_test
    test_loss.append(epoch_loss)
    
    print("| test_loss : {:.5g} | coef_loss : {:.5g} | axis_loss : {:.5g} | avg_coef_std : {:.5g}".format(epoch_loss, epoch_coef_loss, epoch_axis_loss, coef_std), end=' ')



class CoefAxisLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target, alpha=1):
        """
        Use MSE for coef loss function
        Use cosine for rot_axis loss function
        - alpha : weight of cosine loss
        """
        loss = 0
        input_coef = input[:, :-3]
        input_axis = nn.functional.normalize(input[:, -3:])
        target_coef = target[:, :, :-3]
        target_axis = torch.squeeze(target[:, :, -3:])

        torch_MSE = nn.MSELoss()
        torch_cos = nn.CosineEmbeddingLoss()
        loss = torch_MSE(input_coef, target_coef) + alpha * torch_cos(input_axis, target_axis, torch.ones(input_axis.shape[0]))
        return loss
    
    def CoefLoss(input, target):
        torch_MSE = nn.MSELoss()
        return torch_MSE(input[:, :-3], target[:, :, :-3])
    
    def AxisLoss(input, target):
        torch_cos = nn.CosineEmbeddingLoss()
        return torch_cos(input[:, -3:], torch.squeeze(target[:, :, -3:]), torch.ones(input.shape[0]))


optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
loss_fn = CoefAxisLoss()



train_loss = []
test_loss = []


es = EarlyStopping(patience=5, delta=0.002, mode='min')
for epoch in range(max_epoch):
    print("[epochs:{}]".format(epoch), end=' ')
    train_loop(train_dataloader, model, loss_fn, optimizer, train_loss, es)
    test_loop(test_dataloader, model, loss_fn, test_loss)
    print("")

    if es.early_stop:
        print("EarlyStop Triggered : Bestscore = {:.5g}".format(es.best_score))
        break
print("DONE")

plt.plot(train_loss, label='train_loss')
plt.plot(test_loss, label='test_loss')
plt.legend()
plt.title("Train/Test Loss (MSE)")
plt.show()


