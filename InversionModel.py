import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch import optim
from torchsummary import summary
from torch_harmonics import InverseRealSHT
from tqdm import tqdm

from DataPreprocessing import DataPreProcessing, DataScaling
from Loss import CoefAxisLoss, RLoss, ComplexLoss

from SphericalHarmonicsExpansion import SphericalHarmonicsExpansion, SHEcoefDisplay
from Asteroid_Model import AsteroidModel
from LightCurve_Generator import LightCurve


# Hyperparameters
max_epoch = 100
learning_rate = 1e-1
batch_size = 256
trainset_ratio = 0.7
l_max = 7
lightcurve_unit_len = 200
merge_num = 3
complexloss_alpha = 0.1

input_dim = (lightcurve_unit_len+6)*merge_num
lightcurve_len = lightcurve_unit_len * merge_num
output_dim = (l_max+1)**2

data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_total.npz"

dataPP = DataPreProcessing(data_path=data_path)
dataPP.Y_total = dataPP.Y_total[:, 0:(l_max+1)**2]
dataPP.merge(merge_num=merge_num, ast_repeat_num=10, lc_len=200)
dataPP.scale_data(mode="dir_scaling", scaled_size = 20)
#dataPP.scale_data(mode="lc_scaling", scaled_mean=30.0)
#dataPP.scale_data(mode="exponential", exp=10, linear=1)
X_train0, X_test0, y_train0, y_test0 = dataPP.train_test_split(trainset_ratio=trainset_ratio)
train_dataloader = dataPP.return_dataloader(X_train0, y_train0, batch_size)
test_dataloader = dataPP.return_dataloader(X_test0, y_test0, batch_size)


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
class LCInversion(nn.Module):
    def __init__(self, lightcurve_len = lightcurve_len, output_dim = output_dim):
        """
        lightcurve_len : length of lightcurve
        output_dim : dimension of output = (coef_l+1)^2 + 3
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
        len_after_cnn = LCInversion.__1DCNNSize(lightcurve_len, 3, 1, 1)
        len_after_cnn = LCInversion.__1DCNNSize(len_after_cnn, 2, 2)

        self.ConvLayer2 = nn.Sequential(
            nn.Conv1d(in_channels=4, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout()
        )
        len_after_cnn = LCInversion.__1DCNNSize(len_after_cnn, 3, 1, 1)
        len_after_cnn = LCInversion.__1DCNNSize(len_after_cnn, 2, 2)

        self.ConvLayer3 = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2, 2),
            nn.Dropout()
        )

        self.LinearLayer1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16*len_after_cnn, 4096),#nn.Linear(64*len_after_cnn, 1024),
            nn.ReLU(),
            nn.Dropout()
        )

        self.LinearLayer11 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2048),#nn.Linear(64*len_after_cnn, 1024),
            nn.ReLU(),
            nn.Dropout()
        )

        # layers for concatenated data
        self.LinearLayer2 = nn.Sequential(
            nn.Linear(4096+int(6*merge_num), 2048),
            nn.ReLU(),
            nn.Dropout()
        )

        self.LinearLayer3 = nn.Sequential(
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout()
        )

        self.LinearLayer4 = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout()
        )

        self.OutLayer = nn.Linear(512, output_dim)

    @staticmethod
    def __1DCNNSize(input, kernel_size, stride=1, padding=0):
        return round((input + 2*padding - kernel_size)/stride + 1)

    def forward(self, x):
        # x : [[lc_arr], [lc_info]] (BatchSize X 1 X lc_len+6*merge_num)
        lc_info = x[..., self.lc_len:] #ERROR : torch.index_select(x, dim=-1, index=torch.tensor(range(self.lc_len, x.shape[-1])))
        x = x[..., :self.lc_len]#torch.index_select(x, dim=-1, index=torch.tensor(range(0, self.lc_len))) #lc_arr
        # 1D CNN
        x = self.ConvLayer1(x)
        x = self.ConvLayer2(x)
        #x = self.ConvLayer3(x)
        x = self.LinearLayer1(x)
        #x = self.LinearLayer11(x)

        # concatenate data
        x = torch.cat((x, torch.squeeze(lc_info, dim=1)), dim=1)

        x = self.LinearLayer2(x)
        #x = self.LinearLayer3(x)
        x = self.LinearLayer4(x)
        output = self.OutLayer(x)
        
        return output

# Model Class (without 1D CNN)
class LCInversionNN(nn.Module):
    def __init__(self, input_dim = input_dim, output_dim = output_dim):
        """
        input_dim : dimension of input
        output_dim : dimension of output = (coef_l+1)^2 + 3
        """
        super().__init__()

        # layers for lightcurve data
        self.LinearLayer01 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim, 1024),
            nn.ReLU(),
            nn.Dropout()
        )

        self.LinearLayer02 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Dropout()
        )

        self.LinearLayer03 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout()
        )

        self.LinearLayer04 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4096, 2048),
            nn.ReLU(),
            nn.Dropout()
        )        

        self.LinearLayer05 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout()
        )
        self.LinearLayer06 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout()
        )

        self.OutLayer = nn.Linear(256, output_dim)

    def forward(self, x):
        # x : [[lc_arr], [lc_info]] (BatchSize X 1 X lc_len+6*merge_num)
        x = self.LinearLayer01(x) #input-1024
        x = self.LinearLayer02(x) #1024-4096

        x = self.LinearLayer03(x) #4096-4096
        x = self.LinearLayer04(x) #4096-2048

        x = self.LinearLayer05(x) #2048-512
        x = self.LinearLayer06(x) #512-256
        output = self.OutLayer(x) #256-output
        
        return output

model = LCInversionNN().to(device)
#print(model)
summary(model, (1, input_dim))


class EarlyStopping():
    def __init__(self, patience=7, delta=5, mode='min'):
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



def train_loop(dataloader, model, loss_fn, optimizer, train_loss, es:EarlyStopping, sch:optim.lr_scheduler):
    epoch_loss = 0
    n_train = 0
    
    model.train()
    for X_train, y_train in tqdm(dataloader):
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

    sch.step()

    print("train_loss : {:9.4g}".format(epoch_loss), end=' ')

def test_loop(dataloader, model, loss_fn, test_loss):
    epoch_loss = 0
    epoch_coef_loss = 0
    epoch_Rloss = 0
    RLoss_fn = RLoss(l_max, relative=True, percent=True)
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
            epoch_Rloss += RLoss_fn.forward(pred, y_test).item()*X_test.size(0)
            coef_std += np.sum(np.std((y_test.numpy())[:, :, :-3], axis=-1)) # coef_std
            n_test += X_test.size(0)
        #print(pred)
        
    epoch_loss /= n_test
    epoch_coef_loss /= n_test
    epoch_Rloss /= n_test
    coef_std /= n_test
    test_loss.append(epoch_loss)
    
    print("| test_loss : {:9.4g}".format(epoch_loss), end=' ')
    #print("\n            ", end=' ')
    print("\n", end=' ')
    print("coef_loss : {:9.4g} |     Rloss : {:9.4g} | avg_coef_std : {:9.4g}".format(epoch_coef_loss, epoch_Rloss, coef_std), end=' ')



def model_return(input, model : LCInversion):
    """
    input : with reduced batch dimension
    model : LCInversion model
    """
    model.eval()
    with torch.no_grad():
        input = input.to(device)

        pred = model(input)
        rot_axis = pred[:, -3:]
        pred_coef = DataPreProcessing.coef_unzip(pred)#[:, :-3])
        coef_arr = torch.view_as_complex(pred_coef.view([-1, 2]))
        return coef_arr, rot_axis


def lr_lambda(epoch):
    if epoch % 4 == 0 and epoch <= 30 and epoch != 0:
        return 0.3
    else:
        return 1
    """
    if epoch == 5:
        return 1#.002
    elif epoch == 10:
        return 0.01
    elif epoch == 20:
        return 0.01
    else:
        return 1
    """

optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
# DO NOT USE SGD!!!!!!!!
#loss_fn = RLoss(l_max, relative=True, percent=True)
loss_fn = CoefAxisLoss(relative=True, percent=True, frac_reg=False)
#loss_fn = ComplexLoss(alpha=complexloss_alpha, l_max=l_max, relative=True, percent=True)
scheduler = optim.lr_scheduler.LambdaLR(optimizer=optimizer, lr_lambda=lr_lambda)


train_loss = []
test_loss = []


es = EarlyStopping()
for epoch in range(max_epoch):
    print("[epochs:{:2}]".format(epoch), end=' ')
    print("EPOCH "+str(epoch)+" TRAINING...")
    train_loop(train_dataloader, model, loss_fn, optimizer, train_loss, es, scheduler)
    test_loop(test_dataloader, model, loss_fn, test_loss)
    print("")

    if es.early_stop:
        print("EarlyStop Triggered : Bestscore = {:7.4g}".format(es.best_score))
        break
print("DONE")

plt.plot(train_loss[3:], label='train_loss')
plt.plot(test_loss[3:], label='test_loss')
plt.legend()
plt.title("Train/Test Loss (MSE)")
plt.show()


#coef_arr, rot_axis = model_return(input=torch.unsqueeze(torch.unsqueeze(X_test0[-1], dim=0), dim=0), model=model)
model.eval()
with torch.no_grad():
    coef_arr = model(torch.unsqueeze(torch.unsqueeze(X_test0[-1], dim=0), dim=0))
    coef_arr = torch.view_as_complex(DataPreProcessing.coef_unzip(coef_arr).reshape(-1, 2))
    y_test0_temp = torch.unsqueeze(torch.empty_like(y_test0[-1]).copy_(y_test0[-1]), dim=0)
    y_test0_temp = torch.view_as_complex(DataPreProcessing.coef_unzip(y_test0_temp).reshape([-1, 2]))
    print("y0 :", y_test0_temp)#[0, :-3]).reshape([-1, 2])))
    print("pred :", coef_arr)

    SHEcoefDisplay(y_test0_temp, coef_arr, l_max)
    
    
#print("y0 :", y_test0_temp[0, -3:])
#print("pred :", rot_axis)