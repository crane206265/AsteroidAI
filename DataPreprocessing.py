import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils import data
from itertools import combinations
from scipy.special import factorial
from tqdm import tqdm
import random
from scipy.special import sph_harm_y


random.seed(1)
np.random.seed(1)

class DataPreProcessing():
    def __init__(self, data_path, ell_approx=False):
        # Dataset Preparation
        total_data = np.load(data_path)
        self.X_total = torch.tensor(total_data['X_total'].astype(np.float32))
        self.Y_total = torch.tensor(total_data['Y_total'].astype(np.complex64))
        self.dataset_len = self.X_total.shape[0]
        self.merge_num = 1

        # complex to real
        self.Y_total = torch.view_as_real(self.Y_total).type(torch.float32)
        self.Y_total = torch.flatten(self.Y_total, 1)
        self.Y_total = torch.cat((self.Y_total[:, :-15], self.Y_total[:, -14::2]), dim=1)
        self.Y_total = torch.cat((DataPreProcessing.coef_zip(self.Y_total[:, :-(3+5)]), self.Y_total[:, -(3+5):]), dim=1)
        self.X_total = torch.cat((self.X_total, self.Y_total[:, -(3+5):]), dim=1)
        self.Y_total = self.Y_total[:, :-(3+5)]

        #self.X_total = torch.cat((self.X_total[..., :100], self.X_total[..., -6:]), dim=1)

        self.coef2R_prepared = False


    def coef2R(self, coef_arr, l_max=8, N_set=(40, 20)):
        if not self.coef2R_prepared:
            self.__coef2R_prepare(l_max, N_set)

        coef_arr_complex = torch.view_as_complex(DataPreProcessing.coef_unzip(coef_arr).reshape(coef_arr.shape[0], -1, 2))
        self.Y_total = torch.real(torch.tensordot(coef_arr_complex, self.sph_values, dims=([-1], [0]))) #[Data Number X Nphi*Ntheta]
        self.Y_total = torch.flatten(self.Y_total, start_dim=-2)

    def __coef2R_prepare(self, l_max, N_set):
        PI = 3.1415926535
        self.Nphi, self.Ntheta = N_set[0], N_set[1]
        self.dphi, self.dtheta = 2*PI/self.Nphi, PI/self.Ntheta
        self.l_max = l_max
        self.sph_values = torch.zeros((self.l_max+1)**2, self.Nphi, self.Ntheta, dtype=torch.complex64)

        for l in range(self.l_max+1):
            for m in range(-l, l+1):
                for i in range(self.Ntheta):
                    for j in range(self.Nphi):
                        phi_ij = (i%2)*(self.dphi/2) + j*self.dphi
                        theta_ij = i*self.dtheta
                        self.sph_values[l**2+l+m, j, i] = torch.from_numpy(sph_harm_y(l, m, theta_ij, phi_ij))
        
        self.coef2R_prepared = True


    def merge(self, merge_num:int, ast_repeat_num=10, lc_len=200, dupl_ratio = 0.3):
        """
        merge lightcurves
        merge_num <= ast_repeat_num (lc_num in DataGenerator.py) = 10
        * use before shuffling
        """
        self.merge_num = merge_num
        ast_num = self.X_total.shape[0]//ast_repeat_num
        combi_num0 = int(factorial(ast_repeat_num)/(factorial(merge_num)*factorial(ast_repeat_num-merge_num)))
        combi_num = int(combi_num0 * dupl_ratio)
        new_X_len = combi_num * ast_num
        new_X = torch.zeros(new_X_len, (lc_len+9+5)*merge_num)
        new_Y = torch.zeros(new_X_len, self.Y_total.shape[-1])
        print("MERGING DATASET")
        i = 0
        for ast in tqdm(range(ast_num)):
            combi_list = list(combinations(range(ast_repeat_num), merge_num))
            random.shuffle(combi_list)
            for combi in combi_list[:combi_num]:
                j = 0
                for cb_idx in combi:
                    new_X[i, j*lc_len:(j+1)*lc_len] = self.X_total[ast*ast_repeat_num+cb_idx, :lc_len]
                    new_X[i, lc_len*merge_num+j*(9+5):lc_len*merge_num+(j+1)*(9+5)] = self.X_total[ast*ast_repeat_num+cb_idx, -(9+5):]
                    new_Y[i, :] = self.Y_total[ast*ast_repeat_num+cb_idx, :]
                    j += 1
                i += 1
        
        print(new_X.shape, new_Y.shape)
        self.X_total = new_X
        self.Y_total = new_Y
        self.dataset_len = self.X_total.shape[0]


###### 이 밑으로 ell_approx 수정 안함!! #####
    #raise Warning("Ellipsoid Approximation is not implemented below")

    def scale_data(self, **kwargs):
        self.scaler = DataScaling(self.Y_total)#[:, :-3])
        mode = kwargs["mode"]
        
        if mode == "standard_normalization":
            #self.Y_total[:, :-3] = self.scaler.std_scale(self.Y_total[:, :-3])
            self.Y_total = self.scaler.std_scale(self.Y_total)
        elif mode == "exponential":
            exp_rescale = kwargs["exp"]
            lin_rescale = kwargs["linear"]
            #self.Y_total[:, :-3] = self.scaler.exp_scale(self.Y_total[:, :-3], exp_rescale, lin_rescale)
            self.Y_total = self.scaler.exp_scale(self.Y_total, exp_rescale, lin_rescale)
        elif mode == "lc_scaling":
            scaled_mean = kwargs["scaled_mean"]
            self.X_total[..., :int(-6*self.merge_num)], self.Y_total = self.scaler.lc_scale(self.X_total[..., :int(-6*self.merge_num)], self.Y_total, scaled_mean=scaled_mean)
        elif mode == "dir_scaling":
            scaled_size = kwargs["scaled_size"]
            self.X_total[..., int(-9*self.merge_num):] = scaled_size * self.X_total[..., int(-9*self.merge_num):]

    def return_data(self, type):
        if type=='X':
            return self.X_total
        elif type=='Y':
            return self.Y_total
        elif type=='ALL':
            return self.X_total, self.Y_total

    def train_test_split(self, trainset_ratio=0.7):
        #shuffle
        shuffle_idx = np.arange(0, self.dataset_len)
        np.random.shuffle(shuffle_idx)
        self.X_total = self.X_total[shuffle_idx, :]
        self.Y_total = self.Y_total[shuffle_idx, :]

        #split
        X_train0 = self.X_total[:int(self.dataset_len*trainset_ratio)]
        X_test0 = self.X_total[int(self.dataset_len*trainset_ratio):]
        y_train0 = self.Y_total[:int(self.dataset_len*trainset_ratio)]
        y_test0 = self.Y_total[int(self.dataset_len*trainset_ratio):]

        return X_train0, X_test0, y_train0, y_test0
    
    def return_dataloader(self, X, y, batch_size):
        dataset = LC_dataset(X, y)
        return data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    
    @staticmethod
    def coef_zip(coef_arr):
        """
        zip the coef_arr
        2(l_max+1)^2 -> (l_max+1)^2
        coef_arr = [Data_Number X coef_Number]
        """
        l_max = int(np.round(np.sqrt(coef_arr.shape[-1]/2)-1))
        if coef_arr.dim() == 1:
            coef_arr_zip = torch.unsqueeze(torch.zeros((l_max+1)**2), dim=0)
            coef_arr = torch.unsqueeze(coef_arr, dim=0)
        else:
            coef_arr_zip = torch.zeros(coef_arr.shape[0], (l_max+1)**2)

        for l in range(0, l_max+1):
            for m in range(0, l+1):
                if m != 0:
                    for real in range(2): #real : 0, imaginary : 1
                        coef_arr_zip[:, l**2+2*m-(1-real)] = coef_arr[:, 2*(l**2+l+m)+real]
                else:
                    coef_arr_zip[:, l**2] = coef_arr[:, 2*(l**2+l)]

        return coef_arr_zip
    
    @staticmethod
    def coef_unzip(coef_arr_zip):
        """
        unzip the coef_arr
        (l_max+1)^2 -> 2(l_max+1)^2
        coef_arr_zip = [Data_Number X coef_Number]
        """
        l_max = int(np.round(np.sqrt(coef_arr_zip.shape[-1])-1))
        if coef_arr_zip.dim() == 1:
            coef_arr = torch.unsqueeze(torch.zeros(2*((l_max+1)**2)), dim=0)
            coef_arr_zip = torch.unsqueeze(coef_arr_zip, dim=0)
        elif coef_arr_zip.dim() == 2:
            coef_arr = torch.zeros(coef_arr_zip.shape[0], 2*((l_max+1)**2))
        elif coef_arr_zip.dim() == 3:
            coef_arr = torch.zeros(coef_arr_zip.shape[0], coef_arr_zip.shape[1], 2*((l_max+1)**2))
        #print(coef_arr.shape, coef_arr_zip.shape, "unzip")
        
        for l in range(0, l_max+1):
            for m in range(-l, l+1):
                for real in range(2):
                    idx = 2*(l**2+l+m)+real
                    if m > 0:
                        coef_arr[..., idx] = coef_arr_zip[..., l**2+2*m-(1-real)]
                    elif m == 0 and real == 0:
                        coef_arr[..., idx] = coef_arr_zip[..., l**2]# if real == 0 else torch.zeros(coef_arr_zip.shape[0])
                    else:
                        coef_arr[..., idx] = ((-1)**(m+real))*coef_arr_zip[..., l**2-2*m-(1-real)]

        return coef_arr                        



class LC_dataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor):
        super().__init__()

        self.x = x_tensor
        self.y = y_tensor

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.x)
    

class DataScaling():
    def __init__(self, epsilon=1e-8):
        """
        data : [Number of Data X Feature Number]
        """
        self.epsilon = epsilon

    def std_scale(self, input:torch.Tensor):
        self.mean_arr = torch.mean(input=input, dim=0, keepdim=True)
        self.std_arr = torch.std(input=input, dim=0, keepdim=True, correction=0) + self.epsilon
        mean_ = self.mean_arr.repeat(input.shape[0], 1)
        std_ = self.std_arr.repeat(input.shape[0], 1)
        return (input-mean_)/std_

    def std_unscale(self, input=torch.Tensor):
        mean_ = self.mean_arr.repeat(input.shape[0], 1)
        std_ = self.std_arr.repeat(input.shape[0], 1)
        return input*std_ + mean_
    
    def exp_scale(self, input:torch.Tensor, exp_rescale, lin_rescale):
        """
        new = sgn(old) * (rescale)^(log10(|old|))
        ==> rescale == 10 : no scaling
        """
        self.exp_rescale = exp_rescale
        self.lin_rescale = lin_rescale
        return lin_rescale * torch.sign(input) * torch.pow(exp_rescale, torch.log10(torch.abs(input)))

    def exp_unscale(self, input:torch.Tensor):
        return torch.sign(input) * torch.pow(10**(1/np.log(self.rescale)), torch.log(torch.abs(input)/self.lin_rescale))
    

    # WARNING : DIDN'T EDIT WITH MERGING
    def lc_scale(self, input_lc:torch.Tensor, input_coef:torch.Tensor, scaled_mean=10):
        """
        scaling lc and coef for <lc_mean0> to be <scaled_mean>
        """
        lc_mean0 = self.__lc_mean(input_lc)
        print(lc_mean0)
        ratio = scaled_mean*torch.ones_like(lc_mean0) / lc_mean0
        ratio = torch.unsqueeze(ratio, dim=-1)
        new_lc = input_lc * ratio
        new_coef = input_coef * torch.sqrt(ratio)

        return new_lc, new_coef

    def __lc_mean(self, input_lc:torch.Tensor):
        """
        input_lc = [Data Number X LC Length]
        """
        lc_len = input_lc.shape[-1]
        lc_mean0 = (torch.sum(input_lc, dim=-1) - (input_lc[..., 0] + input_lc[..., -1])/2) / lc_len
        return lc_mean0
