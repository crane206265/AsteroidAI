import torch
import numpy as np
from torch import nn
from scipy.special import sph_harm_y

from DataPreprocessing import DataPreProcessing

class CoefAxisLoss(nn.Module):
    def __init__(self, relative=True, percent=False, frac_reg=False):
        super().__init__()
        self.relative = relative
        self.percent = percent
        self.frac_reg = frac_reg

    def forward(self, input, target, alpha=0):
        """
        Use MSE for coef loss function
        Use cosine for rot_axis loss function
        - alpha : weight of cosine loss
        """
        loss = 0
        input_coef = input[:, :-3]
        input_axis = nn.functional.normalize(input[:, -3:], dim=-1)
        target_coef = target[:, :, :-3]
        target_axis = nn.functional.normalize(torch.squeeze(target[:, :, -3:]), dim=-1)

        torch_MSE = nn.MSELoss()
        torch_cos = nn.CosineEmbeddingLoss()
        #loss = torch.sqrt(torch_MSE(input_coef, target_coef)) + alpha * torch_cos(input_axis, target_axis, torch.ones(input_axis.shape[0]))
        
        if self.relative:
            loss = torch_MSE(input/(target+1e-6), target/(target+1e-6))
            loss = torch.sqrt(loss + 1e-6)
        else:
            loss = torch.sqrt(torch_MSE(input, target))
        if self.percent:
            loss = 100 * loss

        # Need to modify avoid input~0
        # NO USE : Instead transfer learning
        if self.frac_reg:
            reg = torch.mean((1e-2 * 1/(input+1e-8)))
            #print(reg, "reg")
            """
            if torch.abs(reg) > 1e+15:
                reg = 1e+15
            elif torch.abs(reg) < 1e-8:
                reg = 0
            """
            loss = loss + reg

        return loss
    
    @staticmethod
    def CoefLoss(input, target):
        torch_MSE = nn.MSELoss()
        #return torch.sqrt(torch_MSE(input[:, :-3], target[:, :, :-3]))
        return torch.sqrt(torch_MSE(input, target))
    
    @staticmethod
    def AxisLoss(input, target):
        torch_cos = nn.CosineEmbeddingLoss()
        return torch_cos(input[:, -3:], torch.squeeze(target[:, :, -3:]), torch.ones(input.shape[0]))
    


class RLoss(nn.Module):
    def __init__(self, l_max, relative=False, percent=False, no_grad=False):
        if not no_grad:
            super().__init__()

        PI = 3.1415926535
        self.Nphi, self.Ntheta = 40, 20
        self.dphi, self.dtheta = 2*PI/self.Nphi, PI/self.Ntheta
        self.l_max = l_max
        self.sph_values = torch.zeros((self.l_max+1)**2, self.Nphi, self.Ntheta, dtype=torch.complex64)

        for l in range(self.l_max+1):
            for m in range(-l, l+1):
                for i in range(self.Ntheta):
                    for j in range(self.Nphi):
                        self.sph_values[l**2+l+m, j, i] = torch.from_numpy(sph_harm_y(l, m, i*self.dtheta, j*self.dphi))

        self.sph_values = self.sph_values.reshape(-1, self.Nphi*self.Ntheta)

        self.relative = relative
        self.percent = percent

    def forward(self, input, target):
        """
        input, target : zip-coef arr[Data Number X 1 X coef Number]
        """
        input = torch.view_as_complex(DataPreProcessing.coef_unzip(input).reshape(input.shape[0], -1, 2))
        target = torch.view_as_complex(DataPreProcessing.coef_unzip(target).reshape(input.shape[0], 1, -1, 2))
        #print(input.shape, target.shape, "RLOSS")
        input_r = torch.real(torch.tensordot(input, self.sph_values, dims=([-1], [0]))) #[Data Number X Nphi*Ntheta]
        target_r = torch.real(torch.tensordot(target, self.sph_values, dims=([-1], [0])))

        torch_MSE = nn.MSELoss()
        loss = torch.sqrt(torch_MSE(input_r, target_r))

        if self.relative:
            loss = torch.sqrt(torch_MSE(input_r/target_r, target_r/target_r))
        else:
            loss = torch.sqrt(torch_MSE(input_r, target_r))
        #loss = loss / torch.sqrt(torch_MSE(target_r, torch.zeros_like(target_r)))

        if self.percent:
            loss = 100*loss

        return loss
    

class ComplexLoss(nn.Module):
    def __init__(self, alpha, l_max, relative=True, percent=False):
        super().__init__()

        self.alpha = alpha
        self.RLoss = RLoss(l_max=l_max, relative=relative, percent=percent)
        self.CoefLoss = CoefAxisLoss(relative=relative, percent=percent)

    def forward(self, input, target):
        return self.RLoss(input, target) + self.alpha * self.CoefLoss(input, target)