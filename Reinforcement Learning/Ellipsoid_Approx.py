import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from tqdm import tqdm
from torchsummary import summary

from DataPreprocessing import DataPreProcessing

# Choose device to be used for learning
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")


class EllipsoidModel():
    def __init__(self, axes, N_set, tilt=(0, 0), lc_unit_len=100):
        self.Nphi = N_set[0]
        self.Ntheta = N_set[1]
        self.dphi = 2*np.pi/self.Nphi
        self.dtheta = np.pi/self.Ntheta
        self.coord_set = (self.dphi, self.dtheta, self.Nphi, self.Ntheta)
        self.pos_cart_arr = torch.zeros((self.Nphi+1, self.Ntheta+1, 3), requires_grad=True) #last index = first index 
        self.surf_vec_arr = torch.zeros((self.Nphi, self.Ntheta, 2, 3), requires_grad=True)
        self.lc_unit_len = lc_unit_len


        self.axes_R = torch.concat((axes[0], axes[1], axes[2]))
        self.tilt = torch.concat((tilt[0], tilt[1]))

        self.R_eps = torch.matmul(EllipsoidModel.rotArr(self.tilt[0], "z"), EllipsoidModel.rotArr(self.tilt[1], "y"))
        

    @staticmethod
    def rotArr(angle, axis):
        # for rotational matrix
        if axis == "x" or axis == 0:
            arr1 = torch.zeros(3, 3, requires_grad=True)
            arr1 = arr1.clone()
            arr1[0, 0] = 1.0
            arr2 = torch.zeros(3, 3, requires_grad=True)
            arr2 = arr2.clone()
            arr2[1, 1] = 1.0
            arr2[2, 2] = 1.0
            arr3 = torch.zeros(3, 3, requires_grad=True)
            arr3 = arr3.clone()
            arr3[1, 2] = -1.0
            arr3[2, 1] = 1.0
            arr = arr1 + torch.cos(angle) * arr2 + torch.sin(angle) * arr3
        elif axis == "y" or axis == 1:
            arr1 = torch.zeros(3, 3, requires_grad=True)
            arr1 = arr1.clone()
            arr1[1, 1] = 1.0
            arr2 = torch.zeros(3, 3, requires_grad=True)
            arr2 = arr2.clone()
            arr2[0, 0] = 1.0
            arr2[2, 2] = 1.0
            arr3 = torch.zeros(3, 3, requires_grad=True)
            arr3 = arr3.clone()
            arr3[0, 2] = 1.0
            arr3[2, 0] = -1.0
            arr = arr1 + torch.cos(angle) * arr2 + torch.sin(angle) * arr3
        elif axis == "z" or axis == 2:
            arr1 = torch.zeros(3, 3, requires_grad=True)
            arr1 = arr1.clone()
            arr1[2, 2] = 1.0
            arr2 = torch.zeros(3, 3, requires_grad=True)
            arr2 = arr2.clone()
            arr2[0, 0] = 1.0
            arr2[1, 1] = 1.0
            arr3 = torch.zeros(3, 3, requires_grad=True)
            arr3 = arr3.clone()
            arr3[0, 1] = 1.0
            arr3[1, 0] = -1.0
            arr = arr1 + torch.cos(angle) * arr2 + torch.sin(angle) * arr3
        else:
            raise ValueError("Unappropriate axis")
        return arr
    

    # generating with basic frame
    def base_fitting_generator(self, mode="ellipsoid"):
        if mode == "ellipsoid":
            generating_frame = self.__ellipsoid_frame 
        
        for i in range(self.Nphi):
            for j in range(self.Ntheta+1):
                phi_ij = (j%2)*(self.dphi/2) + i*self.dphi
                theta_ij = j*self.dtheta
                phi_ij0 = torch.tensor(phi_ij, requires_grad=True)
                theta_ij0 = torch.tensor(theta_ij, requires_grad=True)
                phi_ij = torch.unsqueeze(phi_ij0, 0)
                theta_ij = torch.unsqueeze(theta_ij0, 0)
                r_ij = generating_frame([phi_ij, theta_ij])

                x_ij = r_ij*torch.sin(theta_ij)*torch.cos(phi_ij)
                y_ij = r_ij*torch.sin(theta_ij)*torch.sin(phi_ij)
                z_ij = r_ij*torch.cos(theta_ij)             
                
                tempx = torch.zeros_like(self.pos_cart_arr)
                tempx[i, j, 0] = 1
                tempx = tempx.clone()
                tempy = torch.zeros_like(self.pos_cart_arr)
                tempy[i, j, 1] = 1
                tempy = tempy.clone()
                tempz = torch.zeros_like(self.pos_cart_arr)
                tempz[i, j, 2] = 1
                tempz = tempz.clone()

                self.pos_cart_arr = self.pos_cart_arr + x_ij * tempx
                self.pos_cart_arr = self.pos_cart_arr + y_ij * tempy
                self.pos_cart_arr = self.pos_cart_arr + z_ij * tempz


                if i == 0:
                    tempx = torch.zeros_like(self.pos_cart_arr)
                    tempx[self.Nphi, j, 0] = 1
                    tempx = tempx.clone()
                    tempy = torch.zeros_like(self.pos_cart_arr)
                    tempy[self.Nphi, j, 1] = 1
                    tempy = tempy.clone()
                    tempz = torch.zeros_like(self.pos_cart_arr)
                    tempz[self.Nphi, j, 2] = 1
                    tempz = tempz.clone()

                    self.pos_cart_arr = self.pos_cart_arr + x_ij * tempx
                    self.pos_cart_arr = self.pos_cart_arr + y_ij * tempy
                    self.pos_cart_arr = self.pos_cart_arr + z_ij * tempz
        

    def __ellipsoid_frame(self, direction):
        """
        ellipsoid generator
        a, b, c : radius corr. axis (default = axes_R)
        """
        a = self.axes_R[0]
        b = self.axes_R[1]
        c = self.axes_R[2]

        """
        tilt_angle = [longitude, latitude]
        * longitude angle : z-axis rotation
        * latitude angle : x-axis rotation
        """
        long = self.tilt[0]
        lat = self.tilt[1]
    
        self.tilt = torch.tensor([1.0, 0.0], requires_grad=True) * long
        self.tilt = self.tilt + torch.tensor([0.0, 1.0], requires_grad=True) * lat
        long_rot_arr = self.rotArr(-long, "z")
        lat_rot_arr = self.rotArr(-lat, "y")
        R_arr = torch.matmul(lat_rot_arr, long_rot_arr)

        
        A_arr = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)/a**2
        A_arr = A_arr + torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], requires_grad=True)/b**2
        A_arr = A_arr + torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], requires_grad=True)/c**2
        
        
        """
        coordinate direction : [phi, theta]
        output : corr. r value
        """
        phi_temp = direction[0]
        theta_temp = direction[1]
        u_vec = torch.concat((torch.sin(theta_temp)*torch.cos(phi_temp), torch.sin(theta_temp)*torch.sin(phi_temp), torch.cos(theta_temp)))
        u_vec = torch.transpose(torch.unsqueeze(u_vec, 0), 0, 1)
        r_temp = torch.matmul(torch.transpose(u_vec, 0, 1), torch.transpose(R_arr, 0, 1))
        r_temp = torch.matmul(r_temp, A_arr)
        r_temp = torch.matmul(r_temp, R_arr)
        r_temp = torch.matmul(r_temp, u_vec)
        r_temp = 1 / torch.sqrt(r_temp)

        return r_temp    
    
    def surf_vec_cal(self):
        for i in range(self.Nphi):
            for j in range(self.Ntheta):
                if j%2 == 0:
                    v11 = self.pos_cart_arr[i+1, j] - self.pos_cart_arr[i, j+1]
                    v12 = self.pos_cart_arr[i+1, j+1] - self.pos_cart_arr[i, j+1]
                    v21 = self.pos_cart_arr[i+1, j] - self.pos_cart_arr[i, j]
                    v22 = self.pos_cart_arr[i, j+1] - self.pos_cart_arr[i, j]
                elif j%2 == 1:
                    v11 = self.pos_cart_arr[i+1, j+1] - self.pos_cart_arr[i, j]
                    v12 = self.pos_cart_arr[i, j+1] - self.pos_cart_arr[i, j]
                    v21 = self.pos_cart_arr[i+1, j] - self.pos_cart_arr[i, j]
                    v22 = self.pos_cart_arr[i+1, j+1] - self.pos_cart_arr[i, j]
                
                temp1 = torch.zeros_like(self.surf_vec_arr)
                temp1[i, j, 0] = temp1[i, j, 0] + 1
                temp1 = temp1.clone()
                temp2 = torch.zeros_like(self.surf_vec_arr)
                temp2[i, j, 1] = temp2[i, j, 1] + 1
                temp2 = temp2.clone()

                self.surf_vec_arr = self.surf_vec_arr + temp1 * 0.5*torch.cross(v11, v12)
                self.surf_vec_arr = self.surf_vec_arr + temp2 * 0.5*torch.cross(v21, v22)



    def __orb2geo(self, vec_orb, rot_angle):
        rot_angle = torch.tensor(rot_angle, requires_grad=True)
        res = torch.matmul(EllipsoidModel.rotArr(-rot_angle, "z"), self.R_eps)
        res = torch.matmul(res, vec_orb)
        return res
    
    def lc_gen(self, lc_info, flux0=1):
        Sdir = torch.tensor(lc_info[0:3], requires_grad=True)
        Edir = torch.tensor(lc_info[3:6], requires_grad=True)
        Sdir = torch.unsqueeze(Sdir, 0)
        Edir = torch.unsqueeze(Edir, 0)
        #rot_axis = lc_info[6:9]
        N_arr = self.surf_vec_arr / torch.sqrt(torch.abs(self.surf_vec_arr)+1e-15)
        N_arr = N_arr.reshape(-1, 3)

        generated_lc = torch.zeros(self.lc_unit_len, requires_grad=True)
        generated_lc = generated_lc.clone()
        for t in range(self.lc_unit_len):
            theta_t = 2*np.pi*t/self.lc_unit_len
            Edir_t = torch.matmul(torch.transpose(self.R_eps, 0, 1), self.__orb2geo(torch.transpose(Edir, 0, 1), theta_t)) #Edir(0) -> Edir(t)
            Sdir_t = torch.matmul(torch.transpose(self.R_eps, 0, 1), self.__orb2geo(torch.transpose(Sdir, 0, 1), theta_t)) #Sdir(0) -> Sdir(t)
            Edir_t = Edir_t.clone().detach().requires_grad_(True)
            Sdir_t = Sdir_t.clone().detach().requires_grad_(True)
            Edir_t = Edir_t / torch.norm(Edir_t)
            Sdir_t = Sdir_t / torch.norm(Sdir_t)
            ReLU = nn.ReLU()
            temp1 = torch.matmul(N_arr, Edir_t)
            temp1 = ReLU(temp1)
            temp2 = torch.matmul(N_arr, Sdir_t)
            temp2 = ReLU(temp2)
            temp1 = torch.squeeze(temp1)
            temp2 = torch.squeeze(temp2)
            generated_lc[t] = torch.dot(temp1, temp2)
        generated_lc = flux0 * generated_lc

        return generated_lc

    @staticmethod
    def lc_mean(input_lc):
        """
        input_lc = [LC Length]
        """
        lc_len = input_lc.shape[-1]
        lc_mean0 = (torch.sum(input_lc) - (input_lc[0] + input_lc[-1])/2) / lc_len
        return lc_mean0


class EllipsoidInversion():
    def __init__(self, repeat_num, lr, max_epoch, N_set=(20, 10), lc_unit_len=100):
        self.repeat_num = repeat_num
        self.lr = lr
        self.max_epoch = max_epoch
        self.N_set = N_set
        self.lc_unit_len = lc_unit_len
        
    def opt(self, target_lc, lc_info):
        target_lc = torch.tensor(target_lc) if not torch.is_tensor(target_lc) else target_lc
        lc_info = torch.tensor(lc_info) if not torch.is_tensor(lc_info) else lc_info

        a0 = torch.tensor(5.0, requires_grad=True)
        a = torch.unsqueeze(a0, 0)

        b0 = torch.randn(1, requires_grad=True)
        c0 = torch.randn(1, requires_grad=True)
        phi0 = torch.randn(1, requires_grad=True)
        theta0 = torch.randn(1, requires_grad=True)

        params = [b0, c0, phi0, theta0]
        optimizer = optim.SGD(params, lr=self.lr)
        lossfn = nn.MSELoss()

        loss_min = 1e+8
        min_param = None
        min_pred = None
        for epoch in range(self.max_epoch):
            b = 3*b0+5
            c = 3*c0+5
            phi = np.pi*phi0/2 + np.pi/2
            theta = np.pi*theta0/2 + np.pi/2

            ell = EllipsoidModel((a, b, c), self.N_set, (phi, theta), self.lc_unit_len)
            ell.base_fitting_generator()
            ell.surf_vec_cal()
            pred = ell.lc_gen(lc_info, 1)
            pred = pred * (EllipsoidModel.lc_mean(target_lc) / EllipsoidModel.lc_mean(pred))
            
            amp = torch.max(target_lc) - torch.min(target_lc)
            #pred_diff = torch.diff(pred)
            #target_diff = torch.diff(target_lc)
            #amp_diff = torch.max(pred_diff) - torch.min(pred_diff)
            
            
            loss0 = lossfn(pred/amp, target_lc/amp)
            #loss1 = lossfn(pred_diff, target_diff)
            loss0 = 100 * torch.sqrt(loss0)
            #loss1 = 5e-4 * loss1
            loss = loss0#loss1 + 1.5*loss0

            """
            loss = torch.mean((50*(target_lc - pred)/amp)**2) #40
            loss_i = 110*torch.trapezoid(torch.abs(target_lc-pred))/(100*amp)
            loss_d = torch.mean((9*(torch.diff(target_lc)-torch.diff(pred)))**2)
            loss = (1.2*loss + loss_i + loss_d)*3/10
            """

            if loss_min > loss.item():
                loss_min = loss.item()
                min_param = params.copy()
                min_pred = pred.clone().detach().numpy()
            
            if loss < -23:
                break

            optimizer.zero_grad()  # Clear the gradients
            loss.backward()  # Backpropagate the loss
            optimizer.step()  # Update parameters using the gradients

            if epoch % 5 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item()}")

        #pred = pred.detach().numpy()
        #target_lc = target_lc.detach().numpy()
        #pred_diff = pred_diff.detach().numpy()
        #target_diff = target_diff.detach().numpy()

        #self.show(pred, target_lc, pred_diff, target_diff)
        #self.show(min_pred, target_lc, 0, 0)
        
        return params, min_param, loss_min

    
    def show(self, pred, target_lc, pred_diff, target_diff):
        plt.plot(pred, label="pred")
        plt.plot(target_lc, label="target")
        plt.plot(pred_diff, label="pred_diff")
        plt.plot(target_diff, label="target_diff")
        plt.legend()
        plt.show()

'''
lc_unit_len = 100
max_epoch = 50
N_set = (20, 10)
l_max = 8
use_ratio = 0.3
batch_size = 64
trainset_ratio = 0.7
learning_rate = 4e-3

data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_total.npz"

dataPP = DataPreProcessing(data_path=data_path)
dataPP.X_total = torch.concat((dataPP.X_total[:, :100], dataPP.X_total[:, -9:]), dim=-1)
dataPP.Y_total = dataPP.Y_total[:, 0:(l_max+1)**2]
dataPP.coef2R(dataPP.Y_total, l_max=l_max, N_set=N_set)
dataPP.merge(merge_num=1, ast_repeat_num=10, lc_len=lc_unit_len, dupl_ratio=0.1)
dataPP.X_total = dataPP.X_total[:int(dataPP.dataset_len*use_ratio)]
dataPP.Y_total = dataPP.Y_total[:int(dataPP.dataset_len*use_ratio)]
dataPP.dataset_len = int(dataPP.dataset_len*use_ratio)

data_no = 1
target_lc = dataPP.X_total[data_no, :-9]
lc_info = dataPP.X_total[data_no, -9:]
print(lc_info)

EllInv = EllipsoidInversion(repeat_num=1, lr=learning_rate, max_epoch=max_epoch, N_set=N_set)
param_res, _ = EllInv.opt(target_lc, lc_info)
print(param_res)

"""
a0 = torch.tensor(5.0, requires_grad=True)
a = torch.unsqueeze(a0, 0)

b0 = torch.randn(1, requires_grad=True)
c0 = torch.randn(1, requires_grad=True)
phi0 = torch.randn(1, requires_grad=True)
theta0 = torch.randn(1, requires_grad=True)




b0 = torch.tensor(4.0, requires_grad=True)
c0 = torch.tensor(7.0, requires_grad=True)
phi0 = torch.tensor(0.0, requires_grad=True)
theta0 = torch.tensor(0.0, requires_grad=True)

b = torch.unsqueeze(b0, 0)
c = torch.unsqueeze(c0, 0)
phi = torch.unsqueeze(phi0, 0)
theta = torch.unsqueeze(theta0, 0)

params = [b0, c0, phi0, theta0]
optimizer = optim.SGD(params, lr=learning_rate)
lossfn = nn.MSELoss()

for epoch in range(max_epoch):
    b = 2*b0+5
    c = 2*c0+5
    phi = np.pi*phi0
    theta = np.pi*theta0

    ell = EllipsoidModel((a, b, c), N_set, (phi, theta))
    ell.base_fitting_generator()
    ell.surf_vec_cal()
    pred = ell.lc_gen(lc_info, 1)
    pred = pred * (EllipsoidModel.lc_mean(target_lc) / EllipsoidModel.lc_mean(pred))
    target_lc = target_lc

    amp = torch.max(target_lc) - torch.min(target_lc)
    pred_diff = torch.diff(pred)
    target_diff = torch.diff(target_lc)
    amp_diff = torch.max(target_diff) - torch.min(target_diff)
    
    loss0 = lossfn(pred/amp, target_lc/amp)
    loss1 = lossfn(100*pred_diff, 100*target_diff)
    loss0 = 100 * torch.sqrt(loss0)
    loss1 = 1e-2 * loss1
    loss = loss1 + 0.2*loss0
    
    if loss < -23:
        break

    optimizer.zero_grad()  # Clear the gradients
    loss.backward()  # Backpropagate the loss
    optimizer.step()  # Update parameters using the gradients

    if epoch % 5 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item()}")

pred = pred.detach().numpy()
target_lc = target_lc.detach().numpy()
pred_diff = pred_diff.detach().numpy()
target_diff = target_diff.detach().numpy()
print(params)

plt.plot(pred, label="pred")
plt.plot(target_lc, label="target")
plt.plot(pred_diff, label="pred_diff")
plt.plot(target_diff, label="target_diff")
plt.legend()
plt.show()
"""
'''
