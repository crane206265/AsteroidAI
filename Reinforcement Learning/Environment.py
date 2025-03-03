import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torch

from DataPreprocessing import DataPreProcessing
from Asteroid_Model import AsteroidModel

PI = 3.1415926535


class AstEnv():
    def __init__(self, target_lc, lc_info, lc_num, N_set=(40, 20), lc_unit_len=200):
        """
        target_lc
        lc_info : [sun_dir, earth_dir, rot_axis]
        lc_num
        obs_set = (obs_lc, obs_time) : what lc to obs, where to obs
        N_set = (Nphi, Ntheta) : asteroid model grid splitting number
        """
        self.lc_unit_len = lc_unit_len
        self.target_lc = target_lc
        self.lc_info = lc_info
        self.lc_num = lc_num

        self.rot_axis = lc_info[-3:]
        initial_eps = np.empty(2)
        initial_eps[0] = np.arctan2(self.rot_axis[1], self.rot_axis[0])
        initial_eps[1] = np.arccos(self.rot_axis[2]/LA.norm(self.rot_axis))
        self.R_eps = AsteroidModel.rotArr(initial_eps[0], "z")@AsteroidModel.rotArr(initial_eps[1], "y")

        self.__set_obs_params()
        """
        self.obs_lc_num = np.random.randint(lc_num)
        self.obs_time = np.argmax(target_lc[lc_unit_len*(self.obs_lc_num):lc_unit_len*(self.obs_lc_num+1)])
        self.obs_lc_full = self.target_lc[self.lc_unit_len*(self.obs_lc_num):self.lc_unit_len*(self.obs_lc_num+1)]
        self.obs_lc_info = lc_info[9*(self.obs_lc_num):9*(self.obs_lc_num+1)]
        self.obs_phi, self.obs_theta = self.__obs_dir_cal()
        """

        self.ast_obs_unit_step = 4
        self.lc_obs_unit_step = 4

        self.Nphi, self.Ntheta = N_set[0], N_set[1]
        self.dphi, self.dtheta = 2*PI/self.Nphi, PI/self.Ntheta
        self.reward_threshold = 90
        self.total_threshold = 90

        # Initialize asteroid
        self.lc_pred = np.zeros(self.lc_unit_len*self.lc_num)
        self.ast_backup = None
        self.reset()
        self.ast_backup = self.ast.copy()
        """
        self.__ellipsoid()
        self.ast = AsteroidModel(axes=self.R_set, N_set=N_set, tilt_mode="assigned", tilt=self.tilt, interior_cal=False)
        self.ast.base_fitting_generator(mode="ellipsoid")


        self.lc_pred = np.zeros(self.lc_unit_len*self.lc_num)
        self.step((0, 100, 0, 0)) #initialize lc_pred
        """
        

    def __ellipsoid(self):
        """
        Generate initial ellipsoid s.t. most fit to target_lc
        """
        self.R_set = np.zeros(3)+5
        self.tilt = np.zeros(2)
        pass

        self.R_set = tuple(self.R_set)
        self.tilt = tuple(self.tilt)

    def __obs_dir_cal(self):
        """
        calculate obs_phi, obs_theta : (direction to earth in geocentric coord.)
        """
        Edir = self.__orb2geo((self.obs_lc_info[3:6]).T, 2*PI*self.obs_time/self.lc_unit_len)
        Edir_phi = np.arctan2(Edir[1], Edir[0])
        Edir_theta = np.arccos(Edir[2]/LA.norm(Edir))
        return Edir_phi, Edir_theta

    def __orb2geo(self, vec_orb, rot_angle):
        return AsteroidModel.rotArr(-rot_angle, "z")@self.R_eps@vec_orb
    
    
    def obs(self):
        #r_arr obs
        obs_r_arr = np.zeros((self.Ntheta*self.Nphi)//(2*self.ast_obs_unit_step))
        idx_j0 = round(self.obs_theta/self.dtheta)
        idx_i0 = round((self.obs_phi - (idx_j0%2)*self.dphi/2)/self.dphi)

        for i in range(self.Nphi//2):
            for j in range(self.Ntheta):
                idx = i*self.Ntheta+j
                idx_i = (idx_i0 + (i-self.Nphi//4)%self.ast_obs_unit_step)%self.Nphi
                idx_j = (idx_j0 + (j-self.Ntheta//2)%self.ast_obs_unit_step)%self.Ntheta
                if idx < obs_r_arr.shape[0]:
                    obs_r_arr[idx] = self.ast.pos_sph_arr[idx_i, idx_j, 0]

        #lc obs
        obs_lc = np.zeros(2*(self.lc_unit_len//(4*self.lc_obs_unit_step)))
        k0 = 0
        for k1 in range((self.obs_time-self.lc_unit_len//(4*self.lc_obs_unit_step))%self.lc_unit_len, (self.obs_time+self.lc_unit_len//(4*self.lc_obs_unit_step))%self.lc_unit_len, self.lc_obs_unit_step):
            obs_lc[k0] = self.obs_lc_full[k1]
            
        
        #obs_lc = self.obs_lc_full[self.obs_time-self.lc_unit_len//(4*self.lc_obs_unit_step):self.obs_time+self.lc_unit_len//(4*self.lc_obs_unit_step)]
        #if obs_lc.shape[0] == 0:
        #    print(self.obs_time, self.obs_time-self.lc_unit_len//(4*self.lc_obs_unit_step), self.obs_time+self.lc_unit_len//(4*self.lc_obs_unit_step))

        obs_tensor = np.concatenate((obs_r_arr, obs_lc))
        return obs_tensor
    
    def step(self, action, mode='ratio_assign'):
        """
        action = [R_cut, r, phi, theta]
        if mode == 'ratio_assign'
            action = [phi, theta, r_cen_ratio, R_cut_ratio]
        """
        done = False
        all_done = False
        passed = False
        if mode == 'ratio_assign':
            phi_action = (self.obs_phi/(2*PI) + action[0])%1 # action be a relative ratio from obs_phi
            theta_action = (self.obs_theta/PI + action[1])%1 # action be a relative ratio from obs_theta
            self.ast.cut_ast(1, 0, True, pos_sph=(phi_action, theta_action, action[2], action[3]))
        elif mode == 'coord_assign':
            cut_sph_pos = AsteroidModel.sph2cart((action[1], action[2], action[3]))
            self.ast.cut_ast(1, 0, True, pos_sph=(action[0], cut_sph_pos[0], cut_sph_pos[1], cut_sph_pos[2]))
        else:
            raise NotImplementedError

        # Maintaining the mean of asteroid r_arr
        mean0 = 7
        r_arr_mean = np.mean(self.ast.pos_sph_arr[:, :, 0])
        self.ast.pos_sph_arr[:, :, 0] = self.ast.pos_sph_arr[:, :, 0] * mean0 / r_arr_mean
        self.ast.pos_cart_arr = self.ast.pos_cart_arr * mean0 / r_arr_mean

        self.ast.surf_vec_cal()

        reward = self.reward(init=100.0, rooted=False)
        #print(reward, "REWARD")
        if reward > self.reward_threshold:
            done = True
            passed = True
            all_done = self.__test__all()
        elif reward <= 20:
            done = True
            passed = False

        observation = self.obs()

        return observation, reward, done, (passed, all_done)

    def reward(self, init=100, rooted=False, include_other=False):
        reward = init
        for i in range(self.lc_num):
            if not include_other and i != self.obs_lc_num:
                continue
            target_lc_temp = self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)]
            target_lc_mean = self.__lc_mean(target_lc_temp)

            lc_temp = self.__lc_gen(self.lc_info[9*i:9*(i+1)]) #generate lc
            lc_temp = lc_temp * target_lc_mean / self.__lc_mean(lc_temp) #scaling lc_temp compared with target_lc_temp
            self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)] = lc_temp
            
            loss = np.mean((target_lc_temp - lc_temp)**2)
            if rooted:
                loss = np.sqrt(loss+1e-15)
            reward = reward - loss
        
        return reward
    
    def reset(self):
        if self.ast_backup == None:
            self.__ellipsoid()
            self.ast = AsteroidModel(axes=self.R_set, N_set=(self.Nphi, self.Ntheta), tilt_mode="assigned", tilt=self.tilt, interior_cal=False)
            self.ast.base_fitting_generator(mode="ellipsoid")
        else:
            #if not passed:
            #    self.ast = self.ast_backup.copy()
            self.__set_obs_params()

        self.step((0, 100, 0, 0)) #initialize/recalculate lc_pred

        return self.obs()
    
    def __set_obs_params(self):
        self.obs_lc_num = np.random.randint(self.lc_num)
        self.obs_time = np.argmax(self.target_lc[self.lc_unit_len*(self.obs_lc_num):self.lc_unit_len*(self.obs_lc_num+1)])
        self.obs_lc_full = self.target_lc[self.lc_unit_len*(self.obs_lc_num):self.lc_unit_len*(self.obs_lc_num+1)]
        self.obs_lc_info = self.lc_info[9*(self.obs_lc_num):9*(self.obs_lc_num+1)]
        self.obs_phi, self.obs_theta = self.__obs_dir_cal()

    def __lc_gen(self, lc_info, flux0=100):
        Sdir = lc_info[0:3]
        Edir = lc_info[3:6]
        #rot_axis = lc_info[6:9]
        N_arr = self.ast.surf_vec_arr / np.sqrt(np.abs(self.ast.surf_vec_arr)+1e-15)
        N_arr = N_arr.reshape(-1, 3)

        generated_lc = np.zeros(self.lc_unit_len)
        for t in range(self.lc_unit_len):
            theta_t = 2*PI*t/self.lc_unit_len
            Edir_t = self.R_eps.T@self.__orb2geo(Edir.T, theta_t) #Edir(0) -> Edir(t)
            Sdir_t = self.R_eps.T@self.__orb2geo(Sdir.T, theta_t) #Sdir(0) -> Sdir(t)
            generated_lc[t] = AstEnv.__ReLU(N_arr@Edir_t).T@AstEnv.__ReLU(N_arr@Sdir_t)
        generated_lc = flux0 * generated_lc

        return generated_lc
    
    def __test__all(self):
        reward_total = self.reward(include_other=True)
        if reward_total > self.total_threshold:
            return True
        else:
            return False

    @staticmethod
    def __ReLU(x):
        return x * (x > 0)

    def __lc_mean(self, input_lc):
        """
        input_lc = [LC Length]
        """
        lc_len = input_lc.shape[-1]
        lc_mean0 = (np.sum(input_lc, axis=-1) - (input_lc[..., 0] + input_lc[..., -1])/2) / lc_len
        return lc_mean0

    def show(self):
        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        lim_set = (-10, 10)

        gridX = self.ast.pos_cart_arr[:, :, 0]
        gridY = self.ast.pos_cart_arr[:, :, 1]
        gridZ = self.ast.pos_cart_arr[:, :, 2]

        color = ['coral', 'skyblue', 'gold']
        for i in range(self.lc_num):
            ax1.plot(self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)], color=color[i], linestyle='solid')
            ax1.plot(self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)], color=color[i], linestyle='dashed')
        ax1.set_title("Lightcurve")

        ax2.set_box_aspect((1, 1, 1))
        ax2.set_xlim(lim_set)
        ax2.set_xlabel('X')
        ax2.set_ylim(lim_set)
        ax2.set_ylabel('Y')
        ax2.set_zlim(lim_set)
        ax2.set_zlabel('Z')
        ax2.set_title("Predicted Model")

        ax2.plot_surface(gridX, gridY, gridZ)
        
        plt.show()

'''
np.random.seed(1)

l_max = 8
merge_num = 3
N_set = (40, 20)
lightcurve_unit_len = 200

data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_total.npz"

dataPP = DataPreProcessing(data_path=data_path)
dataPP.Y_total = dataPP.Y_total[:, 0:(l_max+1)**2]
dataPP.coef2R(dataPP.Y_total, l_max=l_max, N_set=N_set)
dataPP.merge(merge_num=merge_num, ast_repeat_num=10, lc_len=lightcurve_unit_len, dupl_ratio=0.02)
dataPP.X_total = dataPP.X_total.numpy()
dataPP.Y_total = dataPP.Y_total.numpy()


test = AstEnv(dataPP.X_total[0, :-9*merge_num], dataPP.X_total[0, -9*merge_num:], merge_num, N_set, lightcurve_unit_len)
test.show()
print(test.obs())
test.step((5, 8, 0, 0), mode='coord_assign')
#print(test.lc_pred)
print(test.obs())
test.show()
#print(test.lc_pred)
#'''
