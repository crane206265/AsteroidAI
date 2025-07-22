import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt

from Asteroid_Model import AsteroidModel
from Ellipsoid_Approx import EllipsoidInversion

PI = 3.1415926535


class AstEnv():
    def __init__(self, target_lc, lc_info, lc_num, reward_domain, N_set=(40, 20), lc_unit_len=200, ell_init=(False, False)):
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
        self.initial_eps = np.empty(2)
        self.initial_eps[0] = np.arctan2(self.rot_axis[1], self.rot_axis[0])
        self.initial_eps[1] = np.arccos(self.rot_axis[2]/LA.norm(self.rot_axis))
        self.R_eps = AsteroidModel.rotArr(-self.initial_eps[1], "y")@AsteroidModel.rotArr(-self.initial_eps[0], "z")

        self.lc_pred = np.ones(self.lc_unit_len*self.lc_num)

        #set obs lc as highest main freq lc
        lc_stacks = self.target_lc.copy().reshape(-1, self.lc_unit_len)
        lc_fft = np.abs(np.fft.fft(lc_stacks))[:, 1:50]
        main_freq_idx = np.argmax(lc_fft, axis=-1)
        self.obs_lc_num = np.argmax(main_freq_idx)
        #self.obs_lc_num = 0
        
        self.__set_obs_params()
        """
        self.obs_lc_num = np.random.randint(lc_num)
        self.obs_time = np.argmax(target_lc[lc_unit_len*(self.obs_lc_num):lc_unit_len*(self.obs_lc_num+1)])
        self.obs_lc_full = self.target_lc[self.lc_unit_len*(self.obs_lc_num):self.lc_unit_len*(self.obs_lc_num+1)]
        self.obs_lc_info = lc_info[9*(self.obs_lc_num):9*(self.obs_lc_num+1)]
        self.obs_phi, self.obs_theta = self.obs_dir_cal()
        """

        self.ast_obs_unit_step = 2 #2
        self.lc_obs_unit_step = 2 #2

        self.Nphi, self.Ntheta = N_set[0], N_set[1]
        self.dphi, self.dtheta = 2*PI/self.Nphi, PI/self.Ntheta
        
        self.reward_threshold = reward_domain[1] #70
        self.total_threshold = reward_domain[1] #70
        self.err_min = reward_domain[0]
        self.ell_err = False

        # Initialize asteroid
        self.reward0 = 999
        self.max_reward = -9e+8
        self.lc_pred = np.ones(self.lc_unit_len*self.lc_num)
        self.ast_backup = None
        self.reset(True, ell_init)
        self.ast_backup = self.ast.copy()
        """
        self.__ellipsoid()
        self.ast = AsteroidModel(axes=self.R_set, N_set=N_set, tilt_mode="assigned", tilt=self.tilt, interior_cal=False)
        self.ast.base_fitting_generator(mode="ellipsoid")


        self.lc_pred = np.zeros(self.lc_unit_len*self.lc_num)
        self.step((0, 100, 0, 0)) #initialize lc_pred
        """
        

    @staticmethod
    def __ReLU(x):
        return x * (x > 0)

    def __ellipsoid(self):
        """
        Generate initial ellipsoid s.t. most fit to target_lc
        """
        lr = 5e-3
        max_epoch = 20 #30
        EllInv = EllipsoidInversion(1, lr, max_epoch, (40, 20))
        param_res, min_param, min_reward, self.min_pred, self.min_r_arr = EllInv.opt(self.obs_lc_full, self.obs_lc_info)
        self.R_set = (5, 3*min_param[0]+5, 3*min_param[1]+5)
        self.tilt = ((min_param[2]+1)*np.pi/2, ((min_param[3]+1)*np.pi/2))
        
        print("min_param", self.R_set, self.tilt)
        '''
        if min_reward > 100:
            self.ell_err = True
        #'''
        
        #self.R_set = np.random.random(3)+5
        #self.tilt = np.random.random(2) * PI * np.array([2, 1])
        #pass

        #self.R_set = tuple(self.R_set)
        #self.tilt = tuple(self.tilt)

    def obs_dir_cal(self):
        """
        calculate obs_phi, obs_theta : (direction to earth in geocentric coord.)
        """
        Edir = self.orb2geo((self.obs_lc_info[3:6]).T, 2*PI*self.obs_time/self.lc_unit_len)
        Edir_phi = np.arctan2(Edir[1], Edir[0])
        Edir_theta = np.arccos(Edir[2]/LA.norm(Edir))
        return Edir_phi, Edir_theta

    def orb2geo(self, vec_orb, rot_angle):
        return AsteroidModel.rotArr(-rot_angle, "z")@self.R_eps@vec_orb
    

    def __r_arr_obs(self):
        obs_r_arr_temp = self.ast.pos_sph_arr[:-1, :-1, 0] + 0
        obs_vec = np.array([np.sin(self.obs_theta)*np.cos(self.obs_phi), np.sin(self.obs_theta)*np.sin(self.obs_phi), np.cos(self.obs_theta)])
        """
        for i in range(self.Nphi):
            for j in range(self.Ntheta):
                phi_ij = (j%2)*(self.dphi/2) + i*self.dphi
                theta_ij = j*self.dtheta
                r_vec = np.array([np.sin(theta_ij)*np.cos(phi_ij), np.sin(theta_ij)*np.sin(phi_ij), np.cos(theta_ij)])
                if np.dot(obs_vec, r_vec) <= 0:
                    obs_r_arr_temp[i, j] = 0
        """
        obs_r_arr = obs_r_arr_temp[::self.ast_obs_unit_step, ::self.ast_obs_unit_step] + 0
        obs_r_arr = obs_r_arr.flatten()

        return obs_r_arr
    
    def _obs(self):
        #r_arr obs
        obs_r_arr = self.__r_arr_obs()

        #lc obs
        obs_lc = np.zeros(2*(self.lc_unit_len//(4*self.lc_obs_unit_step)))
        k0 = 0
        for k1 in range(self.obs_time-self.lc_unit_len//4, self.obs_time+self.lc_unit_len//4, self.lc_obs_unit_step):
            if k0 >= obs_lc.shape[0]: break
            #obs delta
            obs_lc[k0] = self.obs_lc_full[k1%self.lc_unit_len] - self.lc_pred[self.obs_lc_num*self.lc_unit_len + k1%self.lc_unit_len]
            k0 = k0+1
            
        
        obs_tensor = np.concatenate((obs_r_arr, obs_lc))
        return obs_tensor
    
    def obs(self):
        #r_arr obs
        #r_arr obs
        obs_r_arr = self.__r_arr_obs()

        obs_lc = self.obs_lc_full[::self.lc_obs_unit_step] - self.lc_pred[self.obs_lc_num*self.lc_unit_len:(self.obs_lc_num+1)*self.lc_unit_len:self.lc_obs_unit_step]
        obs_lc_normalized = obs_lc*10/np.max(np.abs(obs_lc))
        obs_tensor = np.concatenate((obs_r_arr, obs_lc_normalized))
        obs_tensor = np.concatenate((obs_tensor, np.repeat(10*self.obs_lc_info, 5)))
        return obs_tensor
    
    def step(self, action, mode='ratio_assign', update=True):
        """
        action = [R_cut, r, phi, theta]
        if mode == 'ratio_assign'
            action = [phi, theta, r_cen_ratio, R_cut_ratio]
        """
        done = False
        all_done = False
        passed = False
        if not (action[0] == 0 and action[1] == 0):
            if mode == 'ratio_assign':
                phi_action = (self.obs_phi/(2*PI) + action[0])%1 # action be a relative ratio from obs_phi
                theta_action = (self.obs_theta/PI + action[1])%1 # action be a relative ratio from obs_theta
                self.ast.cut_ast(1, 0, True, mode='ratio_assign', pos_sph=(phi_action, theta_action, action[2], action[3]))
            elif mode == 'coord_assign':
                cut_sph_pos = AsteroidModel.sph2cart((action[1], action[2], action[3]))
                self.ast.cut_ast(1, 0, True, mode='Rxyz_assign', pos_sph=(action[0], cut_sph_pos[0], cut_sph_pos[1], cut_sph_pos[2]))
            else:
                raise NotImplementedError


        # Maintaining the radius mean of asteroid r_arr
        mean0 = 10
        r_arr_mean = np.mean(self.ast.pos_sph_arr[:, :, 0])
        self.ast.pos_sph_arr[:, :, 0] = self.ast.pos_sph_arr[:, :, 0] * mean0 / r_arr_mean
        self.ast.pos_cart_arr = self.ast.pos_cart_arr * mean0 / r_arr_mean

        self.ast.surf_vec_cal()

        reward = self.reward(init=100.0, rooted=False, include_other=False, relative=True)
        #print(reward, "REWARD")
        if reward > self.reward_threshold:
            done = True
            passed = True
            all_done = self.__test__all()
        elif reward < self.max_reward - 3.5:#min(-4e+2, self.reward0):
            done = True
            passed = False

        observation = self.obs()
        _ = self.reward(include_other=True)

        #print(str(int(reward*1000)/1000) + "|" + str(int(self.max_reward*1000)/1000), end=' ')
        if reward > self.max_reward and update:
            self.max_reward = reward + 0.0
            self.ast_backup = self.ast.copy()
            
        return observation, reward, done, (passed, all_done)

    def _step(self, action, mode='ratio_assign'):
        """
        action = [R_cut, r, phi, theta]
        if mode == 'ratio_assign'
            action = [phi, theta, r_cen_ratio, R_cut_ratio]
        """
        done = False
        all_done = False
        passed = False
        if mode == 'ratio_assign':
            phi_action = action[0] - action[0]//1 # action be a relative ratio from obs_phi
            theta_action = action[1] - action[1]//1 # action be a relative ratio from obs_theta
            self.ast.cut_ast(1, 0, True, mode='ratio_assign', pos_sph=(phi_action, theta_action, action[2], action[3]))
        elif mode == 'coord_assign':
            cut_sph_pos = AsteroidModel.sph2cart((action[1], action[2], action[3]))
            self.ast.cut_ast(1, 0, True, mode='Rxyz_assign', pos_sph=(action[0], cut_sph_pos[0], cut_sph_pos[1], cut_sph_pos[2]))
        else:
            raise NotImplementedError


        # Maintaining the radius mean of asteroid r_arr
        mean0 = 5
        r_arr_mean = np.mean(self.ast.pos_sph_arr[:, :, 0])
        self.ast.pos_sph_arr[:, :, 0] = self.ast.pos_sph_arr[:, :, 0] * mean0 / r_arr_mean
        self.ast.pos_cart_arr = self.ast.pos_cart_arr * mean0 / r_arr_mean

        self.ast.surf_vec_cal()

        reward = self.reward(init=100.0, rooted=False, relative=True)
        _ = self.reward(include_other=True)
        #print(reward, "REWARD")
        if reward > self.reward_threshold:
            done = True
            passed = True
            all_done = self.__test__all()
        elif reward <= min(-15e+2, self.reward0):
            done = True
            passed = False

        observation = self.obs()

        if reward > self.max_reward:
            self.max_reward = reward
            self.ast_backup = self.ast.copy()

        return observation, reward, done, (passed, all_done)
    
    def reward(self, init=100, rooted=False, include_other=False, relative=True):
        reward = init
        for i in range(self.lc_num):
            if not include_other and i != self.obs_lc_num:
                continue
            target_lc_temp = self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)]
            target_lc_mean = self.__lc_mean(target_lc_temp)

            lc_temp = self.__lc_gen(self.lc_info[9*i:9*(i+1)]) #generate lc
            lc_temp = lc_temp * target_lc_mean / self.__lc_mean(lc_temp) #scaling lc_temp compared with target_lc_temp
            self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)] = lc_temp

            #self.__scale_lc(flux0=100)
            
            '''
            if not include_other:
                if self.obs_time-self.lc_unit_len//4 < 0 or self.obs_time+self.lc_unit_len//4 >= self.lc_unit_len:
                    target_lc_temp = np.concatenate((target_lc_temp[(self.obs_time-self.lc_unit_len//4)%self.lc_unit_len:], target_lc_temp[0:(self.obs_time+self.lc_unit_len//4)%self.lc_unit_len]))
                    lc_temp = np.concatenate((lc_temp[(self.obs_time-self.lc_unit_len//4)%self.lc_unit_len:], lc_temp[0:(self.obs_time+self.lc_unit_len//4)%self.lc_unit_len]))
                else:
                    target_lc_temp = target_lc_temp[self.obs_time-self.lc_unit_len//4:self.obs_time+self.lc_unit_len//4]
                    lc_temp = lc_temp[self.obs_time-self.lc_unit_len//4:self.obs_time+self.lc_unit_len//4]
            #'''

            target_lc_temp = target_lc_temp - target_lc_mean
            lc_temp = lc_temp - target_lc_mean

            #loss = 2*np.sqrt(*np.trapz(np.abs(target_lc_temp-lc_temp))/(2*self.__amp_lc(target_lc_temp)))

            
            if relative:
                amp = self.__amp_lc(target_lc_temp)
                loss = np.mean((80*(target_lc_temp - lc_temp)/amp)**2) #40

                loss_i = 60*np.trapz(np.abs(target_lc_temp-lc_temp))/(100*amp)
                loss_d = np.mean((40*(np.diff(target_lc_temp)-np.diff(lc_temp)))**2)
                #loss = (loss + loss_i + loss_d)*3/10
                loss = (1.2*loss + loss_i + loss_d)*2/10
            else:
                loss = np.mean((target_lc_temp - lc_temp)**2)

            if rooted:
                loss = np.sqrt(loss+1e-15)# + np.sqrt(loss_1+1e-15)
            
            '''
            target_lc_pc = np.stack((target_lc_temp, np.arange(target_lc_temp.shape[0]), np.zeros(target_lc_temp.shape[0]))).T
            lc_temp_pc = np.stack((lc_temp, np.arange(target_lc_temp.shape[0]), np.zeros(target_lc_temp.shape[0]))).T
            cd = pcu.chamfer_distance(target_lc_pc, lc_temp_pc)
            if relative:
                loss = cd/(self.__amp_lc(target_lc_temp)**2)
            if rooted:
                loss = np.sqrt(loss+1e-15)
            loss = 10*loss
            '''
                
            reward = reward - loss
        
        return reward
    
    def reset(self, passed, ell_init=(False, False)):
        if self.ast_backup == None:
            max_try = 20 # original : 5, this is changed value for Ellipsoid_Approx_Data 
            for i in range(max_try+1):
                if ell_init[0]:
                    ell_arr = ell_init[1]
                    self.R_set = ell_arr[:3]
                    self.tilt = ell_arr[3:]
                else:
                    self.__ellipsoid()
                self.ast = AsteroidModel(axes=self.R_set, N_set=(self.Nphi, self.Ntheta), tilt_mode="assigned", tilt=self.tilt, interior_cal=False)
                self.ast.base_fitting_generator(mode="ellipsoid")
                self.lc_pred = np.ones(self.lc_unit_len*self.lc_num)
                _, self.reward0, _, _ = self.step((0, 0, 0, 0)) #initialize/recalculate lc_pred
                #print(self.R_eps)
                print("[AstEnv] self.reward0 =", self.reward0)

                self.min_r_arr = np.sqrt(self.min_r_arr[:, :, 0]**2 + self.min_r_arr[:, :, 1]**2 + self.min_r_arr[:, :, 2]**2)
                print("MSE r_arr :", np.sqrt(np.mean((self.min_r_arr * 1 / np.mean(self.min_r_arr) - self.ast.pos_sph_arr[:, :, 0] * 1 / np.mean(self.ast.pos_sph_arr[:, :, 0]))**2)))
                

                plt.plot(self.min_pred * self.__lc_mean(self.lc_pred) / self.__lc_mean(self.min_pred), linestyle='solid', label="min_pred")
                plt.plot(self.target_lc, linestyle='solid', label='target_lc')
                plt.legend()
                plt.show()

                #print(self.reward0)
                if self.reward0 > self.err_min and self.reward0 < self.total_threshold:#self.err_min+30:
                    break

                if i == max_try:
                    self.ell_err = True
        else:
            if not passed:
                self.ast = self.ast_backup.copy()
            self.__set_obs_params()
            self.step((0, 0, 0, 0)) #initialize/recalculate lc_pred

        return self.obs()
    
    def __set_obs_params(self):
        #self.obs_lc_num = 0#1#np.random.randint(self.lc_num)
        self.obs_time = np.argmax(np.abs(self.target_lc[self.lc_unit_len*(self.obs_lc_num):self.lc_unit_len*(self.obs_lc_num+1)] - self.lc_pred[self.lc_unit_len*(self.obs_lc_num):self.lc_unit_len*(self.obs_lc_num+1)]))
        self.obs_lc_full = self.target_lc[self.lc_unit_len*(self.obs_lc_num):self.lc_unit_len*(self.obs_lc_num+1)]
        self.obs_lc_info = self.lc_info[9*(self.obs_lc_num):9*(self.obs_lc_num+1)]
        self.obs_phi, self.obs_theta = self.obs_dir_cal()

    def __lc_gen(self, lc_info, flux0=10):
        Sdir = lc_info[0:3]
        Edir = lc_info[3:6]
        #rot_axis = lc_info[6:9]
        N_arr = self.ast.surf_vec_arr / np.sqrt(np.abs(self.ast.surf_vec_arr)+1e-15)
        N_arr = N_arr.reshape(-1, 3)

        generated_lc = np.zeros(self.lc_unit_len)
        for t in range(self.lc_unit_len):
            theta_t = 2*PI*t/self.lc_unit_len
            Edir_t = self.R_eps.T@self.orb2geo(Edir.T, theta_t) #Edir(0) -> Edir(t)    
            Sdir_t = self.R_eps.T@self.orb2geo(Sdir.T, theta_t) #Sdir(0) -> Sdir(t)    
            Edir_t = Edir_t / LA.norm(Edir_t)
            Sdir_t = Sdir_t / LA.norm(Sdir_t)
            generated_lc[t] = AstEnv.__ReLU(N_arr@Edir_t).T@AstEnv.__ReLU(N_arr@Sdir_t)
        generated_lc = flux0 * generated_lc

        return generated_lc
    
    
    def __test__all(self):
        reward_total = self.reward(include_other=True)
        if reward_total > self.total_threshold:
            return True
        else:
            return False

    def __lc_mean(self, input_lc):
        """
        input_lc = [LC Length]
        """
        lc_len = input_lc.shape[-1]
        lc_mean0 = (np.sum(input_lc, axis=-1) - (input_lc[..., 0] + input_lc[..., -1])/2) / lc_len
        return lc_mean0
    
    def __scale_lc(self, flux0 = 100):
        for i in range(self.lc_num):
            target_lc_temp = self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)]
            target_lc_mean = self.__lc_mean(target_lc_temp)
            self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)] = self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)] * flux0 / target_lc_mean

            pred_lc_temp = self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)]
            pred_lc_mean = self.__lc_mean(pred_lc_temp)
            self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)] = self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)] * flux0 / pred_lc_mean

    def __amp_lc(self, input_lc):
        lc_max = np.max(input_lc)
        lc_min = np.min(input_lc)
        return lc_max - lc_min
    

    def show(self, name="None"):
        fig = plt.figure(figsize=(13, 5))
        ax1 = fig.add_subplot(1, 2, 1)
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        lim_set = (-10, 10)

        gridX = self.ast.pos_cart_arr[:, :, 0]
        gridY = self.ast.pos_cart_arr[:, :, 1]
        gridZ = self.ast.pos_cart_arr[:, :, 2]

        color = ['coral', 'gold', 'skyblue']
        for i in range(self.lc_num):
            if i == self.obs_lc_num:
                ax1.plot(self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)], color=color[i], linestyle='solid') #black
                ax1.plot(self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)], color=color[i], linestyle='dashed')
                ax1.set_ylim([np.min(self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)])-5, np.max(self.target_lc[self.lc_unit_len*i:self.lc_unit_len*(i+1)])+5])
                #ax1.set_ylim([np.min(self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)])-5, np.max(self.lc_pred[self.lc_unit_len*i:self.lc_unit_len*(i+1)])+5])
            else:
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
        
        path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/A2C_train/"
        plt.savefig(path+name)
        #plt.show()

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
