import numpy as np
import numpy.linalg as LA

import matplotlib.pyplot as plt

import random
from tqdm import tqdm

import gc


PI = 3.1415926535
MAX_STEPS = 3000 #per episode

# seed
seed = 722
np.random.seed(seed)
random.seed(seed)



class CutterSphere():
    def __init__(self, ast, random = True, mode = 'Rxyz_assign', *args):
        """
        initialize
        - if random == True, use random parameters
        - if random == False, use parameters from *args
        *args = (R, x1, y1, z1)
            if mode == 'ratio_assign' *args = (phi, theta, r_cen_ratio, r_cut_ratio)
        ast : Asteroid_Model#class
        """
        self.k = 0.1#7e-3 #cut ratio 0.2
        self.min_cen = 7 #3 - for generating asteroid
        self.max_cen = 13 #10

        if random == False and mode == 'Rxyz_assign':
            self.radi = args[0]
            self.x1 = args[1]
            self.y1 = args[2]
            self.z1 = args[3]

            self.r_cen, self.phi_cen, self.theta_cen = CutterSphere.__cart2sph((self.x1, self.y1, self.z1))
            return
        
        elif random == False and mode == 'ratio_assign':
            self.phi_cen = 2*np.pi*args[0]
            self.theta_cen = np.pi*args[1]
            self.r_cen_ratio = args[2]
            self.R_cut_ratio = args[3]
        
        else:
            self.phi_cen = 2*np.pi*np.random.rand()
            self.theta_cen = np.pi*np.random.rand()

        self.j_cen = round(self.theta_cen/ast.dtheta)
        if self.j_cen%2 == 0:
            self.i_cen = round(self.phi_cen/ast.dphi)
        else:
            self.i_cen = round((self.phi_cen-ast.dphi/2)/ast.dphi)
        self.r_ast = ast.pos_sph_arr[self.i_cen, self.j_cen, 0]

        if random == False and mode == 'ratio_assign':
            self.r_cen = (self.min_cen + (self.max_cen-self.min_cen)*self.r_cen_ratio)*self.r_ast
            self.radi = self.k*self.r_ast*self.R_cut_ratio + self.r_cen - self.r_ast
            
        else:
            self.r_cen = (self.min_cen + (self.max_cen-self.min_cen)*np.random.rand())*self.r_ast
            self.radi = self.k*self.r_ast*np.random.rand() + self.r_cen - self.r_ast

        cart = ast.sph2cart([self.r_cen, self.phi_cen, self.theta_cen])
        self.x1 = cart[0]
        self.y1 = cart[1]
        self.z1 = cart[2]

    def f(self, cart_pos):
        """
        Equation of Sphere
        cart_pos : cartesian position coord.
        """
        x = cart_pos[0]
        y = cart_pos[1]
        z = cart_pos[2]

        f = (x-self.x1)**2 + (y-self.y1)**2 + (z-self.z1)**2 - self.radi**2
        return f
    
    def r_f(self, angle_pos):
        """
        <input> angle_pos = given [phi, theta]
        <output> : r coord. corr the input (the point on surface of the sphere)
        """
        phi = angle_pos[0]
        theta = angle_pos[1]
        
        r_f_unit = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
        r_cen_unit = np.array([np.sin(self.theta_cen)*np.cos(self.phi_cen), np.sin(self.theta_cen)*np.sin(self.phi_cen), np.cos(self.theta_cen)])
        cosa = np.dot(r_f_unit, r_cen_unit)

        r_f = self.r_cen*cosa - ((self.r_cen*cosa)**2 - (self.r_cen**2 - self.radi**2))**0.5
        return r_f
    
    @staticmethod
    def __cart2sph(cart_coord):
        """
        cartesian coord -> spherical coord
        input : cart_coord = (x, y, z)
        output : (r, phi, theta)
        """
        x = cart_coord[0]
        y = cart_coord[1]
        z = cart_coord[2]
        r = LA.norm(np.array([x, y, z]))
        phi = np.arctan2(y, x)
        theta = np.arccos(z/r)
        
        return r, phi, theta
    
    
class AsteroidModel():
    def __init__(self, axes, N_set, tilt_mode="assigned", tilt=(0, 0), coord2discrete_param=(0.5, 10), interior_cal=True):
        self.Nphi = N_set[0]
        self.Ntheta = N_set[1]
        self.dphi = 2*np.pi/self.Nphi
        self.dtheta = np.pi/self.Ntheta
        self.coord_set = (self.dphi, self.dtheta, self.Nphi, self.Ntheta)
        self.pos_sph_arr = np.zeros((self.Nphi+1, self.Ntheta+1, 3)) #last index = first index (circular)
        self.pos_cart_arr = np.zeros((self.Nphi+1, self.Ntheta+1, 3)) #last index = first index 
        self.surf_vec_arr = np.zeros((self.Nphi, self.Ntheta, 2, 3))
        self.albedo_arr = np.ones((self.Nphi, self.Ntheta, 2))

        self.axes_R = np.array([axes[0], axes[1], axes[2]])
        self.tilt = np.array([tilt[0], tilt[1]])

        if tilt_mode == "random":
            self.tilt = np.pi*np.array([2*np.random.rand(1)[0], np.random.rand(1)[0]])
        elif tilt_mode == "assigned":
            pass
        else:
            raise ValueError("Unimplemented tilt_mode")
        
        self.discr_param = coord2discrete_param
        self.interior_cal = interior_cal


        prec, bound = self.discr_param
        N_index = 2*int(bound/prec) + 1
        self.is_interior = np.zeros((N_index, N_index, N_index))

        self.inertia_tensor = np.eye(3)

        self.max_r = 0
        self.valid_lw_bd = 0
        self.valid_up_bd = N_index

        self.COM_vec = np.zeros((3))

    @staticmethod
    def rotArr(angle, axis):
        # for rotational matrix
        if axis == "x" or axis == 0:
            arr = np.array([[1, 0            , 0             ],
                            [0, np.cos(angle), -np.sin(angle)],
                            [0, np.sin(angle), np.cos(angle) ]])
        elif axis == "y" or axis == 1:
            arr = np.array([[np.cos(angle), 0, np.sin(angle)],
                            [0            , 1, 0             ],
                            [-np.sin(angle), 0, np.cos(angle) ]])
        elif axis == "z" or axis == 2:
            arr = np.array([[np.cos(angle), -np.sin(angle), 0],
                            [np.sin(angle), np.cos(angle) , 0],
                            [0            , 0             , 1]])
        else:
            raise ValueError("Unappropriate axis")
        return arr

    # necessary calclutating functions
    @staticmethod
    def sph2cart(sph_coord):
        """
        spherical coord -> cartesian coord
        input : sph_coord = (r, phi, theta)
        output : (x, y, z)
        """
        r = sph_coord[0]
        phi = sph_coord[1]
        theta = sph_coord[2]
        x = r*np.sin(theta)*np.cos(phi)
        y = r*np.sin(theta)*np.sin(phi)
        z = r*np.cos(theta)
        return np.array([x, y, z])
    
    @staticmethod
    def cart2sph(cart_coord):
        """
        cartesian coord -> spherical coord
        input : cart_coord = (x, y, z)
        output : (r, phi, theta)
        """
        x = cart_coord[0]
        y = cart_coord[1]
        z = cart_coord[2]
        r = LA.norm(np.array([x, y, z]))
        phi = np.arctan2(y, x)
        theta = np.arccos(z/r)

        return np.array([r, phi, theta])

    def __circular(self, index):
        """
        for circular pos_arr
        index = 'i' : i-axis
              = 'j' : j-axis
              = 'all' : i-axis & j-axis
        """
        if index == 'i' or 'all':
            for j in range(self.Ntheta+1):
                self.pos_sph_arr[self.Nphi, j] = self.pos_sph_arr[0, j]
                self.pos_cart_arr[self.Nphi, j] = self.pos_cart_arr[0, j]
        
        if index == 'j' or 'all':
            for i in range(self.Nphi+1):
                self.pos_sph_arr[i, 0] = self.pos_sph_arr[0, 0]
                self.pos_cart_arr[i, 0] = self.pos_cart_arr[0, 0]

                self.pos_sph_arr[i, self.Ntheta] = self.pos_sph_arr[0, self.Ntheta]
                self.pos_cart_arr[i, self.Ntheta] = self.pos_cart_arr[0, self.Ntheta]



    # generating with basic frame
    def base_fitting_generator(self, mode="ellipsoid"):
        if mode == "ellipsoid":
            generating_frame = self.__ellipsoid_frame 

        for i in range(self.Nphi):
            for j in range(self.Ntheta+1):
                phi_ij = (j%2)*(self.dphi/2) + i*self.dphi
                theta_ij = j*self.dtheta
                r_ij = generating_frame([phi_ij, theta_ij])

                self.max_r = r_ij if r_ij>self.max_r else self.max_r

                x_ij = r_ij*np.sin(theta_ij)*np.cos(phi_ij)
                y_ij = r_ij*np.sin(theta_ij)*np.sin(phi_ij)
                z_ij = r_ij*np.cos(theta_ij)
                
                self.pos_sph_arr[i, j] = np.array([r_ij, phi_ij, theta_ij])
                self.pos_cart_arr[i, j] = np.array([x_ij, y_ij, z_ij])

                if i == 0:
                    self.pos_sph_arr[self.Nphi, j] = np.array([r_ij, phi_ij, theta_ij])
                    self.pos_cart_arr[self.Nphi, j] = np.array([x_ij, y_ij, z_ij])
        

    def __ellipsoid_frame(self, direction, radi=[-1, -1, -1], tilt_angle=[-1, -1]):
        """
        ellipsoid generator
        a, b, c : radius corr. axis (default = axes_R)
        """
        if radi[0] == -1:
            a = self.axes_R[0]
        if radi[1] == -1:
            b = self.axes_R[1]
        if radi[2] == -1:
            c = self.axes_R[2]

        """
        tilt_angle = [longitude, latitude]
        * longitude angle : z-axis rotation
        * latitude angle : x-axis rotation
        """
        if tilt_angle[0] == -1:
            long = self.tilt[0]
        else:
            long = tilt_angle[0]
        if tilt_angle[1] == -1:
            lat = self.tilt[1]
        else:
            lat = tilt_angle[1]

        self.tilt = np.array([long, lat])
        long_rot_arr = self.rotArr(-long, "z")
        lat_rot_arr = self.rotArr(-lat, "y")
        R_arr = lat_rot_arr@long_rot_arr

        A_arr = np.array([[1/a**2, 0, 0],
                          [0, 1/b**2, 0],
                          [0, 0, 1/c**2]])
        
        """
        coordinate direction : [phi, theta]
        output : corr. r value
        """
        phi_temp = direction[0]
        theta_temp = direction[1]
        u_vec = np.array([np.sin(theta_temp)*np.cos(phi_temp), np.sin(theta_temp)*np.sin(phi_temp), np.cos(theta_temp)]).T
        r_temp = 1 / np.sqrt(u_vec.T@R_arr.T@A_arr@R_arr@u_vec)

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
                self.surf_vec_arr[i, j, 0] = 0.5*np.cross(v11, v12)
                self.surf_vec_arr[i, j, 1] = 0.5*np.cross(v21, v22)

    def wobble_r(self, epoch = 1, ratio = 0.005):
        """
        wobble r coord.
        
        epoch : repeat num
        ratio : wobbling ratio
        """
        if epoch == 0:
            self.surf_vec_cal()
            return

        for i in range(self.Nphi):
            for j in range(self.Ntheta+1):
                self.pos_sph_arr[i, j, 0] += self.pos_sph_arr[i, j, 0]*np.random.randn()*ratio
                self.pos_cart_arr[i, j] = AsteroidModel.sph2cart(self.pos_sph_arr[i, j])
        self.__circular('all')
        self.wobble_r(epoch-1, ratio)
    
    def cut_ast(self, sph_num, pla_num, assigned=False, mode='Rxyz_assign', **kwargs):
        """
        cut asteroid with specific shape

        sph_num : cutting spherical num
        pla_num : cutting plane num
        """
        pos_sph = kwargs['pos_sph']
        self.__sph_cut(sph_num, assigned=not(assigned), mode=mode, pos_sph=pos_sph)

    def __sph_cut(self, sph_num, **kwargs):
        """
        cutting with sphere - CutterSphere#class
        """
        for k in range(sph_num):
            sph_temp = CutterSphere(self, kwargs['assigned'], kwargs['mode'], kwargs['pos_sph'][0], kwargs['pos_sph'][1], kwargs['pos_sph'][2], kwargs['pos_sph'][3])
            for i in range(self.Nphi+1):
                for j in range(self.Ntheta+1):
                    if sph_temp.f(self.pos_cart_arr[i, j]) < 0:
                        self.pos_sph_arr[i, j, 0] = sph_temp.r_f(self.pos_sph_arr[i, j, 1:])
                        self.pos_cart_arr[i, j] = AsteroidModel.sph2cart(self.pos_sph_arr[i, j])

    def copy(self):
        ast_copy = AsteroidModel((1, 1, 1), (self.Nphi, self.Ntheta), interior_cal=self.interior_cal, coord2discrete_param=self.discr_param)
        ast_copy.pos_sph_arr = self.pos_sph_arr.copy()
        ast_copy.pos_cart_arr = self.pos_cart_arr.copy()
        ast_copy.surf_vec_arr = self.surf_vec_arr.copy()
        ast_copy.axes_R = self.axes_R.copy()
        ast_copy.tilt = self.tilt.copy()
        ast_copy.max_r = self.max_r + 0
    
        return ast_copy

    

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

        self.ast_obs_unit_step = 1 #2
        self.lc_obs_unit_step = 1 #2

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
        #obs_tensor = np.concatenate((obs_tensor, np.repeat(10*self.obs_lc_info, 5)))
        obs_tensor = np.concatenate((obs_tensor, self.obs_lc_info[:6]))
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

                loss_i = 60*np.trapezoid(np.abs(target_lc_temp-lc_temp))/(100*amp)
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
                    raise NotImplementedError
                self.ast = AsteroidModel(axes=self.R_set, N_set=(self.Nphi, self.Ntheta), tilt_mode="assigned", tilt=self.tilt, interior_cal=False)
                self.ast.base_fitting_generator(mode="ellipsoid")
                self.lc_pred = np.ones(self.lc_unit_len*self.lc_num)
                _, self.reward0, _, _ = self.step((0, 0, 0, 0)) #initialize/recalculate lc_pred
                #print(self.R_eps)
                #print("[AstEnv] self.reward0 =", self.reward0)

                #self.min_r_arr = np.sqrt(self.min_r_arr[:, :, 0]**2 + self.min_r_arr[:, :, 1]**2 + self.min_r_arr[:, :, 2]**2)
                #print("MSE r_arr :", np.sqrt(np.mean((self.min_r_arr * 1 / np.mean(self.min_r_arr) - self.ast.pos_sph_arr[:, :, 0] * 1 / np.mean(self.ast.pos_sph_arr[:, :, 0]))**2)))

                #plt.plot(self.min_pred * self.__lc_mean(self.lc_pred) / self.__lc_mean(self.min_pred), linestyle='solid', label="min_pred")
                #plt.plot(self.target_lc, linestyle='solid', label='target_lc')
                #plt.legend()
                #plt.show()

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
            Edir_t = self.R_eps.T@self.orb2geo(Edir.T, theta_t) #Edir(0) -> Edir(t)    ############# 여기 R_eps 검토하기!!!!!!!!!!!1
            Sdir_t = self.R_eps.T@self.orb2geo(Sdir.T, theta_t) #Sdir(0) -> Sdir(t)    #############
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


class Runner():
    def __init__(self, env:AstEnv, state_dim, action_dim, prec = 5.0):
        self.env = env
        self.done = True
        self.passed = False
        self.all_done = False
        self.reward_past = None
        self.prec = prec
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.data_set_arr = np.zeros((1, self.state_dim+self.action_dim+1))

    def reset(self, passed):
        self.state = self.env.reset(passed)
        self.done = False
        self.passed = False
        self.reward_past = None

    def reward_fn(self, reward, time, t_m=MAX_STEPS, t_g=int(MAX_STEPS*0.6), R_max=50, alpha=1, beta=1, epsilon=0.05):
        """
        <PARAMETERS>
        t_m : MAX_STEPS per episode(env)
        t_g : goal step (want to finish episode in)
        R_max : max reward w/o time penalty (before scaling)
        alpha : [start reward] = alpha * R_max
        beta : [time penalty until t_max] = beta * R_max
        epsilon : [time penalty until t_goal] = epsilon * R_max

        time penalty function : quadratic
        """
        modified_reward = 0#alpha * R_max
        penalty_coef = R_max * LA.inv(np.array([[t_m*(t_m+1)*(2*t_m+1), 6*t_m],
                                                [t_g*(t_g+1)*(2*t_g+1), 6*t_g]])) @ np.array([6*beta, 6*epsilon]).T
        modified_reward += -np.array([time**2, 1])@penalty_coef
        modified_reward += max(reward - self.env_max_reward_past, 0)
        modified_reward /= 1
        modified_reward *= 40/(self.env.total_threshold - self.env.reward0)
        return modified_reward

    def run(self, env_no, random=True, save=False):
        prec = self.prec
        r_cen_low = 0.2
        r_cen_high = 1
        R_cut_low = 0.40 #0.01 #0.25
        R_cut_high = 1 #0.15

        get_env_num = int(30 * (self.env.reward_threshold - max(self.env.reward0, 0)) / self.env.reward_threshold)#int(30 * (self.env.reward_threshold - self.env.reward0) / self.env.reward_threshold)
        get = 0
        reward_threshold_list = np.linspace(max(self.env.reward0, 0), self.env.reward_threshold, get_env_num)
        if reward_threshold_list.shape[0] == 0:
            return
        
        max_reward = -9e+8
        show_bool = True
        with tqdm(total=get_env_num, desc="Reward Map Generation") as pbar:
            for t in range(MAX_STEPS):
                if self.all_done:
                    print(" SUCCESSED in CURRENT ENV")
                    self.env.show()
                    break

                if self.done:
                    self.reset(self.passed)

                actions = np.random.uniform(-prec, prec, 4)

                actions[:-2] = actions[:-2] - actions[:-2]//1
                actions[-2:] = np.clip(actions[-2:], [-prec, -prec], [prec, prec])
                actions[2] = 0.1
                actions[3] = 0.1

                self.env_max_reward_past = self.env.max_reward + 0
                next_state, reward, self.done, info = self.env.step(actions)
                if self.reward_past == None:
                    self.reward_past = reward
                self.passed = info[0]
                self.all_done = info[1]
                weight = 2.5

                #modified_reward = self.reward_fn(reward, t, t_m=500, t_g=300, R_max=50.0, alpha=1.0, beta=0.3, epsilon=0.03)
                if self.done and not self.passed: # if max reward - 3 condition is not satisfied 
                    weight = 1.0

                self.state = next_state

                if weight != 1:
                    self.reward_past = reward + 0


                if reward >= reward_threshold_list[get]:
                    self.make_map(reward, self.state, env_no, random, save)
                    get += 1
                    pbar.update(1)

                if get == get_env_num: break

                max_reward = max(max_reward, reward)
                if t%4 == 0 and get == 0:
                    print(" | Reward : {:7.5g}".format(reward), end='')
                    print(" | actions : [{:6.05g}, {:6.05g}, {:6.05g}, {:6.05g}]".format(actions[0], actions[1], actions[2], actions[3]), end='')
                    print(" | done/pass : "+str(self.done)+"/"+str(self.passed)+" "+str(self.env.max_reward))
                    if t%20 == 0:
                        show_bool = True

                if show_bool:
                    #print("show_passed : "+str(reward)+" | obs_lc_num : "+str(self.env.obs_lc_num))
                    #self.env.show(str(data_num)+"_"+str(et)+"_"+str(k)+"_"+str(int(reward*100)/100)+"_"+"0402ast.png")
                    #plt.close()
                    show_bool = False

                if self.done:
                    if self.passed:
                        #print("show_passed : "+str(reward)+" | obs_lc_num : "+str(self.env.obs_lc_num))
                        break

    def make_map(self, ref_reward, state, env_no, random, save):
        ratio_action_set = [(0.1, 0.1)]#, (0.3, 0.3), (0.5, 0.5), (1, 1)]
        ref_ast = self.env.ast.copy()
        resol = 1

        rot_axis = self.env.initial_eps * 180/np.pi
        rot_axis[0] = rot_axis[0]%360
        rot_axis[1] = rot_axis[1]%180

        path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/reward_maps/"

        for ratio_actions in ratio_action_set:
            delta_map_temp = np.zeros((resol*self.env.Nphi, resol*self.env.Ntheta))
            print("\nGenerating Map... (at Reward="+str(int(ref_reward*100)/100)+")", end='')
            for idx in range(self.env.Nphi*self.env.Ntheta*resol*resol):
                i = idx//int(resol*self.env.Ntheta)
                j = idx%int(resol*self.env.Ntheta)

                if random:
                    phi_action = (i/(resol*self.env.Nphi) + np.random.normal(0, 0.05, 1)[0])%1
                    theta_action = (j/(resol*self.env.Ntheta) + np.random.normal(0, 0.05, 1)[0])%1
                else:
                    phi_action = (i/(resol*self.env.Nphi))%1
                    theta_action = (j/(resol*self.env.Ntheta))%1
                actions = np.array([phi_action, theta_action, ratio_actions[0], ratio_actions[1]])
                
                obs, reward, _, _ = self.env.step(actions, update=False)
                obs
                delta_map_temp[i, j] = reward - ref_reward
                self.data_set_arr = np.concatenate((self.data_set_arr, np.array([np.concatenate((self.state, actions, np.array([delta_map_temp[i, j]])))])), axis=0)
                
                self.env.ast = ref_ast.copy()

            
            if save:
                phi_ticks = self.__map_tick_list(resol, self.env.Nphi, 360)
                theta_ticks = self.__map_tick_list(resol, self.env.Ntheta, 180)
        
                circle_points = 200
                Edirs = np.zeros((2, circle_points))
                Sdirs = np.zeros((2, circle_points))
                for t in range(circle_points):
                    Edir = self.env.R_eps.T@self.env.orb2geo((self.env.obs_lc_info[3:6]).T, 2*np.pi*t/circle_points)   ##########AstEnv에 N_arr랑 하는거 수정시 이것도 수정 필요
                    Edirs[0, t] = (np.arctan2(Edir[1], Edir[0]) * 180/np.pi)%360
                    Edirs[1, t] = (np.arccos(Edir[2]/LA.norm(Edir)) * 180/np.pi)%180
                    Sdir = self.env.R_eps.T@self.env.orb2geo((self.env.obs_lc_info[0:3]).T, 2*np.pi*t/circle_points)   ##########
                    Sdirs[0, t] = (np.arctan2(Sdir[1], Sdir[0]) * 180/np.pi)%360
                    Sdirs[1, t] = (np.arccos(Sdir[2]/LA.norm(Sdir)) * 180/np.pi)%180

                delta_map_temp = delta_map_temp.T
                reward_map_temp = delta_map_temp + ref_reward

                grady, gradx = np.gradient(delta_map_temp)
                x = np.arange(delta_map_temp.shape[1])
                y = np.arange(delta_map_temp.shape[0])
                X, Y = np.meshgrid(x, y)


                plt.figure(figsize=(20, 11))
                plt.imshow(delta_map_temp)

                plt.plot(rot_axis[0]*resol*self.env.Nphi/360, rot_axis[1]*resol*self.env.Ntheta/180, color='red', marker='X', markersize=10)
                plt.plot(((rot_axis[0]+180)%360)*resol*self.env.Nphi/360, (180-rot_axis[1])*resol*self.env.Ntheta/180, color='blue', marker='X', markersize=10)
                
                for i in range(circle_points):
                    plt.plot(Edirs[0, i]*resol*self.env.Nphi/360, Edirs[1, i]*resol*self.env.Ntheta/180, color='blue', marker='.', markersize=6)
                    plt.plot(Sdirs[0, i]*resol*self.env.Nphi/360, Sdirs[1, i]*resol*self.env.Ntheta/180, color='red', marker='.', markersize=6)
                
                plt.colorbar()
                plt.quiver(X, Y, gradx, grady, color='gold', angles='xy', headwidth=2, headlength=4)
                name = "Env No."+str(env_no)+" (ref_reward="+str(int(ref_reward*1000)/1000)+") Delta+Grad Reward MAP"
                name_ratio = "(Ratio actions = ["+str(ratio_actions[0])+", "+str(ratio_actions[1])+"])"
                name_rot_axis = "(Rot_Axis = ["+str(int(100*rot_axis[0])/100)+", "+str(int(100*rot_axis[1])/100)+"])"
                plt.title(name+"\n"+name_ratio)
                plt.xticks(phi_ticks[0], phi_ticks[1])
                plt.yticks(theta_ticks[0], theta_ticks[1])
                plt.savefig(path+name+name_ratio+".png", dpi=300)
                plt.close()            

    def __map_tick_list(self, resol:int, N_ang:int, max_ang):
        tick_num = 12
        tick_value = []
        tick_label = []

        for i in range(tick_num):
            tick_value.append(i*(resol*N_ang)/tick_num)
            tick_label.append(str(i*int((max_ang*100)//tick_num)/100))

        return tick_value, tick_label
    

l_max = 8 
merge_num = 1
N_set = (40, 20)
lightcurve_unit_len = 100

data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_total_preprocessed.npz"
total_data = np.load(data_path)
X_total = total_data["lc_arr"]
ell_total = total_data["ell_arr"]

print("X_total shape:", X_total.shape, "| ell_total shape:", ell_total.shape)
#reward_arr = total_data["reward_arr"]

#shuffle_idx = np.arange(0, X_total.shape[0])
#np.random.shuffle(shuffle_idx)
#X_total = X_total[shuffle_idx, :]
#ell_total = ell_total[shuffle_idx, :]
#reward_arr = reward_arr[shuffle_idx]

prec = 8
reward_domain = [-100, 50] #-20, 50

##### start_idx 997부터 filter 적용시켰음 (25.09.18)
def lc_filter(lc_arr, threshold=4):
    fft_coef_zip = np.abs(np.fft.fft(lc_arr))[:lc_arr.shape[0]//2+1]
    fft_coef_zip = np.log10(fft_coef_zip+1e-8)
    log_thr = np.log10(threshold)
    return np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:])

save_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/"
lc_filtering = True
if lc_filtering: save_path = save_path + "RL_preset_filtered/"

start_idx = 997 #여기부터 해야함
final_idx = 2000 #to be next start_idx
total_data_set_defined = False
if total_data_set_defined:
    total_data_set_arr = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_RL_preset_"+str(start_idx)+".npy")
    total_data_set_info = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_pole_axis_RL_preset_info"+str(start_idx)+".npz")
    passed_idx = total_data_set_info["passed_idx"]
    reward0 = total_data_set_info["reward0"]
    
passed_idx = np.array([], dtype=int)
reward0 = np.array([], dtype=float)

X_total_new = X_total[start_idx:final_idx].copy()
ell_total_new = ell_total[start_idx:final_idx].copy()
del X_total, ell_total
X_total = X_total_new
ell_total = ell_total_new
gc.collect()

for i in tqdm(range(X_total.shape[0])):
    if i%4 == 0 and i != 0 and total_data_set_defined:
        np.save(save_path + "data_pole_axis_RL_preset_"+str(i+start_idx)+".npy",
                  total_data_set_arr)
        np.savez(save_path + "data_pole_axis_RL_preset_info"+str(i+start_idx)+".npz",
                passed_idx=passed_idx, reward0=reward0)
        final_idx = i + 0
        print(total_data_set_arr.shape)

    if i%4 == 0 and i != 0 and total_data_set_defined:
        passed_idx = np.array([], dtype=int)
        reward0 = np.array([], dtype=float)
        total_data_set_arr = None
        total_data_set_defined = False
        gc.collect()
        print("-----Dataset Seperated-----")

    #env = AstEnv(X_total[i+start_idx, :-9*merge_num], X_total[i+start_idx, -9*merge_num:], merge_num, reward_domain, N_set, lightcurve_unit_len, (True, ell_total[i+start_idx, :]))
    env = AstEnv(X_total[i, :-9*merge_num], X_total[i, -9*merge_num:], merge_num, reward_domain, N_set, lightcurve_unit_len, (True, ell_total[i, :]))
    if env.ell_err or (lc_filtering and not lc_filter(X_total[i, :-9*merge_num])):
        passed_idx = np.append(passed_idx, i+start_idx)
        reward0 = np.append(reward0, env.reward0)
        continue
    state_dim = (env.Ntheta//env.ast_obs_unit_step)*(env.Nphi//env.ast_obs_unit_step) + (lightcurve_unit_len//env.lc_obs_unit_step) + 6
    action_dim = 4

    runner = Runner(env, state_dim, action_dim)
    #runner.run(env_no=i, random=True, save=False) #when generate data for training
    runner.run(env_no=i, random=False, save=False)

    if not total_data_set_defined:
        total_data_set_arr = runner.data_set_arr
        total_data_set_defined = True
    else:
        total_data_set_arr = np.concatenate((total_data_set_arr, runner.data_set_arr[1:]), axis=0)
        print(total_data_set_arr.shape)

data_len = total_data_set_arr.shape[0]

# generate test_images
test_image_num = 10
test_image_generated = 0
for i in range(X_total[:].shape[0]):
    env = AstEnv(X_total[i, :-9*merge_num], X_total[i, -9*merge_num:], merge_num, reward_domain, N_set, lightcurve_unit_len, (True, ell_total[i, :]))
    if env.ell_err:# or env.reward0 >= 10:
        continue
    state_dim = (env.Ntheta//env.ast_obs_unit_step)*(env.Nphi//env.ast_obs_unit_step) + (lightcurve_unit_len//env.lc_obs_unit_step) + 9*5
    action_dim = 4

    runner = Runner(env, state_dim, action_dim)
    runner.run(env_no=i, random=False, save=False)

    total_data_set_arr = np.concatenate((total_data_set_arr, runner.data_set_arr[1:]), axis=0)
    print(total_data_set_arr.shape)
    test_image_generated += 1
    if test_image_generated == test_image_num:
        break

total_data_set_arr[0, 0] = data_len
total_data_set_arr[0, 1] = total_data_set_arr.shape[0] - data_len

np.save(save_path + "data_pole_axis_RL_preset_"+str(final_idx)+".npy",
            total_data_set_arr)
print(total_data_set_arr.shape)
