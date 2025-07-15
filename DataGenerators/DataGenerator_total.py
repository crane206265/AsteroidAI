import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from scipy.special import sph_harm_y

from tqdm import tqdm

import os
import gc

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

#-----------------------------------

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
    
    def coord2discr(self, coord, param):
        prec, bound = param
        x_idx = int((coord[0] + bound)/prec)
        y_idx = int((coord[1] + bound)/prec)
        z_idx = int((coord[2] + bound)/prec)
        return x_idx, y_idx, z_idx
    
    def discr2coord(self, coord, param):
        prec, bound = param
        x = coord[0]*prec - bound
        y = coord[1]*prec - bound
        z = coord[2]*prec - bound
        return x, y, z

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
            if self.interior_cal:
                self.__ellipsoid_is_interior()

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
        
        # valid region (discrete coord)
        self.valid_lw_bd, _, _ = self.coord2discr((-self.max_r, 0, 0), self.discr_param)
        self.valid_up_bd, _, _ = self.coord2discr((self.max_r, 0, 0), self.discr_param)
        

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

    def __ellipsoid_is_interior(self, radi=[-1, -1, -1], tilt_angle=[-1, -1]):
        if radi[0] == -1:
            a = self.axes_R[0]
        if radi[1] == -1:
            b = self.axes_R[1]
        if radi[2] == -1:
            c = self.axes_R[2]

        if tilt_angle[0] == -1:
            long = self.tilt[0]
        else:
            long = tilt_angle[0]
        if tilt_angle[0] == -1:
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
        
        #print("UPDATING [is_interior] MATRIX : for ellipsoid generation"
        for x_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
            for y_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                for z_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                    x, y, z = self.discr2coord((x_idx, y_idx, z_idx), self.discr_param)
                    pos_vec = np.array([x, y, z]).T
                    s = pos_vec.T@R_arr.T@A_arr@R_arr@pos_vec 
                    if s <= 1:
                        self.is_interior[x_idx, y_idx, z_idx] = 1
    
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
        if sph_num > 1: raise NotImplementedError
        sph_temp = CutterSphere(self, kwargs['assigned'], kwargs['mode'], kwargs['pos_sph'][0], kwargs['pos_sph'][1], kwargs['pos_sph'][2], kwargs['pos_sph'][3])
        for i in range(self.Nphi+1):
            for j in range(self.Ntheta+1):
                if sph_temp.f(self.pos_cart_arr[i, j]) < 0:
                    self.pos_sph_arr[i, j, 0] = sph_temp.r_f(self.pos_sph_arr[i, j, 1:])
                    self.pos_cart_arr[i, j] = AsteroidModel.sph2cart(self.pos_sph_arr[i, j])
        if self.interior_cal:
            self.__is_interior_sph_cut(sph_temp)

    def __is_interior_sph_cut(self, sph_temp):
        #print("UPDATING [is_interior] MATRIX : for sphere cut")
        for x_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
            for y_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                for z_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                    x, y, z = self.discr2coord((x_idx, y_idx, z_idx), self.discr_param)
                    pos_vec = np.array([x, y, z])
                    if sph_temp.f(pos_vec) < 0:
                        self.is_interior[x_idx, y_idx, z_idx] = 0

    def COM_correction(self, rho_func, const_rho="Y"):
        if const_rho == "Y":
            for component in range(3):
                prec = self.discr_param[0]
                bound = self.discr_param[1]
                N_index = 2*int(bound/prec) + 1
                integrand = np.zeros((N_index, N_index, N_index))
                for x_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                    for y_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                        for z_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                            x, y, z = self.discr2coord((x_idx, y_idx, z_idx), self.discr_param)
                            if component == 0:
                                integrand[x_idx, y_idx, z_idx] = self.is_interior[x_idx, y_idx, z_idx]*x
                            if component == 1:
                                integrand[x_idx, y_idx, z_idx] = self.is_interior[x_idx, y_idx, z_idx]*y
                            if component == 2:
                                integrand[x_idx, y_idx, z_idx] = self.is_interior[x_idx, y_idx, z_idx]*z
                self.COM_vec[component] = np.sum(integrand*(prec**3))/np.sum(self.is_interior*(prec**3)+1e-8)
            #print(self.COM_vec)
            #COM_vec으로 평행이동 구현 필요요
        else:
            pass

    def inertia_tensor_cal(self, rho_func, const_rho="Y"):
        """
        Calculate inertia tensor of Asteroid
        rho_func : density funtion
        const_rho : if uniform density
        """
        if const_rho == "Y":
            #print("CALCULATING INERTIA TENSOR FOR THE ASTEROID")
            for inertia_idx in range(6):
                prec = self.discr_param[0]
                bound = self.discr_param[1]
                N_index = 2*int(bound/prec) + 1
                integrand = np.zeros((N_index, N_index, N_index))
                for x_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                    for y_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                        for z_idx in range(self.valid_lw_bd, self.valid_up_bd+1):
                            x, y, z = self.discr2coord((x_idx, y_idx, z_idx), self.discr_param)
                            if inertia_idx == 0:
                                integrand[x_idx, y_idx, z_idx] = rho_func*self.is_interior[x_idx, y_idx, z_idx]*(y*y + z*z)
                            if inertia_idx == 3:
                                integrand[x_idx, y_idx, z_idx] = rho_func*self.is_interior[x_idx, y_idx, z_idx]*(x*x + z*z)
                            if inertia_idx == 5:
                                integrand[x_idx, y_idx, z_idx] = rho_func*self.is_interior[x_idx, y_idx, z_idx]*(x*x + y*y)
                            if inertia_idx == 1:
                                integrand[x_idx, y_idx, z_idx] = -rho_func*self.is_interior[x_idx, y_idx, z_idx]*x*y
                            if inertia_idx == 2:
                                integrand[x_idx, y_idx, z_idx] = -rho_func*self.is_interior[x_idx, y_idx, z_idx]*x*z
                            if inertia_idx == 4:
                                integrand[x_idx, y_idx, z_idx] = -rho_func*self.is_interior[x_idx, y_idx, z_idx]*y*z
                inertia_res = np.sum(integrand*(prec**3))
                if inertia_idx == 0:
                    self.inertia_tensor[0, 0] = inertia_res
                if inertia_idx == 1:
                    self.inertia_tensor[0, 1] = inertia_res
                    self.inertia_tensor[1, 0] = inertia_res
                if inertia_idx == 2:
                    self.inertia_tensor[0, 2] = inertia_res
                    self.inertia_tensor[2, 0] = inertia_res
                if inertia_idx == 3:
                    self.inertia_tensor[1, 1] = inertia_res
                if inertia_idx == 4:
                    self.inertia_tensor[1, 2] = inertia_res
                    self.inertia_tensor[2, 1] = inertia_res
                if inertia_idx == 5:
                    self.inertia_tensor[2, 2] = inertia_res
        else:
            pass

    def copy(self):
        ast_copy = AsteroidModel((1, 1, 1), (self.Nphi, self.Ntheta), interior_cal=self.interior_cal, coord2discrete_param=self.discr_param)
        ast_copy.pos_sph_arr = self.pos_sph_arr.copy()
        ast_copy.pos_cart_arr = self.pos_cart_arr.copy()
        ast_copy.surf_vec_arr = self.surf_vec_arr.copy()
        ast_copy.albedo_arr = self.albedo_arr.copy()
        ast_copy.axes_R = self.axes_R.copy()
        ast_copy.tilt = self.tilt.copy()
        ast_copy.is_interior = self.is_interior.copy()
        ast_copy.max_r = self.max_r + 0
        ast_copy.valid_lw_bd = self.valid_lw_bd + 0
        ast_copy.valid_up_bd = self.valid_up_bd + 0
        ast_copy.COM_vec = self.COM_vec.copy()

        return ast_copy

#-----------------------------------

class LightCurve():
    def __init__(self, Asteroid : AsteroidModel, Keplerian_elem, eps, principle_axis=True):
        """
        bring the variables from Asteroid Class\\
        N_set = (Nphi, Ntheta)\\
        surf_vec_arr = surf_vec_arr\\
        albedo_arr = albdeo_arr\\
        Keplerian_elem = (a, ecc, asc_long, inc, peri, eps)
            * a : semi-major axis
            * ecc : eccentricity
            * asc_long : longitude of ascending node
            * inc : inclination
            * peri : argument of periapsis\\
            
        eps = (eps_phi, eps_theta)
            * eps_phi : precession rate
            * eps_theta : axial tilt
        """
        self.Asteroid = Asteroid
        self.a = Keplerian_elem[0]
        self.ecc = Keplerian_elem[1]
        self.Keplerian = Keplerian_elem[2:]
        self.eps = eps

        self.lc_arr = np.array([])

        self.K = self.rotArr(-self.Keplerian[2], "z")@self.rotArr(-self.Keplerian[1], "x")@self.rotArr(-self.Keplerian[0], "z")
        self.R_eps = self.rotArr(-self.eps[1], "y")@self.rotArr(-self.eps[0], "z")

        Asteroid.inertia_tensor_cal(0.5)
        Asteroid.COM_correction(0.5)
        self.inertia_tensor = Asteroid.inertia_tensor
        self.initial_w0 = self.R_eps.T@np.array([0, 0, 1]).T
        if principle_axis:
            self.principle_axis(1)
            initial_eps = np.empty(2)
            initial_eps[0] = np.arctan2(self.initial_w0[1], self.initial_w0[0])
            initial_eps[1] = np.arccos(self.initial_w0[2]/LA.norm(self.initial_w0))
            self.R_eps = self.rotArr(-initial_eps[1], "y")@self.rotArr(-initial_eps[0], "z")
    

        self.rot_angle_list = [0]
        self.R_eps_list = [self.R_eps]


    def rotArr(self, angle, axis):
        return AsteroidModel.rotArr(angle, axis)

    def orbit_coord_set(self, f=-1, ecl_O=-1, mode="random"):
        """
        <input>
        f : true anomaly
        ecl_O = (ecl_long_O, ecl_lat_O) : ecliptic coordinate of asteroid observed from the Earth
        """
        if mode == "assigned":
            self.f = f
            self.ecl_O = ecl_O
        elif mode == "random":
            self.f = 2*np.pi*np.random.rand(1)[0]
            ecl_long_O = 2*np.pi*np.random.rand(1)[0]
            tan_ecl_lat_O_max = np.tan(self.Keplerian[1]) / (1 - 1/(self.a*(1-self.ecc)*np.cos(self.Keplerian[1])))
            ecl_lat_O = 2*np.arctan(tan_ecl_lat_O_max)*np.random.rand(1)[0] - np.arctan(tan_ecl_lat_O_max)
            self.ecl_O = (ecl_long_O, ecl_lat_O)
        else:
            raise ValueError("Unimplemented Mode")

    def orb2geo(self, vec_orb, rot_angle):
        return self.rotArr(-rot_angle, "z")@self.R_eps@vec_orb

    def geo2orb(self, vec_geo, rot_angle):
        return self.R_eps.T@self.rotArr(rot_angle, "z")@vec_geo

    def precession(self, w0, inertia_tensor, ex_torque, rot_angle_0, dt, mode="Precession"):
        """
        as col vec
        """
        if mode == "Precession":
            dw = LA.inv(inertia_tensor)@(np.cross(inertia_tensor@w0, w0)+ex_torque)*dt
            w_updated = w0 + dw
            pseudo_eps = [0, 0]
            pseudo_eps[0] = np.arctan2(w_updated[1], w_updated[0])
            pseudo_eps[1] = np.arccos(w_updated[2]/LA.norm(w_updated))
            self.R_eps = self.rotArr(pseudo_eps[0], "z")@self.rotArr(pseudo_eps[1], "y")
        elif mode == "No_Precession":
            #self.R_eps = self.rotArr(self.eps[0], "z")@self.rotArr(self.eps[1], "y")
            w_updated = w0
            dw = 0

        rot_w = LA.norm(w_updated)
        rot_angle = rot_angle_0 + rot_w*dt
        self.rot_angle_list.append(rot_angle)
        self.R_eps_list.append(self.R_eps)
        #print("dw :", dw)
        return w_updated, rot_angle

    def principle_axis(self, period):
        """
        set rotation axis as principle axis
        period : rotation period
        """
        ang_vel = 2*np.pi/period
        self.initial_w0 = ang_vel*LA.eig(self.inertia_tensor)[1][0]/LA.norm(LA.eig(self.inertia_tensor)[1][0])
        #print("Iw :", self.Asteroid.inertia_tensor@self.initial_w0)
        #print("w :", self.initial_w0)

    def direction_cal(self, rot_angle):
        sun_dir_orb = self.rotArr(self.f, "z")@self.K@np.array([1, 0, 0]).T
        earth_dir_orb = self.rotArr(-self.ecl_O[0], "z")@self.rotArr(-self.ecl_O[1], "y")@np.array([1, 0, 0]).T
        sun_dir_geo = self.orb2geo(sun_dir_orb, rot_angle)
        earth_dir_geo = self.orb2geo(earth_dir_orb, rot_angle)
        return (sun_dir_geo, earth_dir_geo)

    def flux(self, flux0, direction):
        """
        flux calculater
        # rotating light source instead asteroid -> less calculation
            -> source rotating direction : opposite to one's of asteroid 

        <input>
        flux0 : flux from sun
        direction = (sun_direction, earth_direction)
            * sun_direction : direction of light source (from asteroid) (unit vector)
            * earth_direction : direction of observer (from asteroid) (unit vector)

        <output>
        flux_surf : flux arr. for each surface
        """
        sun_direction = direction[0]
        earth_direction = direction[1]
        flux0_vec = flux0*sun_direction
        flux_surf = np.zeros((self.Asteroid.Nphi, self.Asteroid.Ntheta, 2))
        flux_surf_obs = np.zeros((self.Asteroid.Nphi, self.Asteroid.Ntheta, 2))

        for i in range(self.Asteroid.Nphi):
            for j in range(self.Asteroid.Ntheta):
                flux_in_0 = max(0, np.dot(flux0_vec, self.Asteroid.surf_vec_arr[i, j, 0]))
                flux_in_1 = max(0, np.dot(flux0_vec, self.Asteroid.surf_vec_arr[i, j, 1]))
                
                if LA.norm(self.Asteroid.surf_vec_arr[i, j, 0]) >= 1e-8:
                    flux_surf[i, j, 0] = flux_in_0*self.Asteroid.albedo_arr[i, j, 0]
                    flux_surf_obs[i, j, 0] = flux_surf[i, j, 0]*max(0, np.dot(earth_direction, self.Asteroid.surf_vec_arr[i, j, 0]))/LA.norm(self.Asteroid.surf_vec_arr[i, j, 0])
                else:
                    flux_surf[i, j, 0] = 0
                    flux_surf_obs[i, j, 0] = 0
                
                if LA.norm(self.Asteroid.surf_vec_arr[i, j, 1]) >= 1e-8:
                    flux_surf[i, j, 1] = flux_in_1*self.Asteroid.albedo_arr[i, j, 1]
                    flux_surf_obs[i, j, 1] = flux_surf[i, j, 1]*max(0, np.dot(earth_direction, self.Asteroid.surf_vec_arr[i, j, 1]))/LA.norm(self.Asteroid.surf_vec_arr[i, j, 1])
                else:
                    flux_surf[i, j, 1] = 0
                    flux_surf_obs[i, j, 1] = 0

        return flux_surf_obs

    def lc_gen(self, rot_div_N, len_ratio):
        """
        lightcurve arr. generation
        * for rotating animation, set rot_div_N same as param. of rotate_anim

        <input>
        rot_div_N : # of (discrete) rotation per 1 period
        len_ratio : length of lc for ratio of period
        """
        self.lc_arr = np.zeros(int(rot_div_N*len_ratio))
        dt = 2*np.pi/rot_div_N
        w0 = self.initial_w0
        rot_angle = 0
        for i in range(int(rot_div_N*len_ratio)):    
            self.lc_arr[i] = np.sum(self.flux(1, self.direction_cal(rot_angle)))
            w0, rot_angle = self.precession(w0, self.inertia_tensor, 0, rot_angle, dt, "No_Precession")

#-----------------------------------

class SphericalHarmonicsExpansion():
    def __init__(self, Asteroid : AsteroidModel, LightCurve : LightCurve, l_range=8):
        self.Asteroid = Asteroid
        self.LightCurve = LightCurve
        self.l_range = l_range
        self.coef_arr = np.zeros(((l_range+1)**2), dtype=np.complex64)

    def SHE_coef(self):
        """
        Find spherical harmonics expansion of r-function
        OUTPUT : SHE coef. array

        l_range : range of l [0, _]
        --> # of coef. = (l_range + 1)^2

        * Yml coef : index l^2+l+m
        """
        Asteroid = self.Asteroid
        dtheta = Asteroid.dtheta
        dphi = Asteroid.dphi

        for l in range(self.l_range+1):
            for m in range(-l, l+1):
                idx = l**2+l+m
                for i in range(Asteroid.Ntheta):
                    for j in range(Asteroid.Nphi):
                        self.coef_arr[idx] += Asteroid.pos_sph_arr[j, i, 0]*np.conjugate(sph_harm_y(l, m, i*dtheta, j*dphi))*np.sin(i*dtheta)*dtheta*dphi
        return self.coef_arr
    
    def SHE_r_func(self, theta, phi, diff_n=0):
        if diff_n == 0:
            value = 0
        else:
            value = np.zeros((diff_n), dtype=np.complex64)
        for l in range(self.l_range):
            for m in range(-l, l+1):
                idx = l**2+l+m
                if diff_n == 0:
                    value = value + self.coef_arr[idx]*sph_harm_y(l, m, theta, phi, diff_n=diff_n)
                else:
                    value = value + self.coef_arr[idx]*sph_harm_y(l, m, theta, phi, diff_n=diff_n)[diff_n]
        return value

#-----------------------------------

#DATA GEN
def data_gen(file_num, ast_num = 50, lc_num = 10, lc_len = 200, coef_l = 8, N = (40, 20), transfer = False):

    lc_arr = np.zeros((ast_num, lc_num, lc_len))
    lc_info = np.zeros((ast_num, lc_num, 2, 3))
    coef_arr = np.zeros((ast_num, (coef_l+1)**2), dtype=np.complex64)
    rot_axis = np.zeros((ast_num, 3))
    ell_approx = np.zeros((ast_num, 5))

    print("GENERATING LIGHTCURVE DATASET...")
    print("File Name : data_"+str(file_num)+".npz")
    print("Save Path : \".../asteroid_AI/data/\"")
    for i in tqdm(range(ast_num*lc_num)):
        if i%ast_num == 0: 
            if transfer:
                rand_radi = tuple(4*np.random.rand(3)+4)
                rand_tilt = (2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0])
                ast_temp = AsteroidModel(axes=rand_radi, N_set=N, tilt_mode="assigned", tilt=rand_tilt, interior_cal=False)
                ast_temp.base_fitting_generator()
                ast_temp.surf_vec_cal()
                lc_temp = LightCurve(Asteroid=ast_temp, Keplerian_elem=(3, 0, 2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0]/6, 2*np.pi*np.random.rand(1)[0]), 
                                    eps=(0, 0), principle_axis=True)
            else:
                #while(True):
                rand_radi = tuple(4*np.random.rand(3)+4)
                rand_tilt = (2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0])
                ast_temp = AsteroidModel(axes=rand_radi, N_set=N, tilt_mode="assigned", tilt=rand_tilt, interior_cal=False)
                ast_temp.base_fitting_generator()
                #ast_temp.cut_ast(20, 0, pos_sph=[None, None, None, None])
                for j in range(40):
                    ast_temp.cut_ast(1, 0, pos_sph=[np.random.rand(1)[0], np.random.rand(1)[0], 0.5*np.random.rand(1)[0], 0.5*np.random.rand(1)[0]], assigned=True, mode="ratio_assign")
                ast_temp.surf_vec_cal()
                lc_temp = LightCurve(Asteroid=ast_temp, Keplerian_elem=(3, 0, 2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0]/6, 2*np.pi*np.random.rand(1)[0]), 
                                    eps=(0, 0), principle_axis=False)
                #No need to COM correction : not highly cut asteroid
                #if ast_temp.COM_vec[0] < 0.5 and ast_temp.COM_vec[1] < 0.5 and ast_temp.COM_vec[2] < 0.5:
                #    break
                
                gc.collect()
            SHE_temp = SphericalHarmonicsExpansion(Asteroid=ast_temp, LightCurve=lc_temp, l_range=coef_l)
            SHE_temp.coef_arr = SHE_temp.SHE_coef()

        coef_arr[i//lc_num] = SHE_temp.coef_arr
        rot_axis[i//lc_num] = lc_temp.initial_w0/LA.norm(lc_temp.initial_w0)
        for j in range(3): ell_approx[i//lc_num, j] = rand_radi[j] + 0.2*np.random.rand(1)[0]
        for j in range(2): ell_approx[i//lc_num, j+3] = rand_tilt[j] + 0.1*np.pi*np.random.rand(1)[0]
            
        while(True):
            lc_temp.orbit_coord_set(mode="random")
            dir_threshold = np.pi*(1/2)
            if np.cos(dir_threshold) <= np.dot(lc_temp.direction_cal(0)[0], lc_temp.direction_cal(0)[1]):
                break
        lc_temp.lc_gen(100, lc_len/100)

        lc_info[i//lc_num, i%lc_num, 0] = lc_temp.rotArr(lc_temp.f, "z")@lc_temp.K@np.array([1, 0, 0]).T #sun direction
        lc_info[i//lc_num, i%lc_num, 1] = lc_temp.rotArr(-lc_temp.ecl_O[0], "z")@lc_temp.rotArr(-lc_temp.ecl_O[1], "y")@np.array([1, 0, 0]).T #earth direction
        lc_arr[i//lc_num, i%lc_num] = lc_temp.lc_arr
        gc.collect()

    if not transfer:
        #np.savez(data_folder_path+"data_"+str(file_num)+".npz",
        #        lc_arr=lc_arr, lc_info=lc_info, coef_arr=coef_arr, rot_axis=rot_axis)
        np.savez(data_folder_path+"data_pole_axis_"+str(file_num)+".npz",
                lc_arr=lc_arr, lc_info=lc_info, coef_arr=coef_arr, rot_axis=rot_axis, ell_approx=ell_approx) # axis is fixed at pole
    else:
        np.savez(data_folder_path+"transfer_data/transfer_data_"+str(file_num)+".npz",
                lc_arr=lc_arr, lc_info=lc_info, coef_arr=coef_arr, rot_axis=rot_axis, ell_approx=ell_approx)
    
def data_concatenate(data_folder_path, lc_num = 10):
    """
    Concantenate the data to the form that can be directly used for the model
    [X_total, Y_total]
    * lc_num <- same value with what you used at data_gen function
    """

    folder_path = data_folder_path
    file_list = os.listdir(folder_path)
    file_list = [file for file in file_list if file.endswith(".npz")]
    file_list = [file for file in file_list if 'pole_axis_' in file]

    file_list.remove("data_pole_axis_total000.npz") if "data_pole_axis_total000.npz" in file_list else None

    if 'data_pole_axis_total.npz' in file_list:
        file_1st = np.load(folder_path+'data_pole_axis_total.npz')
        file_list.remove('data_pole_axis_total.npz')

        X_total = file_1st['X_total']
        Y_total = file_1st['Y_total']

    else:
        file_1st = np.load(folder_path+file_list[0])
        file_list = file_list[1:]

        # flatten lc_info
        shape_temp = file_1st['lc_info'].shape
        flat_temp = file_1st['lc_info'].reshape(shape_temp[0], shape_temp[1], 6)

        X_total = np.concatenate((file_1st['lc_arr'], flat_temp), axis=2)
        Y_total = np.concatenate((file_1st['coef_arr'], file_1st['rot_axis'], file_1st['ell_approx']), axis=1)

        # repeat & reshape X_total and Y_total
        X_total = X_total.reshape(-1, X_total.shape[-1])
        Y_total = np.repeat(Y_total, lc_num, axis=0)
    

    print("CONCATENATING FILES...")
    for file_name in tqdm(file_list):
        file_temp = np.load(folder_path+file_name)

        shape_temp = file_temp['lc_info'].shape
        flat_temp = file_temp['lc_info'].reshape(shape_temp[0], shape_temp[1], 6)

        X_temp = np.concatenate((file_temp['lc_arr'], flat_temp), axis=2)
        Y_temp = np.concatenate((file_temp['coef_arr'], file_temp['rot_axis'], file_temp['ell_approx']), axis=1)
        X_temp = X_temp.reshape(-1, X_temp.shape[-1])
        Y_temp = np.repeat(Y_temp, lc_num, axis=0)

        X_total = np.concatenate((X_total, X_temp), axis=0)
        Y_total = np.concatenate((Y_total, Y_temp), axis=0)


    print("X_total shape :", X_total.shape)
    print("Y_total shape :", Y_total.shape)
    np.savez(data_folder_path+"data_pole_axis_total.npz", X_total=X_total, Y_total=Y_total)


data_folder_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/"

start_file_num = 156 #normal
end_file_num = 169 #normal

transfer = False
#start_file_num = 57 #transfer
#end_file_num = 200 #transfer

for file_num in range(start_file_num, end_file_num+1):
    data_gen(file_num, ast_num = 25, lc_num = 20, lc_len = 200, coef_l = 8, N = (40, 20), transfer=transfer)
    continue

#data_concatenate(data_folder_path=data_folder_path, lc_num=20)
