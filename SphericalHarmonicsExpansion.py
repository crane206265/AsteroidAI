import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.special import sph_harm_y

from Asteroid_Model import AsteroidModel
from LightCurve_Generator import LightCurve

class SphericalHarmonicsExpansion():
    def __init__(self, Asteroid : AsteroidModel, LightCurve : LightCurve, l_range=8):
        self.Asteroid = Asteroid
        self.LightCurve = LightCurve
        self.l_range = l_range
        self.coef_arr = np.zeros(((l_range+1)**2), dtype="complex_")

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
            value = np.zeros((diff_n), dtype="complex_")
        for l in range(self.l_range):
            for m in range(-l, l+1):
                idx = l**2+l+m
                if diff_n == 0:
                    value = value + self.coef_arr[idx]*sph_harm_y(l, m, theta, phi, diff_n=diff_n)
                else:
                    value = value + self.coef_arr[idx]*sph_harm_y(l, m, theta, phi, diff_n=diff_n)[diff_n]
        return value

    def SHE_RMSE(self, relative=False, percent=False):
        Asteroid = self.Asteroid

        #OLD VERSION RRMSE
        """
        rmse = 0
        rms = 0
        for i in range(Asteroid.Ntheta):
            for j in range(Asteroid.Nphi):
                rmse += (np.real(self.SHE_r_func(i*Asteroid.dtheta, j*Asteroid.dphi)).astype(np.float64) - Asteroid.pos_sph_arr[j, i, 0])**2
                rms += Asteroid.pos_sph_arr[j, i, 0]**2
        rmse = (rmse/(Asteroid.Ntheta*Asteroid.Nphi))**0.5
        """

        rmse = 0
        for i in range(Asteroid.Ntheta):
            for j in range(Asteroid.Nphi):
                if relative:
                    rmse += ((np.real(self.SHE_r_func(i*Asteroid.dtheta, j*Asteroid.dphi)).astype(np.float64) - Asteroid.pos_sph_arr[j, i, 0])/Asteroid.pos_sph_arr[j, i, 0])**2
                else:
                    rmse += (np.real(self.SHE_r_func(i*Asteroid.dtheta, j*Asteroid.dphi)).astype(np.float64) - Asteroid.pos_sph_arr[j, i, 0])**2
        rmse = (rmse/(Asteroid.Ntheta*Asteroid.Nphi))**0.5

        if percent:
            rmse *= 100
        print("SHE MSE :", rmse, end=' ')
        print("(relative : "+str(relative)+")")

    def SHE_display(self, divN=(40, 40)):
        phi = np.linspace(0, 2*np.pi, divN[0])
        theta = np.linspace(0, np.pi, divN[1])
        phi, theta = np.meshgrid(phi, theta)
        cart_coord = AsteroidModel.sph2cart(np.real(self.SHE_r_func(theta, phi)).astype(np.float64), phi, theta)
        gridX = cart_coord[0]
        gridY = cart_coord[1]
        gridZ = cart_coord[2]
        
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection":"3d"})
        ax.set_box_aspect((1, 1, 1))
        lim_set = (-10, 10)
        ax.set_xlim(lim_set)
        ax.set_xlabel('X')
        ax.set_ylim(lim_set)
        ax.set_ylabel('Y')
        ax.set_zlim(lim_set)
        ax.set_zlabel('Z')
        ax.set_title("Asteroid Model - Spherical Hamonics Expansion")

        ax.plot_surface(gridX, gridY, gridZ)
        plt.show()
        
    def Ast_SHE_display(self, SHE_divN=(40, 40)):
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        

        gridX = self.Asteroid.pos_cart_arr[:, :, 0]
        gridY = self.Asteroid.pos_cart_arr[:, :, 1]
        gridZ = self.Asteroid.pos_cart_arr[:, :, 2]
        ax1.set_box_aspect((1, 1, 1))
        lim_set = (-10, 10)
        ax1.set_xlim(lim_set)
        ax1.set_xlabel('X')
        ax1.set_ylim(lim_set)
        ax1.set_ylabel('Y')
        ax1.set_zlim(lim_set)
        ax1.set_zlabel('Z')
        ax1.set_title("Asteroid Model")

        ax1.plot_surface(gridX, gridY, gridZ)


        phi = np.linspace(0, 2*np.pi, SHE_divN[0])
        theta = np.linspace(0, np.pi, SHE_divN[1])
        phi, theta = np.meshgrid(phi, theta)
        cart_coord = AsteroidModel.sph2cart((np.real(self.SHE_r_func(theta, phi)).astype(np.float64), phi, theta))
        gridX = cart_coord[0]
        gridY = cart_coord[1]
        gridZ = cart_coord[2]
        
        ax2.set_box_aspect((1, 1, 1))
        ax2.set_xlim(lim_set)
        ax2.set_xlabel('X')
        ax2.set_ylim(lim_set)
        ax2.set_ylabel('Y')
        ax2.set_zlim(lim_set)
        ax2.set_zlabel('Z')
        ax2.set_title("Asteroid Model - Spherical Hamonics Expansion")

        ax2.plot_surface(gridX, gridY, gridZ)
        
        plt.show()


    #SHE LC - Not Using
    """
    def flux_SHE(self, flux0, direction):
        
        #sun_direction = direction[0]
        #earth_direction = direction[1]
        
        dtheta_num, dphi_num = 20, 10
        dtheta = np.pi/dtheta_num
        dphi = 2*np.pi/dphi_num
        #flux = dblquad(self.__flux_SHE_integrand, 0, 2*np.pi, 0, np.pi, (flux0, direction), 1e-3)
        flux = 0
        for i in range(dtheta_num):
            for j in range(dphi_num):
                flux += self.__flux_SHE_integrand(i*dtheta, j*dphi, flux0, direction)*dtheta*dphi
        return flux
        
    def __flux_SHE_integrand(self, theta, phi, flux0, direction):
        if LA.norm(self.__surf_cross(theta, phi)) < 1e-8:
            return 0
        else:
            return flux0*max(0, np.dot(self.__surf_cross(theta, phi), direction[0]))*max(0, np.dot(self.__surf_cross(theta, phi), direction[1]))/LA.norm(self.__surf_cross(theta, phi))

    def __surf_cross(self, theta, phi):
        r = np.real(self.SHE_r_func(theta, phi, 0))
        r_theta, r_phi = np.real(self.SHE_r_func(theta, phi, 1))
        x_component = r*(np.sin(phi)*r_phi + np.cos(phi)*np.sin(theta)*(np.cos(theta)*r_theta - r*np.sin(theta)))
        y_component = -r*(np.cos(phi)*r_phi + np.sin(phi)*np.sin(theta)*(np.cos(theta)*r_theta - r*np.sin(theta)))
        z_component = r*np.sin(theta)*(np.sin(theta)*r_theta + np.cos(theta)*r)
        
        return np.array([x_component, y_component, z_component])

    def lc_gen_SHE(self, rot_div_N, len_ratio):
        #lightcurve arr. generation (for SHE)
        #* for rotating animation, set rot_div_N same as param. of rotate_anim

        #<input>
        #rot_div_N : # of (discrete) rotation per 1 period
        #len_ratio : length of lc for ratio of period
        
        self.lc_SHE_arr = np.zeros(int(rot_div_N*len_ratio))
        dt = 1/rot_div_N
        w0 = self.LightCurve.initial_w0
        rot_angle = 0
        print("GENERATING LIGHTCURVE by SPHERICAL HARMONICS EXPANSION...")
        for i in tqdm(range(int(rot_div_N*len_ratio))):    
            self.lc_SHE_arr[i] = self.flux_SHE(1, self.LightCurve.direction_cal(rot_angle))
            w0, rot_angle = self.LightCurve.precession(w0, self.LightCurve.inertia_tensor, 0, rot_angle, dt, "No_Precession")

    def lc_SHE_display(self, rot_div_N, len_ratio, comparision=True):
        lc_time = np.arange(int(rot_div_N*len_ratio))
        plt.plot(lc_time, self.lc_SHE_arr, 'b')
        if comparision:
            plt.plot(lc_time, self.LightCurve.lc_arr, 'r')
        plt.show()    
    """


class SHEcoefDisplay():
    def __init__(self, y0, pred, l_max=8):
        self.y0 = np.array(y0)
        self.pred = np.array(pred)
        _ = 0 # trash value
        self.y0_SHE = SphericalHarmonicsExpansion(_, _, l_range=l_max)
        self.pred_SHE = SphericalHarmonicsExpansion(_, _, l_range=l_max)
        self.y0_SHE.coef_arr = self.y0
        self.pred_SHE.coef_arr = self.pred
        self.SHE_divN = (40, 20)
        self.coef_arr_display()

    def coef_arr_display(self):
        """
        SHE display function for pred / y0 coef arr
        """
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        lim_set = (-10, 10)
        phi = np.linspace(0, 2*np.pi, self.SHE_divN[0])
        theta = np.linspace(0, np.pi, self.SHE_divN[1])
        phi, theta = np.meshgrid(phi, theta)

        y0_cart_coord = AsteroidModel.sph2cart((np.real(self.y0_SHE.SHE_r_func(theta, phi)).astype(np.float64), phi, theta))
        y0_gridX = y0_cart_coord[0]
        y0_gridY = y0_cart_coord[1]
        y0_gridZ = y0_cart_coord[2]
        
        ax1.set_box_aspect((1, 1, 1))
        ax1.set_xlim(lim_set)
        ax1.set_xlabel('X')
        ax1.set_ylim(lim_set)
        ax1.set_ylabel('Y')
        ax1.set_zlim(lim_set)
        ax1.set_zlabel('Z')
        ax1.set_title("Correct Model - Spherical Hamonics Expansion")

        ax1.plot_surface(y0_gridX, y0_gridY, y0_gridZ)

        pred_cart_coord = AsteroidModel.sph2cart((np.real(self.pred_SHE.SHE_r_func(theta, phi)).astype(np.float64), phi, theta))
        pred_gridX = pred_cart_coord[0]
        pred_gridY = pred_cart_coord[1]
        pred_gridZ = pred_cart_coord[2]
        
        ax2.set_box_aspect((1, 1, 1))
        ax2.set_xlim(lim_set)
        ax2.set_xlabel('X')
        ax2.set_ylim(lim_set)
        ax2.set_ylabel('Y')
        ax2.set_zlim(lim_set)
        ax2.set_zlabel('Z')
        ax2.set_title("Predicted Model - Spherical Hamonics Expansion")

        ax2.plot_surface(pred_gridX, pred_gridY, pred_gridZ)
        
        plt.show()



class R_ArrDisplay():
    def __init__(self, y0, pred, N_set):
        self.y0 = np.array(y0)
        self.pred = np.array(pred)
        PI = 3.1415926535
        self.Nphi, self.Ntheta = N_set[0], N_set[1]
        self.dphi, self.dtheta = 2*PI/self.Nphi, PI/self.Ntheta
        self.y0 = y0.reshape(self.Nphi, self.Ntheta)
        self.pred = pred.reshape(self.Nphi, self.Ntheta)
        self.coef_arr_display()

    def coef_arr_display(self):
        """
        SHE display function for pred / y0 coef arr
        """
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2, projection='3d')
        
        lim_set = (-2, 2)

        y0_gridX = np.zeros((self.Nphi+1, self.Ntheta))
        y0_gridY = np.zeros((self.Nphi+1, self.Ntheta))
        y0_gridZ = np.zeros((self.Nphi+1, self.Ntheta))
        pred_gridX = np.zeros((self.Nphi+1, self.Ntheta))
        pred_gridY = np.zeros((self.Nphi+1, self.Ntheta))
        pred_gridZ = np.zeros((self.Nphi+1, self.Ntheta))
        for i in range(self.Nphi):
            for j in range(self.Ntheta):
                phi_ij = (j%2)*(self.dphi/2) + i*self.dphi
                theta_ij = j*self.dtheta
                pos_cart_temp = AsteroidModel.sph2cart((self.y0[i, j], phi_ij, theta_ij))
                y0_gridX[i, j] = pos_cart_temp[0]
                y0_gridY[i, j] = pos_cart_temp[1]
                y0_gridZ[i, j] = pos_cart_temp[2]
                pos_cart_temp = AsteroidModel.sph2cart((self.pred[i, j], phi_ij, theta_ij))
                pred_gridX[i, j] = pos_cart_temp[0]
                pred_gridY[i, j] = pos_cart_temp[1]
                pred_gridZ[i, j] = pos_cart_temp[2]
        y0_gridX[-1, :] = y0_gridX[0, :]
        y0_gridY[-1, :] = y0_gridY[0, :]
        y0_gridZ[-1, :] = y0_gridZ[0, :]
        pred_gridX[-1, :] = pred_gridX[0, :]
        pred_gridY[-1, :] = pred_gridY[0, :]
        pred_gridZ[-1, :] = pred_gridZ[0, :]

        ax1.set_box_aspect((1, 1, 1))
        ax1.set_xlim(lim_set)
        ax1.set_xlabel('X')
        ax1.set_ylim(lim_set)
        ax1.set_ylabel('Y')
        ax1.set_zlim(lim_set)
        ax1.set_zlabel('Z')
        ax1.set_title("Correct Model - Spherical Hamonics Expansion")

        ax1.plot_surface(y0_gridX, y0_gridY, y0_gridZ)
        
        ax2.set_box_aspect((1, 1, 1))
        ax2.set_xlim(lim_set)
        ax2.set_xlabel('X')
        ax2.set_ylim(lim_set)
        ax2.set_ylabel('Y')
        ax2.set_zlim(lim_set)
        ax2.set_zlabel('Z')
        ax2.set_title("Predicted Model - Spherical Hamonics Expansion")

        ax2.plot_surface(pred_gridX, pred_gridY, pred_gridZ)
        
        plt.show()

"""
np.random.seed(1)
test_ast = AsteroidModel(axes=(7, 4, 5), N_set=(40, 20), tilt_mode="random") #745
test_ast.base_fitting_generator()
test_ast.cut_ast(20, 0) #10,0
#test_ast.wobble_r(0) #10
test_ast.surf_vec_cal()

test_lc = LightCurve(Asteroid=test_ast, Keplerian_elem=(3, 0, 0, 0, 0), eps=(0, 0), principle_axis=True)
print(test_ast.inertia_tensor)
test_ast.COM_correction(0.5)

test_lc.orbit_coord_set()
test_lc.lc_gen(100, 1) #use lc_gen before use rotate_anim
test_lc.rotate_anim(100, 1)

test_SHE = SphericalHarmonicsExpansion(Asteroid=test_ast, LightCurve=test_lc, l_range=8)
test_SHE.coef_arr = test_SHE.SHE_coef()
test_SHE.Ast_SHE_display()
test_SHE.SHE_RMSE(relative=True)
print(test_SHE.coef_arr)

#test_SHE.lc_gen_SHE(100, 1)
#test_SHE.lc_SHE_display(100, 1)

#print(test_SHE.lc_SHE_arr)

#plt.plot(test_lc.lc_arr)
#plt.show()
#"""
