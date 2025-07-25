import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm

import os
import gc



class AsteroidModel():
    def __init__(self, axes, N_set, tilt_mode="assigned", tilt=(0, 0), coord2discrete_param=(0.5, 10), interior_cal=True):
        self.Nphi = N_set[0]
        self.Ntheta = N_set[1]
        self.dphi = 2*np.pi/self.Nphi
        self.dtheta = np.pi/self.Ntheta
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
                self.surf_vec_arr[i, j, 0] = -0.5*np.cross(v11, v12)
                self.surf_vec_arr[i, j, 1] = -0.5*np.cross(v21, v22)


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

        self.inertia_tensor = np.eye(3)
        self.initial_w0 = self.R_eps.T@np.array([0, 0, 1]).T

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



def MSE_Roll_r_sim(target_img, ref_img):
    sims = np.zeros((40))
    for i in range(40):
        target_img_roll = np.roll(target_img, i, axis=0)
        sims[i] = np.sqrt(np.mean((target_img_roll - ref_img) ** 2))
    sims = sims * 100

    return np.min(sims), np.argmin(sims)

def reward(pred, target):
    lc_temp = pred * np.mean(target) / np.mean(pred)
    target_lc_temp = target
    amp = np.max(target) - np.min(target)
    loss = np.mean((80*(target_lc_temp - lc_temp)/amp)**2)  
    loss_i = 60*np.trapezoid(np.abs(target_lc_temp-lc_temp))/(100*amp)
    loss_d = np.mean((40*(np.diff(target_lc_temp)-np.diff(lc_temp)))**2)
    #loss = (loss + loss_i + loss_d)*3/10
    loss = (1.2*loss + loss_i + loss_d)*2/10
    return 100-loss, loss*1.2*2/10, loss_i*2/10, loss_d*2/10


np.random.seed(100)

mean0 = 5
k_elem = (3, 0, 2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0]/6, 2*np.pi*np.random.rand(1)[0])

ast0 = AsteroidModel(axes=(mean0, mean0, mean0), N_set=(40, 20), tilt_mode="assigned", tilt=(0, 0))
ast0.base_fitting_generator()
ast0.surf_vec_cal()
r_arr0 = ast0.pos_sph_arr[:-1, :-1, 0]

while True:
    lc0 = LightCurve(Asteroid=ast0, Keplerian_elem=k_elem, 
                        eps=(0, 0), principle_axis=False)
    lc0.orbit_coord_set(f=0, ecl_O=(0, 0), mode="assigned")

    dir_threshold = np.pi*(1/2)
    if np.cos(dir_threshold) <= np.dot(lc0.direction_cal(0)[0], lc0.direction_cal(0)[1]):
        break
    else:
        k_elem = (3, 0, 2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0]/6, 2*np.pi*np.random.rand(1)[0])
    
lc0.lc_gen(100, 1)
lc_arr0 = lc0.lc_arr


r_list = []
sim_list = []
lc_list = []
lc_label_list = []
delta0 = 1
for i, delta in enumerate(np.linspace(-delta0, +delta0, 100)):
    ast_temp = AsteroidModel(axes=(mean0, mean0+delta, mean0), N_set=(40, 20), tilt_mode="assigned", tilt=(0, 0))
    ast_temp.base_fitting_generator()
    ast_temp.surf_vec_cal()
    r_arr_temp = ast_temp.pos_sph_arr[:-1, :-1, 0]

    sim_temp, _ = MSE_Roll_r_sim(r_arr_temp, r_arr0)

    r_list.append(mean0+delta)
    sim_list.append(sim_temp)

    if np.abs(np.abs(delta) - 0.25) < 1e-2:
        lc_temp = LightCurve(Asteroid=ast_temp, Keplerian_elem=k_elem, 
                        eps=(0, 0), principle_axis=False)
        lc_temp.orbit_coord_set(f=0, ecl_O=(0, 0), mode="assigned")   
        lc_temp.lc_gen(100, 1)
        lc_list.append(lc_temp.lc_arr.copy())
        lc_label_list.append("$\delta=$"+str(int(100*delta)/100))

fig = plt.figure(figsize=(12, 4))
ax11 = fig.add_subplot(1, 2, 1)
ax12 = fig.add_subplot(1, 2, 2)

ax11.plot([mean0-delta0, mean0+delta0], [10, 10], color='gray', linestyle='dashed', alpha=0.5)
ax11.plot(r_list, sim_list)
ax11.set_xlabel("$\delta$ (radius)")
ax11.set_ylabel("MSE_Roll_Similarity")
ax11.set_title("Ellipsoid ("+str(mean0)+", "+str(mean0)+", "+str(mean0)+") VS ("+str(mean0)+", "+str(mean0)+"+$\delta$, "+str(mean0)+")")

ax12.plot(lc_arr0, label='$\delta=0.0$')
for lc, label in zip(lc_list, lc_label_list):
    ax12.plot(lc, label=label)
ax12.set_title("Lightcurves")
ax12.set_xlabel("time")
ax12.set_ylabel("flux")
ax12.legend()

print("-"*20)
reward_total, loss_0, loss_i, loss_d = reward(lc_list[0], lc_arr0)
print("Total Reward : "+str(int(100*reward_total)/100))
print("loss_0 : "+str(int(100*loss_0)/100))
print("loss_i : "+str(int(100*loss_i)/100))
print("loss_d : "+str(int(100*loss_d)/100))

plt.show()
    

