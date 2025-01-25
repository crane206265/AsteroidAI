import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

from Asteroid_Model import AsteroidModel

#np.random.seed(1)

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
        self.initial_w0 = 4*self.R_eps.T@np.array([0, 0, 1]).T
        if principle_axis:
            self.principle_axis(1)
            initial_eps = np.empty(2)
            initial_eps[0] = np.arctan2(self.initial_w0[1], self.initial_w0[0])
            initial_eps[1] = np.arccos(self.initial_w0[2]/LA.norm(self.initial_w0))
            self.R_eps = self.rotArr(initial_eps[0], "z")@self.rotArr(initial_eps[1], "y")
    

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
        dt = 1/rot_div_N
        w0 = self.initial_w0
        rot_angle = 0
        for i in range(int(rot_div_N*len_ratio)):    
            self.lc_arr[i] = np.sum(self.flux(1, self.direction_cal(rot_angle)))
            w0, rot_angle = self.precession(w0, self.inertia_tensor, 0, rot_angle, dt, "No_Precession")

    # Overiding animation function
    #   - to implement tilted rotation
    def anim_frame(self, frame, rot_div_N, len_ratio, ax1, ax2):
        """
        animation frame : display
        * including lightcurve
        """
        dt = 1/rot_div_N
        R_eps = self.R_eps_list[frame]
        gridX = np.zeros((self.Asteroid.Nphi+1, self.Asteroid.Ntheta+1))
        gridY = np.zeros((self.Asteroid.Nphi+1, self.Asteroid.Ntheta+1))
        gridZ = np.zeros((self.Asteroid.Nphi+1, self.Asteroid.Ntheta+1))
        for i in range(self.Asteroid.Nphi+1):
            for j in range(self.Asteroid.Ntheta+1):
                pos_cart_temp = R_eps.T@self.rotArr(self.rot_angle_list[frame], "z")@R_eps@self.Asteroid.pos_cart_arr[i, j]
                pos_sph_temp = self.Asteroid.cart2sph(pos_cart_temp)
                gridX[i, j] = pos_cart_temp[0]
                gridY[i, j] = pos_cart_temp[1]
                gridZ[i, j] = pos_cart_temp[2]

        ax1.clear()
        ax1.set_box_aspect((1, 1, 1))
        lim_set = (-10, 10)
        ax1.set_xlim(lim_set)
        ax1.set_xlabel('X')
        ax1.set_ylim(lim_set)
        ax1.set_ylabel('Y')
        ax1.set_zlim(lim_set)
        ax1.set_zlabel('Z')
        ax1.set_title("Asteroid Model")

        ax2.clear()
        ax2.set_title("Light Curve")
        lc_time = np.arange(int(rot_div_N*len_ratio))

        rot_axis_plot = 2*lim_set[0]*np.array([R_eps@np.array([0, 0, 1]).T, -R_eps@np.array([0, 0, 1]).T]).T
        sun_dir_plot, earth_dir_plot = self.direction_cal(0)
        sun_dir_plot = lim_set[0]*np.array([sun_dir_plot, np.array([0, 0, 0])]).T
        earth_dir_plot = lim_set[0]*np.array([earth_dir_plot, np.array([0, 0, 0])]).T

        model_drawer = [ax1, ax2]
        model_drawer[0].scatter(gridX, gridY, gridZ) #plot_surface
        #model_drawer[0].plot([lim_set[1], lim_set[1]-3], [0, 0], [lim_set[0], lim_set[0]])
        model_drawer[0].plot(rot_axis_plot[0], rot_axis_plot[1], rot_axis_plot[2], color='gold', linestyle='solid') #plot rotational axis
        model_drawer[0].plot(sun_dir_plot[0], sun_dir_plot[1], sun_dir_plot[2], color='orange', linestyle='solid') #plot sun direction
        model_drawer[0].plot(earth_dir_plot[0], earth_dir_plot[1], earth_dir_plot[2], color='green', linestyle='solid') #plot earth direction
        model_drawer[1].plot(lc_time, self.lc_arr, 'b-')
        model_drawer[1].plot(frame, self.lc_arr[frame], 'r.')

        return model_drawer

    def rotate_anim(self, rot_div_N, len_ratio):
        """
        play rotating animation
        * including lightcurve
        """
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)

        #plot_reset = np.zeros((self.Nphi+1, self.Ntheta+1))
        #model_drawer = ax1.plot_surface(plot_reset, plot_reset, plot_reset)

        ani = animation.FuncAnimation(fig, self.anim_frame, fargs=(rot_div_N, len_ratio, ax1, ax2), frames=int(rot_div_N*len_ratio))
        plt.show()