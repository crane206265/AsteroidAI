import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt
from matplotlib import animation
from tqdm import tqdm

from Model_Cutter import CutterSphere


class AsteroidModel():
    def __init__(self, axes, N_set, tilt_mode="assigned", tilt=(0, 0), coord2discrete_param=(0.5, 10)):
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



        prec, bound = self.discr_param
        N_index = 2*int(bound/prec) + 1
        self.is_interior = np.zeros((N_index, N_index, N_index))

        self.inertia_tensor = np.eye(3)

        self.max_r = 0
        self.valid_lw_bd = 0
        self.valid_up_bd = N_index

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
    def sph2cart(self, sph_coord):
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
    
    def cart2sph(self, cart_coord):
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
        if LA.norm(np.array([x, y])) == 0:
            if z >= 0:
                theta = np.pi/2
            else:
                theta = -np.pi/2
        else:
            theta = np.arctan(z/((x*x + y*y)**0.5))
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
                self.pos_cart_arr[i, j] = self.sph2cart(self.pos_sph_arr[i, j])
        self.__circular('all')
        self.wobble_r(epoch-1, ratio)
    
    def cut_ast(self, sph_num, pla_num):
        """
        cut asteroid with specific shape

        sph_num : cutting spherical num
        pla_num : cutting plane num
        """
        self.__sph_cut(sph_num)

    def __sph_cut(self, sph_num):
        """
        cutting with sphere - CutterSphere#class
        """
        for k in range(sph_num):
            sph_temp = CutterSphere(ast = self, random=True)
            for i in range(self.Nphi+1):
                for j in range(self.Ntheta+1):
                    if sph_temp.f(self.pos_cart_arr[i, j]) < 0:
                        self.pos_sph_arr[i, j, 0] = sph_temp.r_f(self.pos_sph_arr[i, j, 1:])
                        self.pos_cart_arr[i, j] = self.sph2cart(self.pos_sph_arr[i, j])
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
        self.COM_vec = np.zeros((3))
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
                self.COM_vec[component] = np.sum(integrand*(prec**3))/np.sum(self.is_interior*(prec**3))
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


    def ast_display(self):
        """
        stationary display
        * not including lightcurve
        """
        gridX = self.pos_cart_arr[:, :, 0]
        gridY = self.pos_cart_arr[:, :, 1]
        gridZ = self.pos_cart_arr[:, :, 2]
        #print(np.shape(gridX))
        #print(gridX)
        fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={"projection":"3d"})
        ax.set_box_aspect((1, 1, 1))
        lim_set = (-10, 10)
        ax.set_xlim(lim_set)
        ax.set_xlabel('X')
        ax.set_ylim(lim_set)
        ax.set_ylabel('Y')
        ax.set_zlim(lim_set)
        ax.set_zlabel('Z')
        ax.set_title("Asteroid Model")

        ax.plot_surface(gridX, gridY, gridZ)
        plt.show()

    # animation / displaying functions
    # more detailed implement is at LightCurve Class
        # : tilted rotation, ...
        # not use anim_frame(), rotation_anim() function

    def anim_frame(self, frame, rot_div_N, ax1, ax2):
        """
        animation frame : display
        * including lightcurve
        """
        gridX = np.zeros((self.Nphi+1, self.Ntheta+1))
        gridY = np.zeros((self.Nphi+1, self.Ntheta+1))
        gridZ = np.zeros((self.Nphi+1, self.Ntheta+1))
        for i in range(self.Nphi+1):
            for j in range(self.Ntheta+1):
                pos_rot_temp = self.pos_sph_arr[i, j] + np.array([0, 2*np.pi*frame/rot_div_N, 0])
                pos_cart_temp = self.sph2cart(pos_rot_temp)
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
        lc_time = np.arange(rot_div_N)

        model_drawer = [ax1, ax2]
        model_drawer[0].plot_surface(gridX, gridY, gridZ)
        model_drawer[0].plot([lim_set[1], lim_set[1]-3], [0, 0], [lim_set[0], lim_set[0]])
        model_drawer[1].plot(lc_time, self.lc_arr, 'b-')
        model_drawer[1].plot(frame, self.lc_arr[frame], 'r.')

        return model_drawer

    def rotate_anim(self, rot_div_N):
        """
        play rotating animation
        * including lightcurve
        """
        fig = plt.figure(figsize=(10, 5))
        ax1 = fig.add_subplot(1, 2, 1, projection='3d')
        ax2 = fig.add_subplot(1, 2, 2)

        #plot_reset = np.zeros((self.Nphi+1, self.Ntheta+1))
        #model_drawer = ax1.plot_surface(plot_reset, plot_reset, plot_reset)

        ani = animation.FuncAnimation(fig, self.anim_frame, fargs=(rot_div_N, ax1, ax2), frames=rot_div_N)
        plt.show()


"""
    def is_interior_cal(self, param):
        
        #is_interior function : 0 if the coordinate is not interior of the asteroid
        #                     : 1 else
        #prec : "unit length"
        #bound : domain range
        #coord. x -> index : (x + bound)/prec
        prec, bound = param
        N_index = 2*int(bound/prec) + 1
        self.is_interior = np.zeros((N_index, N_index, N_index))

        surf_coef_arr = np.zeros((self.Nphi, self.Ntheta, 2, 3))
        surf_origin_dist = np.zeros((self.Nphi, self.Ntheta, 2))
        for i in range(self.Nphi):
            for j in range(self.Ntheta):
                for k in range(2):
                    if j%2 == 0:
                        if k == 0:
                            p_arr = np.array([self.pos_cart_arr[i+1, j],
                                              self.pos_cart_arr[i, j+1],
                                              self.pos_cart_arr[i+1, j+1]])
                        if k == 1:
                            p_arr = np.array([self.pos_cart_arr[i, j],
                                              self.pos_cart_arr[i+1, j],
                                              self.pos_cart_arr[i, j+1]])
                    if j%2 == 1:
                        if k == 0:
                            p_arr = np.array([self.pos_cart_arr[i, j],
                                              self.pos_cart_arr[i, j+1],
                                              self.pos_cart_arr[i+1, j+1]])
                        if k == 1:
                            p_arr = np.array([self.pos_cart_arr[i, j],
                                              self.pos_cart_arr[i+1, j],
                                              self.pos_cart_arr[i+1, j+1]])
                    if LA.det(p_arr) != 0:
                        surf_coef_arr[i, j, k] = LA.inv(p_arr)@np.array([1, 1, 1]).T
                        surf_origin_dist[i, j, k] = -1 / np.sqrt(surf_coef_arr[i, j, k].T@surf_coef_arr[i, j, k])

        for x_idx in tqdm(range(self.is_interior.shape[0])):
            for y_idx in range(self.is_interior.shape[1]):
                for z_idx in range(self.is_interior.shape[2]):
                    x, y, z = self.__is_interior_transform_arr2coord((x_idx, y_idx, z_idx), self.is_interior_param)
                    phi = self.cart2sph((x, y, z))[1]
                    theta = self.cart2sph((x, y, z))[2]
                    j = int(theta/self.dtheta)
                    i = int((phi - (j%2)*self.dphi/2)/self.dphi)
                    interior = False
                    for k in range(2):
                        if LA.norm(surf_coef_arr[i, j, k]) == 0 or surf_origin_dist[i, j, k] == 0:
                            continue
                        point_plane_dist = (np.array([x, y, z])@surf_coef_arr[i, j, k] - 1) / np.sqrt(surf_coef_arr[i, j, k].T@surf_coef_arr[i, j, k])
                        if point_plane_dist * surf_origin_dist[i, j, k] >= 0 and abs(point_plane_dist) <= abs(surf_origin_dist[i, j, k]):
                        #if point_plane_dist <= -5:
                            interior = interior and True
                    if interior or True:
                        self.is_interior[x_idx, y_idx, z_idx] = surf_origin_dist[i, j, k]
"""