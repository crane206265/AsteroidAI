import numpy as np
from numpy import linalg as LA


class CutterPlane():
    def __init__(self, ast, random = True, mode = 'angle_assign', **kwargs):
        self.phi = kwargs['phi']
        self.theta = kwargs['theta']
        self.dist = kwargs['distance'] #distance from ast_r
        self.ast = ast

        self.j_cen = round(self.theta_cen/ast.dtheta)
        if self.j_cen%2 == 0:
            self.i_cen = round(self.phi_cen/ast.dphi)
        else:
            self.i_cen = round((self.phi_cen-ast.dphi/2)/ast.dphi)
        self.r_ast = ast.pos_sph_arr[self.i_cen, self.j_cen, 0]

    


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
