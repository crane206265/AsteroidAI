import numpy as np

class CutterSphere():
    def __init__(self, ast, random = True, *args):
        """
        initialize
        - if random == True, use random parameters
        - if random == False, use parameters from *args
        *args = (R, x1, y1, z1)
        ast : Asteroid_Model#class
        """
        if random == False:
            self.radi = args[0]
            self.x1 = args[1]
            self.y1 = args[2]
            self.z1 = args[3]
            return
        
        self.k = 0.2 #cut ratio 0.2
        self.min_cen = 3 #3
        self.max_cen = 10

        self.phi_cen = 2*np.pi*np.random.rand()
        self.theta_cen = np.pi*np.random.rand()
        self.j_cen = round(self.theta_cen/ast.dtheta)
        if self.j_cen%2 == 0:
            self.i_cen = round(self.phi_cen/ast.dphi)
        else:
            self.i_cen = round((self.phi_cen-ast.dphi/2)/ast.dphi)

        self.r_ast = ast.pos_sph_arr[self.i_cen, self.j_cen, 0]
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