import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm

from Asteroid_Model import AsteroidModel
from LightCurve_Generator import LightCurve
from SphericalHarmonicsExpansion import R_ArrDisplay, SphericalHarmonicsExpansion



def reward(pred, target):
    lc_temp = pred * np.mean(target) / np.mean(pred)
    target_lc_temp = target
    amp = np.max(target) - np.min(target)
    loss = np.mean((80*(target_lc_temp - lc_temp)/amp)**2)  
    loss_i = 60*np.trapz(np.abs(target_lc_temp-lc_temp))/(100*amp)
    loss_d = np.mean((40*(np.diff(target_lc_temp)-np.diff(lc_temp)))**2)
    #loss = (loss + loss_i + loss_d)*3/10
    loss = (1.2*loss + loss_i + loss_d)*2/10
    return 100-loss

N_set = (40, 20)
rand_radi = (5, 4, 5)
tilt = (np.pi/6, np.pi/4)
k_elem = (3, 0, 2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0]/6, 2*np.pi*np.random.rand(1)[0])

ast_temp = AsteroidModel(axes=rand_radi, N_set=N_set, tilt_mode="assigned", tilt=tilt)
ast_temp.base_fitting_generator()
        
def lc(N_set, k_elem, cut):
    for i in range(cut):
        ast_temp.surf_vec_cal()
        while True:
            lc_temp = LightCurve(Asteroid=ast_temp, Keplerian_elem=k_elem, 
                                eps=(0, 0), principle_axis=False)
            lc_temp.orbit_coord_set(f=0, ecl_O=(0, 0), mode="assigned")

            dir_threshold = np.pi*(1/2)
            if np.cos(dir_threshold) <= np.dot(lc_temp.direction_cal(0)[0], lc_temp.direction_cal(0)[1]):
                break
            else:
                k_elem = (3, 0, 2*np.pi*np.random.rand(1)[0], np.pi*np.random.rand(1)[0]/6, 2*np.pi*np.random.rand(1)[0])
                print("retry:"+str(N_set))

        lc_temp.lc_gen(100, 1)
        #plt.plot(lc_temp.lc_arr, label=str(N_set))
        lc_list.append(lc_temp.lc_arr.copy())
        
        for i in range(40):
            ast_temp.cut_ast(1, 0, pos_sph=[np.random.rand(1)[0], np.random.rand(1)[0], 0.5*np.random.rand(1)[0], 0.5*np.random.rand(1)[0]], assigned=True, mode="ratio_assign")

    return lc_temp.lc_arr, k_elem

N_set_list = [(40, 20)]
lc_list = []
for N_set in N_set_list:
    lc_arr, k_elem = lc(N_set, k_elem, 2)

#plt.legend()
#plt.show()

#test_SHE = SphericalHarmonicsExpansion(Asteroid=ast_temp, LightCurve=None, l_range=8)
#test_SHE.coef_arr = test_SHE.SHE_coef()
#test_SHE.Ast_SHE_display()
#ast_temp.ast_display()

rand_radi_noise = tuple(r+0.2*np.random.rand(1)[0] for r in rand_radi)
tilt_noise = tuple(r+np.pi*0.1*np.random.rand(1)[0] for r in tilt)
print(rand_radi_noise, tilt_noise)
ell_temp = AsteroidModel(axes=rand_radi_noise, N_set=N_set, tilt_mode="assigned", tilt=tilt_noise)
ell_temp.base_fitting_generator()
_ = R_ArrDisplay(y0=ast_temp.pos_sph_arr[:-1, :-1, 0].reshape(-1)*0.3, pred=ell_temp.pos_sph_arr[:-1, :-1, 0].reshape(-1)*0.3, N_set=(40, 20))

ell_temp.surf_vec_cal()
ell_lc_temp = LightCurve(Asteroid=ell_temp, Keplerian_elem=k_elem, 
                                eps=(0, 0), principle_axis=False)
ell_lc_temp.orbit_coord_set(f=0, ecl_O=(0, 0), mode="assigned")
ell_lc_temp.lc_gen(100, 1)

plt.plot(lc_list[0], label='ellipsoid0')
plt.plot(lc_list[1], label='asteroid')
plt.plot(ell_lc_temp.lc_arr, label='approximation')
plt.legend()
reward_approx = reward(lc_list[1], ell_lc_temp.lc_arr)
plt.title("reward = "+str(int(reward_approx*1000)/1000))
plt.show()

#acc_arr = np.zeros((len(lc_list), len(lc_list)))
#for i in range(len(lc_list)):
#    for j in range(len(lc_list)):
#        acc_arr[i, j] = reward(lc_list[i], lc_list[j])
#plt.imshow(acc_arr)
#plt.colorbar()
#plt.show()
