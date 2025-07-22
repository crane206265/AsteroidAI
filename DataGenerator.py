import numpy as np
import numpy.linalg as LA
from tqdm import tqdm
import os
import gc

from Asteroid_Model import AsteroidModel
from LightCurve_Generator import LightCurve
from SphericalHarmonicsExpansion import SphericalHarmonicsExpansion

#DATA GEN
def data_gen(file_num, ast_num = 50, lc_num = 10, lc_len = 200, coef_l = 8, N = (40, 20), transfer = False):

    lc_arr = np.zeros((ast_num, lc_num, lc_len))
    lc_info = np.zeros((ast_num, lc_num, 2, 3))
    coef_arr = np.zeros((ast_num, (coef_l+1)**2), dtype="complex_")
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

start_file_num = 0 #normal
end_file_num = 100 #normal

transfer = False
#start_file_num = 57 #transfer
#end_file_num = 200 #transfer

for file_num in range(start_file_num, end_file_num+1):
    data_gen(file_num, ast_num = 50, lc_num = 20, lc_len = 200, coef_l = 8, N = (40, 20), transfer=transfer)
    continue

data_concatenate(data_folder_path=data_folder_path)
