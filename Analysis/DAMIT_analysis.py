import numpy as np
import matplotlib.pyplot as plt

import tarfile


file_path = "C:/Users/dlgkr/Downloads/damit-20200211T010301Z.tar.gz"

def data_extract(tar:tarfile.TarFile, lc_path, spin_path):
    # tar : TarFile objects of tar.gzip file
    # lc_path : lc file name
    # spin_path : spin file name
    # 
    # return : time_arr list, intensity_arr list, period(unit of days)  

    spin_member = tar.getmember(spin_path)
    spin_f = tar.extractfile(spin_member)
    spin_content = spin_f.read().decode('utf-8')
    spin_rows = spin_content.split("\n")

    period = float(spin_rows[0].split(' ')[-1])

    lc_member = tar.getmember(lc_path)
    lc_f = tar.extractfile(lc_member)
    lc_content = lc_f.read().decode('utf-8')
    lc_rows = lc_content.split("\n")
    
    num_lc = int(lc_rows[0])
    t_arr_list = []
    intensity_arr_list = []
    _ = lc_rows.pop(0) # delete the first row

    print("Total # of LCs : %02d"%(num_lc))
    for idx in range(num_lc):
        lc_point_num, mode = lc_rows[0].split(" ")
        lc_point_num = int(lc_point_num)
        mode = int(mode)
        _ = lc_rows.pop(0)

        t_arr = np.array([])
        intensity_arr = np.array([])
        for i in range(lc_point_num):
            t_arr = np.append(t_arr, float(lc_rows[0].split(" ")[0]))
            for j in range(1, 10):
                if lc_rows[0].split(" ")[j] != '':
                    intensity_arr = np.append(intensity_arr, float(lc_rows[0].split(" ")[j]))
                    break
            _ = lc_rows.pop(0)

        t_arr_list.append(t_arr)
        intensity_arr_list.append(intensity_arr)

    return t_arr_list, intensity_arr_list, period/24

# ---------- main ----------

with tarfile.open(file_path, 'r:gz') as tar:
    total_files = tar.getnames() # type : list
    lc_txts0 = [name for name in total_files if name.endswith('lc.txt')]
    spin_txts0 = [name for name in total_files if name.endswith('spin.txt') and name.count('IAU') == 0] #all spin.txt files
    
    asts_idx_list = [int(name.split('/')[2].removeprefix('asteroid_')) for name in lc_txts0]

    # filtering lc_txts if they have corr spin.txt files
    spin_txts = []
    spin_txts_include = []
    for name in spin_txts0:
        ast_idx = int(name.split('/')[2].removeprefix('asteroid_'))
        if ast_idx in asts_idx_list and ast_idx not in spin_txts_include:
            spin_txts.append(name)
            spin_txts_include.append(ast_idx)

    lc_txts = []
    for ast_idx in spin_txts_include:
        lc_txts = lc_txts + [name for name in lc_txts0 if ('asteroid_'+str(ast_idx)+'/') in name]

    # sorting
    lc_txts.sort(key=lambda name: int(name.split('/')[2].removeprefix('asteroid_')))
    spin_txts.sort(key=lambda name: int(name.split('/')[2].removeprefix('asteroid_'))) 
    

    # print all routes of lc
    check_all = False
    if check_all:
        for i, (lc_name, spin_name) in enumerate(zip(lc_txts, spin_txts)):
            print("%04d | "%(i) + str(lc_name) +" | " + str(spin_name))

        plt.hist(asts_idx_list, bins=len(lc_txts))
        plt.title("# of lc.txt Files for each Asteroids")
        plt.show()

    t_arr_list, intensity_arr_list, period = data_extract(tar, lc_txts[0], spin_txts[0])

    for t_arr, intensity_arr in zip(t_arr_list, intensity_arr_list):
        print("Period : %.5f (hour) / %.5f (days)"%(period*24, period))
        print("Time Range : [%.5f, %.5f] --> dt = %.5f (%.5f P)"%(t_arr[0], t_arr[-1], t_arr[-1]-t_arr[0], (t_arr[-1]-t_arr[0])/period))
        plt.plot(t_arr, intensity_arr, marker='.', color='royalblue', linestyle='none')
        ylims = plt.ylim()
        plt.plot([t_arr[0]]*2, ylims, linestyle='dotted', color='orangered', alpha=0.7)
        plt.plot([t_arr[0]+period]*2, ylims, linestyle='dotted', color='orangered', alpha=0.7)
        plt.show()


    

raise NotImplementedError

lc_arr = data[i*800+1, 800:900]
fft_coef_zip = np.abs(np.fft.fft(lc_arr))[:lc_arr.shape[0]//2+1]
fft_coef_zip = np.log10(fft_coef_zip)
log_thr = np.log10(4)#4
if np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:]):
    filtered_data_temp = np.concatenate((filtered_data_temp, data[i*800+1:(i+1)*800+1, :]), axis=0)
