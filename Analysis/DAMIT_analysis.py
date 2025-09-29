import numpy as np
import matplotlib.pyplot as plt

import tarfile


file_path = "C:/Users/dlgkr/Downloads/damit-20200211T010301Z.tar.gz"


with tarfile.open(file_path, 'r:gz') as tar:
    total_files = tar.getnames() # type : list
    lc_txts = [name for name in total_files if name.endswith('lc.txt')]
    
    asts_idx_list = [int(name.split('/')[-2].removeprefix('asteroid_')) for name in lc_txts]

    # print all routes of lc
    check_all = False
    if check_all:
        for i, name in enumerate(lc_txts):
            print("%04d | "%(i) + str(name))

        plt.hist(asts_idx_list, bins=len(lc_txts))
        plt.title("# of lc.txt Files for each Asteroids")
        plt.show()

    member = tar.getmember(lc_txts[0])
    f = tar.extractfile(member)
    content = f.read().decode('utf-8')
    rows = content.split("\n")
    
    num_lc = int(rows[0])
    t_arr_list = []
    intensity_arr_list = []
    _ = rows.pop(0) # delete the first row

    print("Total # of LCs : %02d"%(num_lc))
    for idx in range(num_lc):
        lc_point_num, mode = rows[0].split(" ")
        lc_point_num = int(lc_point_num)
        mode = int(mode)
        _ = rows.pop(0)

        t_arr = np.array([])
        intensity_arr = np.array([])
        for i in range(lc_point_num):
            t_arr = np.append(t_arr, float(rows[0].split(" ")[0]))
            for j in range(1, 10):
                if rows[0].split(" ")[j] != '':
                    intensity_arr = np.append(intensity_arr, float(rows[0].split(" ")[j]))
                    break
            _ = rows.pop(0)

        t_arr_list.append(t_arr)
        intensity_arr_list.append(intensity_arr)

        print("Time Range : [%.5f, %.5f] --> dt = %.5f"%(t_arr[0], t_arr[-1], t_arr[-1]-t_arr[0]))
        plt.plot(t_arr, intensity_arr, 'b.')
        plt.show()


    

raise NotImplementedError

lc_arr = data[i*800+1, 800:900]
fft_coef_zip = np.abs(np.fft.fft(lc_arr))[:lc_arr.shape[0]//2+1]
fft_coef_zip = np.log10(fft_coef_zip)
log_thr = np.log10(4)#4
if np.all(fft_coef_zip[2] - log_thr >= fft_coef_zip[3:]):
    filtered_data_temp = np.concatenate((filtered_data_temp, data[i*800+1:(i+1)*800+1, :]), axis=0)