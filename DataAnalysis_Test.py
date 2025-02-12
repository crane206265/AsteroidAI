import numpy as np
import torch
import matplotlib.pyplot as plt
from DataPreprocessing import DataPreProcessing

total_data = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_total.npz")
Y_total = torch.tensor(total_data['Y_total'].astype(np.complex64))
Y_total = torch.view_as_real(Y_total).type(torch.float32)
Y_total = torch.flatten(Y_total, 1)
Y_arr = DataPreProcessing.coef_zip(Y_total[:, :-3]).numpy()

repeat1 = 100

def mean_std_anlysis(exclude0 = True, analysis_ratio = 0.3, Y_arr=Y_arr):
    start_idx = 1 if exclude0 else 0
    for r in range(repeat1):
        shuffle_idx = np.arange(0, Y_total.shape[0])
        np.random.shuffle(shuffle_idx)
        Y_arr = Y_arr[shuffle_idx, :]
        Y_train0 = Y_arr[:int(analysis_ratio*Y_total.shape[0])]

        #Y_train0 = np.log(Y_train0)

        mean_arr = np.array([])
        std_arr = np.array([])

        param = 81
        for i in range(start_idx, param):
            mean = np.mean(Y_train0[:, i])
            std = np.std(Y_train0[:, i])
            mean_arr = np.append(mean_arr, mean)
            std_arr = np.append(std_arr, std)

        if r == 0:
            plt.plot(mean_arr, color='cornflowerblue', label='mean')
            plt.plot(std_arr, color='lightcoral', label='std')
        else:
            plt.plot(mean_arr, color='cornflowerblue')
            plt.plot(std_arr, color='lightcoral')
    plt.legend()
    plt.show()


repeat2 = 10

def log_anlysis(rescale, exclude0 = True, Y_arr=Y_arr):
    start_idx = 1 if exclude0 else 0
    for r in range(repeat2):
        shuffle_idx = np.arange(0, Y_total.shape[0])
        np.random.shuffle(shuffle_idx)
        Y_arr = Y_arr[shuffle_idx, :]
        Y_sample = 32*Y_arr[0]

        log_arr = np.array([])
        minus_log_idx = []

        param = 81
        for i in range(start_idx, param):
            if Y_sample[i] > 0:
                log_arr = np.append(log_arr, np.log10(Y_sample[i]))
            else:
                log_arr = np.append(log_arr, np.log10(-Y_sample[i]))
                minus_log_idx.append(i-1)

        plt.subplot(211)
        plt.plot(log_arr, color='cornflowerblue', linestyle='solid')
        plt.plot(minus_log_idx, log_arr[minus_log_idx], color='lightcoral', marker='.', linestyle='None')
        
    for rescale_val in rescale:
        rescale_arr = np.array([])
        for i in range(start_idx, param):
            if Y_sample[i] > 0:
                rescale_arr = np.append(rescale_arr, rescale_val**(np.log10(Y_sample[i])))
            else:
                rescale_arr = np.append(rescale_arr, -rescale_val**(np.log10(-Y_sample[i])))

        plt.subplot(212)
        if rescale_val == 10:
            plt.plot(rescale_arr, label=str(rescale_val), linestyle='dotted')
        else:
            plt.plot(rescale_arr, label=str(rescale_val))
        plt.ylim(-15, 30)
        plt.legend()

    plt.show()


log_anlysis(rescale=[2, 2.5, 3, 3.3, 10],exclude0=False, Y_arr=Y_arr)
#mean_std_anlysis(exclude0=True, analysis_ratio=0.3, Y_arr=Y_arr)


