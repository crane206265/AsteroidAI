import numpy as np
import matplotlib.pyplot as plt

import gc

data0 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_0.npy")
data1 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_1.npy")
data_train = np.concatenate((data0[1:], data1[1:]), axis=0)
del data0, data1
gc.collect()
data2 = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/pole_axis_RL_data_batches/data_pole_axis_RL_preset_batch_2.npy")[1:]


train_reward = data_train[:, -1].copy()
test_reward = data2[:, -1].copy()
del data2, data_train
gc.collect()


def reward_distn(bins=20):
    fig = plt.figure(figsize=(12, 7))
    ax11 = fig.add_subplot(2, 2, 1)
    ax12 = fig.add_subplot(2, 2, 2)
    ax21 = fig.add_subplot(2, 2, 3)
    ax22 = fig.add_subplot(2, 2, 4)

    ax11.hist(train_reward, bins=bins, log=False)
    ax12.hist(train_reward, bins=bins, log=True)
    ax21.hist(test_reward, bins=bins, log=False)
    ax22.hist(test_reward, bins=bins, log=True)

    ax11.set_title("$\Delta$Reward Distn. (Train Set, Linear Scale)")
    ax12.set_title("$\Delta$Reward Distn. (Train Set, Logarithm Scale)")
    ax21.set_title("$\Delta$Reward Distn. (Test Set, Linear Scale)")
    ax22.set_title("$\Delta$Reward Distn. (Test Set, Logarithm Scale)")

    plt.show()

def scaling_distn(bins=20, logscale=True):
    fig = plt.figure(figsize=(12, 7))
    ax11 = fig.add_subplot(2, 3, 1)
    ax12 = fig.add_subplot(2, 3, 2)
    ax13 = fig.add_subplot(2, 3, 3)
    ax21 = fig.add_subplot(2, 3, 4)
    ax22 = fig.add_subplot(2, 3, 5)
    ax23 = fig.add_subplot(2, 3, 6)

    ax11.hist(train_reward, bins=bins, log=logscale)
    ax12.hist(sc1(train_reward), bins=bins, log=logscale)
    ax13.hist(sc2(train_reward), bins=bins, log=logscale)
    ax21.hist(sc3(train_reward), bins=bins, log=logscale)
    ax22.hist(sc4(train_reward), bins=bins, log=logscale)
    ax23.hist(sc5(train_reward), bins=bins, log=logscale)

    ax11.set_title("$\Delta$Reward Distn. (No Scaling)")
    ax12.set_title("$\Delta$Reward Distn. (Scaling 1)")
    ax13.set_title("$\Delta$Reward Distn. (Scaling 2)")
    ax21.set_title("$\Delta$Reward Distn. (Scaling 3)")
    ax22.set_title("$\Delta$Reward Distn. (Scaling 4)")
    ax23.set_title("$\Delta$Reward Distn. (Scaling 5)")

    plt.show()

def sc1(x): return 6 * (2/np.pi) * np.arctan(x/2)
def sc2(x): return 6 * (2/np.pi) * np.arctan(x/8)
def sc3(x): return 7 * 2*(1/(1+np.exp(x/4)) - 0.5)
def sc4(x): return 7 * 2*(1/(1+np.exp(x/7)) - 0.5)
def sc5(x): return (x - np.quantile(x, 0.50)) / (np.quantile(x, 0.75) - np.quantile(x, 0.25))



scaling_distn(bins=30, logscale=False)


