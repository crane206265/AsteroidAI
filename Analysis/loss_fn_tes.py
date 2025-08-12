import numpy as np
import matplotlib.pyplot as plt
import cv2

def sample_generator(roll=0):
    res = np.zeros((20, 40))
    w = 2*np.random.rand(1)[0] + 3
    idx_w = np.random.randint(0, 5)
    h = 5*np.random.rand(4) + 10
    idx_h = np.random.randint(0, 5, 4)

    res[idx_h[0]:idx_h[0]+int(h[0]), idx_w   :idx_w+int(w)   ] = 2*np.random.rand(int(h[0]), int(w)) + 1
    res[idx_h[1]:idx_h[1]+int(h[1]):, idx_w+20:idx_w+int(w)+20] = 2*np.random.rand(int(h[1]), int(w)) + 1
    res[idx_h[2]:idx_h[2]+int(h[2]):, idx_w+10:idx_w+int(w)+10] = -2*np.random.rand(int(h[2]), int(w)) - 1
    res[idx_h[3]:idx_h[3]+int(h[3]):, idx_w+30:idx_w+int(w)+30] = -2*np.random.rand(int(h[3]), int(w)) - 1

    res = res + (np.random.randn(20, 40)-0.5) * 0.7

    res = cv2.GaussianBlur(res, (3, 3), 2)
    res = 1.3 * res

    res = np.roll(res, roll, axis=1)

    return res

def plotter(res0, res, loss_fn):
    fig = plt.figure(figsize=(12, 4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    res0_img = ax1.imshow(res0, vmax=6, vmin=-6)
    ax1.set_title("Reward Map Sample 0")
    plt.colorbar(res0_img, ax=ax1, fraction=0.026, pad=0.04)
    
    res_img = ax2.imshow(res, vmax=6, vmin=-6)
    loss = loss_fn(res, res0)
    ax2.set_title("Reward Map Sample (Loss = "+str(int(1000*loss)/1000)+")")
    plt.colorbar(res_img, ax=ax2, fraction=0.026, pad=0.04)

    proc_res = processer(res)
    proc_res_img = ax3.imshow(proc_res, vmax=6, vmin=-6)
    loss = loss_fn(proc_res, res0)
    ax3.set_title("Reward Map Sample (Loss = "+str(int(1000*loss)/1000)+")")
    plt.colorbar(proc_res_img, ax=ax3, fraction=0.026, pad=0.04)

    plt.show()

def rolling_plotter(res0, res, loss_fn_list):
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    for axis in [0, 1]:
        loss_map = np.zeros((len(loss_fn_list), 20+20*axis))
        for j in range(20+20*axis):
            res = np.roll(res, 1, axis=axis)
            for i, loss_fn in enumerate(loss_fn_list):
                loss_map[i, j] = loss_fn(res0, res)

        if axis == 0:
            loss_map1 = ax1.imshow(loss_map)
            ax1.set_title("Loss Map (Vertical Rolling)")
            ax1.set_xlabel("Rolling Steps")
            ax1.set_ylabel("Loss Value")
            plt.colorbar(loss_map1, ax=ax1, fraction=0.026, pad=0.04)
        elif axis == 1:
            loss_map2 = ax2.imshow(loss_map)
            ax2.set_title("Loss Map (Horizontal Rolling)")
            ax2.set_xlabel("Rolling Steps")
            ax2.set_ylabel("Loss Value")
            plt.colorbar(loss_map2, ax=ax2, fraction=0.026, pad=0.04)
    plt.show()

def loss1(x, y):
    loss = np.sqrt(np.mean((x - y) ** 2))
    return loss

def loss2(x, y):
    weight = 0.5 + 0.5 * np.abs(y)
    loss = np.sqrt(np.sum((x - y) ** 2 * weight)/np.sum(weight))
    return loss

def loss3(x, y):
    # vertical pressing
    x_line = np.mean(x, axis=0)
    y_line = np.mean(y, axis=0)
    loss = np.sqrt(np.mean((x_line - y_line) ** 2))
    return loss

def loss4(x, y):
    # propagation of positive values
    hori_prop = 1
    vert_prop = 1
    x_prop = processer(x, propagation=(hori_prop, vert_prop))
    y_prop = processer(y, propagation=(hori_prop, vert_prop))

    loss = np.sqrt(np.mean((x_prop - y_prop) ** 2))

    return loss

def loss5(x, y):
    # propagation of positive values with different exponent
    hori_prop = 1
    vert_prop = 1
    x_prop = processer(x, propagation=(hori_prop, vert_prop))
    y_prop = processer(y, propagation=(hori_prop, vert_prop))
    x_prop_pos = np.where(x_prop > 0, x_prop, 0)
    y_prop_pos = np.where(y_prop > 0, y_prop, 0)

    #loss4 = np.sqrt(np.mean((x_prop - y_prop) ** 2))
    loss5 = np.sqrt(np.mean((x_prop_pos - y_prop_pos) ** 2))

    #loss = 0.5*(loss)

    return loss5

def loss6(x, y): return 0.5*(loss3(x, y) + loss4(x, y))

def processer(reward_map, propagation=(3, 1)):
    hori_prop, vert_prop = propagation
    reward_map_pos = np.where(reward_map > 0, reward_map, 0)
    reward_map_neg = np.where(reward_map < 0, reward_map, 0)

    exp = 2
    div = hori_prop + vert_prop - 0.5
    reward_map_prop = reward_map_pos**exp
    for i in range(1, hori_prop+1):
        reward_map_prop[:, :-i] = reward_map_prop[:, :-i] + reward_map_pos[:, i:]**exp
        reward_map_prop[:,  i:] = reward_map_prop[:,  i:] + reward_map_pos[:, :-i]**exp
    for j in range(1, vert_prop+1):
        reward_map_prop[:-j, :] = reward_map_prop[:-j, :] + reward_map_pos[j:, :]**exp
        reward_map_prop[ j:, :] = reward_map_prop[ j:, :] + reward_map_pos[:-j, :]**exp
    reward_map_prop = (reward_map_prop / div)**(1/exp)
    reward_map_prop += reward_map_neg

    return reward_map_prop


np.random.seed(722)

loss_list = [loss1, loss2, loss3, loss4, loss5, loss6]

res0 = sample_generator()
res = sample_generator(roll=0)

for i, loss_fn in enumerate(loss_list):
    print("loss "+str(i+1)+": "+str(int(1000*loss_fn(res, res0))/1000))

plotter(res0, res, loss4)
rolling_plotter(res0, res, loss_list)


