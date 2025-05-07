import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from Environment import AstEnv
from DataPreprocessing import DataPreProcessing


# seed
seed = 722
np.random.seed(seed)
random.seed(seed)

MAX_STEPS = 700 #per episode


class Runner():
    def __init__(self, env:AstEnv, state_dim, action_dim, prec = 5.0):
        self.env = env
        self.done = True
        self.passed = False
        self.all_done = False
        self.reward_past = None
        self.prec = prec
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.data_set_arr = np.zeros((1, self.state_dim+self.action_dim+1))

    def reset(self, passed):
        self.state = self.env.reset(passed)
        self.done = False
        self.passed = False
        self.reward_past = None

    def reward_fn(self, reward, time, t_m=MAX_STEPS, t_g=int(MAX_STEPS*0.6), R_max=50, alpha=1, beta=1, epsilon=0.05):
        """
        <PARAMETERS>
        t_m : MAX_STEPS per episode(env)
        t_g : goal step (want to finish episode in)
        R_max : max reward w/o time penalty (before scaling)
        alpha : [start reward] = alpha * R_max
        beta : [time penalty until t_max] = beta * R_max
        epsilon : [time penalty until t_goal] = epsilon * R_max

        time penalty function : quadratic
        """
        modified_reward = 0#alpha * R_max
        penalty_coef = R_max * LA.inv(np.array([[t_m*(t_m+1)*(2*t_m+1), 6*t_m],
                                                [t_g*(t_g+1)*(2*t_g+1), 6*t_g]])) @ np.array([6*beta, 6*epsilon]).T
        modified_reward += -np.array([time**2, 1])@penalty_coef
        modified_reward += max(reward - self.env_max_reward_past, 0)
        modified_reward /= 1
        modified_reward *= 40/(self.env.total_threshold - self.env.reward0)
        return modified_reward

    def run(self, env_no, random=True):
        prec = self.prec
        r_cen_low = 0.2
        r_cen_high = 1
        R_cut_low = 0.40 #0.01 #0.25
        R_cut_high = 1 #0.15

        get_env_num = int(30 * (self.env.reward_threshold - self.env.reward0) / self.env.reward_threshold)
        get = 0
        reward_threshold_list = np.linspace(self.env.reward0, self.env.reward_threshold, get_env_num)
        
        max_reward = -9e+8
        show_bool = True
        for t in range(MAX_STEPS):
            if self.all_done:
                print(" SUCCESSED in CURRENT ENV")
                self.env.show()
                break

            if self.done:
                self.reset(self.passed)

            actions = np.random.uniform(-prec, prec, 4)
            
            actions[:-2] = actions[:-2] - actions[:-2]//1
            actions[-2:] = np.clip(actions[-2:], [-prec, -prec], [prec, prec])
            actions[2] = 0.1#(actions[2]+prec)*(r_cen_high-r_cen_low)/(2*prec) + r_cen_low
            actions[3] = 0.1#(actions[3]+prec)*(R_cut_high-R_cut_low)/(2*prec) + R_cut_low

            self.env_max_reward_past = self.env.max_reward + 0
            next_state, reward, self.done, info = self.env.step(actions)
            if self.reward_past == None:
                self.reward_past = reward
            self.passed = info[0]
            self.all_done = info[1]
            weight = 2.5

            #modified_reward = self.reward_fn(reward, t, t_m=500, t_g=300, R_max=50.0, alpha=1.0, beta=0.3, epsilon=0.03)
            if self.done and not self.passed: # if max reward - 3 condition is not satisfied 
                weight = 1.0

            self.state = next_state

            if weight != 1:
                self.reward_past = reward + 0

            
            if reward >= reward_threshold_list[get]:
                self.make_map(reward, self.state, env_no, random)
                get += 1
            
            if get == get_env_num: break
            
            max_reward = max(max_reward, reward)
            if t%4 == 0:
                print(" | Reward : {:7.5g}".format(reward), end='')
                print(" | actions : [{:6.05g}, {:6.05g}, {:6.05g}, {:6.05g}]".format(actions[0], actions[1], actions[2], actions[3]), end='')
                print(" | done/pass : "+str(self.done)+"/"+str(self.passed)+" "+str(self.env.max_reward))
                if t%20 == 0:
                    show_bool = True

            if show_bool:
                print("show_passed : "+str(reward)+" | obs_lc_num : "+str(self.env.obs_lc_num))
                #self.env.show(str(data_num)+"_"+str(et)+"_"+str(k)+"_"+str(int(reward*100)/100)+"_"+"0402ast.png")
                #plt.close()
                show_bool = False
           
            if self.done:
                if self.passed:
                    print("show_passed : "+str(reward)+" | obs_lc_num : "+str(self.env.obs_lc_num))
                    break

    def make_map(self, ref_reward, state, env_no, random):
        ratio_action_set = [(0.1, 0.1)]#, (0.3, 0.3), (0.5, 0.5), (1, 1)]
        ref_ast = self.env.ast.copy()
        rot_axis = self.env.initial_eps * 180/np.pi
        rot_axis[0] = rot_axis[0]%360
        rot_axis[1] = rot_axis[1]%180
        resol = 1

        phi_ticks = self.__map_tick_list(resol, self.env.Nphi, 360)
        theta_ticks = self.__map_tick_list(resol, self.env.Ntheta, 180)
        path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/reward_maps/"

        for ratio_actions in ratio_action_set:
            delta_map_temp = np.zeros((resol*self.env.Nphi, resol*self.env.Ntheta))
            print("\nGenerating Map... (at Reward="+str(int(ref_reward*100)/100)+")", end='')
            for idx in tqdm(range(self.env.Nphi*self.env.Ntheta*resol*resol)):
                i = idx//int(resol*self.env.Ntheta)
                j = idx%int(resol*self.env.Ntheta)

                if random:
                    phi_action = (i/(resol*self.env.Nphi) + np.random.normal(0, 0.05, 1)[0])%1
                    theta_action = (j/(resol*self.env.Ntheta) + np.random.normal(0, 0.05, 1)[0])%1
                else:
                    phi_action = (i/(resol*self.env.Nphi))%1
                    theta_action = (j/(resol*self.env.Ntheta))%1
                actions = np.array([phi_action, theta_action, ratio_actions[0], ratio_actions[1]])
                
                _, reward, _, _ = self.env.step(actions, update=False)
                delta_map_temp[i, j] = reward - ref_reward
                self.data_set_arr = np.concatenate((self.data_set_arr, np.array([np.concatenate((self.state, actions, np.array([delta_map_temp[i, j]])))])), axis=0)
                
                self.env.ast = ref_ast.copy()

            delta_map_temp = delta_map_temp.T
            reward_map_temp = delta_map_temp + ref_reward

            grady, gradx = np.gradient(delta_map_temp)
            x = np.arange(delta_map_temp.shape[1])
            y = np.arange(delta_map_temp.shape[0])
            X, Y = np.meshgrid(x, y)
            
            """
            plt.figure(figsize=(20, 11))
            plt.imshow(delta_map_temp)
            plt.plot(rot_axis[0]*resol*self.env.Nphi/360, rot_axis[1]*resol*self.env.Ntheta/180, color='red', marker='X', markersize=10)
            plt.plot(((rot_axis[0]+180)%360)*resol*self.env.Nphi/360, (180-rot_axis[1])*resol*self.env.Ntheta/180, color='blue', marker='X', markersize=10)
            plt.colorbar()
            name = "Env No."+str(env_no)+" (ref_reward="+str(int(ref_reward*1000)/1000)+") Delta Reward MAP"
            name_ratio = "(Ratio actions = ["+str(ratio_actions[0])+", "+str(ratio_actions[1])+"])"
            name_rot_axis = "(Rot_Axis = ["+str(int(100*rot_axis[0])/100)+", "+str(int(100*rot_axis[1])/100)+"])"
            plt.title(name+"\n"+name_ratio)
            plt.xticks(phi_ticks[0], phi_ticks[1])
            plt.yticks(theta_ticks[0], theta_ticks[1])
            plt.savefig(path+name+name_ratio+".png", dpi=300)
            plt.close()
            """

            """
            plt.figure(figsize=(20, 11))
            plt.imshow(delta_map_temp)
            plt.plot(rot_axis[0]*resol*self.env.Nphi/360, rot_axis[1]*resol*self.env.Ntheta/180, color='red', marker='X', markersize=10)
            plt.plot(((rot_axis[0]+180)%360)*resol*self.env.Nphi/360, (180-rot_axis[1])*resol*self.env.Ntheta/180, color='blue', marker='X', markersize=10)
            plt.colorbar()
            plt.quiver(X, Y, gradx, grady, color='gold', angles='xy', headwidth=2, headlength=4)
            name = "Env No."+str(env_no)+" (ref_reward="+str(int(ref_reward*1000)/1000)+") Delta+Grad Reward MAP"
            name_ratio = "(Ratio actions = ["+str(ratio_actions[0])+", "+str(ratio_actions[1])+"])"
            name_rot_axis = "(Rot_Axis = ["+str(int(100*rot_axis[0])/100)+", "+str(int(100*rot_axis[1])/100)+"])"
            plt.title(name+"\n"+name_ratio)
            plt.xticks(phi_ticks[0], phi_ticks[1])
            plt.yticks(theta_ticks[0], theta_ticks[1])
            plt.savefig(path+name+name_ratio+".png", dpi=300)
            plt.close()
            """
            

    def __map_tick_list(self, resol:int, N_ang:int, max_ang):
        tick_num = 12
        tick_value = []
        tick_label = []

        for i in range(tick_num):
            tick_value.append(i*(resol*N_ang)/tick_num)
            tick_label.append(str(i*int((max_ang*100)//tick_num)/100))

        return tick_value, tick_label



l_max = 8
merge_num = 3
N_set = (40, 20)
lightcurve_unit_len = 100

data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/ell_upto_-100.npz"
total_data = np.load(data_path)
X_total = total_data["lc_arr"]
ell_total = total_data["ell_arr"]
reward_arr = total_data["reward_arr"]

shuffle_idx = np.arange(0, X_total.shape[0])
np.random.shuffle(shuffle_idx)
X_total = X_total[shuffle_idx, :]
ell_total = ell_total[shuffle_idx, :]
reward_arr = reward_arr[shuffle_idx]

prec = 8
reward_domain = [0, 50] #-20, 50

start_idx = 510
total_data_set_defined = True
if total_data_set_defined:
    total_data_set_arr = np.load("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_RL_preset_"+str(start_idx)+".npy")
    
for i in tqdm(range(X_total[start_idx:].shape[0])):
    if i%30 == 0 and i != 0:
        np.save("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_RL_preset_"+str(i+start_idx)+".npy",
                  total_data_set_arr)
        print(total_data_set_arr.shape)

    env = AstEnv(X_total[i, :-9*merge_num], X_total[i, -9*merge_num:], merge_num, reward_domain, N_set, lightcurve_unit_len, (True, ell_total[i, :]))
    if env.ell_err:# or env.reward0 >= 10:
        continue
    state_dim = (env.Ntheta//env.ast_obs_unit_step)*(env.Nphi//env.ast_obs_unit_step) + (lightcurve_unit_len//env.lc_obs_unit_step) + 9*5
    action_dim = 4

    runner = Runner(env, state_dim, action_dim)
    runner.run(env_no=i, random=True)

    if not total_data_set_defined:
        total_data_set_arr = runner.data_set_arr
        total_data_set_defined = True
    else:
        total_data_set_arr = np.concatenate((total_data_set_arr, runner.data_set_arr[1:]), axis=0)
        print(total_data_set_arr.shape)

    if i >= 600:
        break

test_image_num = 1
test_image_generated = 0
test_image_gen_done = False
for i in tqdm(range(X_total[:].shape[0])):
    env = AstEnv(X_total[i, :-9*merge_num], X_total[i, -9*merge_num:], merge_num, reward_domain, N_set, lightcurve_unit_len, (True, ell_total[i, :]))
    if env.ell_err:# or env.reward0 >= 10:
        continue
    test_image_generated += 1
    if test_image_generated == test_image_num:
        test_image_gen_done = True
    state_dim = (env.Ntheta//env.ast_obs_unit_step)*(env.Nphi//env.ast_obs_unit_step) + (lightcurve_unit_len//env.lc_obs_unit_step) + 9*5
    action_dim = 4

    runner = Runner(env, state_dim, action_dim)
    runner.run(env_no=i, random=False)

    total_data_set_arr = np.concatenate((total_data_set_arr, runner.data_set_arr[1:]), axis=0)
    print(total_data_set_arr.shape)

    if test_image_gen_done:
        np.save("C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_RL_preset_"+str(i+start_idx)+".npy",
                  total_data_set_arr)
        print(total_data_set_arr.shape)
        break