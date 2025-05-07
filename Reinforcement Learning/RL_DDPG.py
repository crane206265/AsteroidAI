import os

import torch.distributions.constraints
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from numpy import linalg as LA
import torch
from torch import nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

from Environment import AstEnv
from DataPreprocessing import DataPreProcessing


# seed
seed = 722
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# hyperparameters
LR_ACTOR = 1e-4
LR_CRITIC = 5e-4
BATCH_SIZE = 64
BUFFER_SIZE = int(1e7)
GAMMA = 0.99
TAU = 5e-3
COLLECT_EPISODES = 60
MAX_STEPS = 500 #per episode

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
	def __init__(self, state_dim, action_dim, max_size=BUFFER_SIZE):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, reward, next_state):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
        # 0 ~ self.size에서 batch_size만큼 샘플링해주는 함수(dtype=int)
		ind = np.random.randint(0, self.size, size=batch_size) 

		return (
			torch.FloatTensor(self.state[ind]).to(self.device), # FloatTensor의 기본형은 torch.float32
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device)
        )


class ActorNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, activation=nn.ReLU, dropout=0.3):
        super(ActorNet, self).__init__()   
        
        self.model = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, action_dim),
            
            nn.Tanh()
        )
        
    def forward(self, state):
        return self.model(state)
        
class CriticNet(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256, activation=nn.ReLU, dropout=0.3):
        super(CriticNet, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(obs_dim+action_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)
        return self.model(cat)
    

class DDPG:
    def __init__(self, obs_dim, action_dim, hidden_dim=256, activation=nn.ReLU, dropout=0.3):
        self.buffer = ReplayBuffer(obs_dim, action_dim, BUFFER_SIZE)
        
        self.criticNet = CriticNet(obs_dim, action_dim, hidden_dim, activation, dropout).to(device)
        self.actorNet = ActorNet(obs_dim, action_dim, hidden_dim, activation, dropout).to(device)
        self.criticTargetNet = CriticNet(obs_dim, action_dim, hidden_dim, activation, 0.0).to(device)
        self.actorTargetNet = ActorNet(obs_dim, action_dim, hidden_dim, activation, 0.0).to(device)
        self.criticTargetNet.load_state_dict(self.criticNet.state_dict())
        self.actorTargetNet.load_state_dict(self.actorNet.state_dict())
        
        self.critic_optimizer = torch.optim.Adam(self.criticNet.parameters(), lr=LR_CRITIC)
        self.actor_optimizer = torch.optim.Adam(self.actorNet.parameters(), lr=LR_ACTOR)
        
    def sample_action(self, state:np.ndarray):
        state = torch.from_numpy(state).float().to(device)
        with torch.no_grad():
            action = self.actorNet(state)
        return action.cpu().numpy()
        
    def soft_target_net_update(self):
        actor_net_state_dict = self.actorNet.state_dict()
        critic_net_state_dict = self.criticNet.state_dict()
        actor_target_net_state_dict = self.actorTargetNet.state_dict()
        critic_target_net_state_dict = self.criticTargetNet.state_dict()
        
        for a_key, c_key in zip(actor_net_state_dict, critic_net_state_dict):
            actor_target_net_state_dict[a_key] = actor_net_state_dict[a_key]*TAU + actor_target_net_state_dict[a_key]*(1-TAU)
            critic_target_net_state_dict[c_key] = critic_net_state_dict[c_key]*TAU + critic_target_net_state_dict[c_key]*(1-TAU)
            
        self.actorTargetNet.load_state_dict(actor_target_net_state_dict)
        self.criticTargetNet.load_state_dict(critic_target_net_state_dict)
        
    def update(self):
        state, action, reward, next_state = self.buffer.sample(BATCH_SIZE)
        
        q_values = self.criticNet(state, action) # size [BATCH_SIZE,1]
        
        with torch.no_grad():
            next_q_values= self.criticTargetNet(next_state, self.actorTargetNet(next_state))
            target = reward + GAMMA * next_q_values
            
        self.critic_optimizer.zero_grad()
        #critic_loss = F.smooth_l1_loss(q_values, target)
        critic_loss = F.mse_loss(q_values/(target+1e-8), target/(target+1e-8))
        critic_losses.append(critic_loss.item())
        critic_loss.backward()
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        actor_loss = -self.criticNet(state, self.actorNet(state)).mean()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_target_net_update()

"""
def main():
    env = gym.make('Pendulum-v1')
    #env = gym.make('Pendulum-v1', g=9.81, render_mode = 'human') # 학습하는 것을 보고싶다면
    obs_space_dims = env.observation_space.shape[0]
    action_space_dims = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])
    expl_noise = 0.05
    
    # 출력 설정
    eval_freq = 1000
    
    # seed 설정
    #seed = 117 
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)
    
    state, _ =env.reset(seed=seed)
    episode_timesteps = 0
    episode_num = 0
        
    agent = DDPG(obs_space_dims, action_space_dims)
    score = 0.0
        
    for t in range(int(MAX_TIMESTEPS)):
        
        episode_timesteps += 1
                
        if t < START_TIMESTEPS: # 학습 전 충분한 데이터를 모으기 위해서.
            action = env.action_space.sample() # action은 numpy array
        else:
            action = (
                agent.sample_action(state)
                + np.random.normal(0, max_action * expl_noise, size=action_space_dims)
            ).clip(-max_action, max_action)
            
        observation, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        next_state = observation if not terminated else None
        
        agent.buffer.add(state, action, reward, next_state)
        agent.update()
        
        score += reward
        state = next_state
        
        if done:
            state, _ = env.reset(seed=seed)
            episode_num += 1
            
        if (t + 1) % eval_freq == 0:
            print(f"time_step : {t+1} --> avg_score : {score/eval_freq}")
            score = 0.0
"""
            

class Runner():
    def __init__(self, env:AstEnv, prec = 5.0):
        self.env = env
        self.state = None
        self.done = True
        self.passed = False
        self.all_done = False
        self.reward_past = None
        self.episode_reward = 0
        self.episode_rewards = []
        self.prec = prec

    def reset(self, passed):
        self.done = False
        self.state = self.env.reset(passed)
        self.state = np.concatenate((self.state, np.array([self.env.reward0])))
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

    def run(self, collecting, reward_lists, success_bool):
        max_reward_list = reward_lists[0]
        done_lengths = reward_lists[1]
        modified_reward_list = reward_lists[2]

        prec = self.prec
        r_cen_low = 0.2
        r_cen_high = 1
        R_cut_low = 0.40 #0.01 #0.25
        R_cut_high = 1 #0.15
        expl_noise = 1e-5

        max_reward = -9e+8
        show_bool = True
        semi_success = success_bool
        for t in range(MAX_STEPS):
            if self.all_done:
                print(" SUCCESSED in CURRENT ENV")
                print(" episode:", len(self.episode_rewards), ", episode reward:", self.episode_reward)
                self.env.show()
                break

            if self.done:
                self.reset(self.passed)

            if collecting:
                actions = np.random.uniform(-prec, prec, 4)
            else:
                actions = (
                    agent.sample_action(self.state)
                    + np.random.normal(0, prec * expl_noise, size=4)
                )

            actions[:-2] = actions[:-2] - actions[:-2]//1
            actions[-2:] = np.clip(actions[-2:], [-prec, -prec], [prec, prec])
            actions[2] = (actions[2]+prec)*(r_cen_high-r_cen_low)/(2*prec) + r_cen_low
            actions[3] = (actions[3]+prec)*(R_cut_high-R_cut_low)/(2*prec) + R_cut_low

            self.env_max_reward_past = self.env.max_reward + 0
            next_state, reward, self.done, info = self.env.step(actions)
            next_state = np.concatenate((next_state, np.array([reward])))
            if self.reward_past == None:
                self.reward_past = reward
            self.passed = info[0]
            self.all_done = info[1]
            weight = 2.5

            modified_reward = self.reward_fn(reward, t, t_m=500, t_g=300, R_max=50.0, alpha=1.0, beta=0.3, epsilon=0.03)
            if self.done and not self.passed: # if max reward - 3 condition is not satisfied 
                weight = 1.0
                #modified_reward += max((reward - self.env_max_reward_past)/5, -2)
            
            modified_reward_list.append(modified_reward)

            agent.buffer.add(self.state, actions, modified_reward, next_state) #ToDo: weight implementation
            agent.update()

            self.state = next_state
            self.episode_reward += modified_reward
            if weight != 1:
                self.reward_past = reward + 0
            
            max_reward = max(max_reward, reward)
            if t%4 == 0:
                print("{:02d}/{:02d}".format(success, trial-1), end='')
                print(" | Reward : {:7.5g}".format(reward), end='')
                print(" | actions : [{:6.05g}, {:6.05g}, {:6.05g}, {:6.05g}]".format(actions[0], actions[1], actions[2], actions[3]), end='')
                print(" | episode_reward : {:6.05g}".format(self.episode_reward))
                #max_reward = -9e+8
                self.episode_rewards.append(self.episode_reward)
                max_reward_list.append(max_reward)
                if t%20 == 0:
                    show_bool = True

            if show_bool:# and reward >= self.env.reward_threshold - 35:
                print("show_passed : "+str(reward)+" | obs_lc_num : "+str(self.env.obs_lc_num))
                #self.env.show(str(data_num)+"_"+str(et)+"_"+str(k)+"_"+str(int(reward*100)/100)+"_"+"0402ast.png")
                #plt.close()
                show_bool = False

            if reward >= self.env.reward_threshold - 10 and not semi_success:
                semi_success = True
                if self.env.total_threshold - self.env.reward0 >= 20:
                    done_lengths.append(40*t/(self.env.total_threshold - self.env.reward0))
                #self.env.show(str(data_num)+"_"+str(et)+"_"+str(k)+"_"+str(int(reward*100)/100)+"_"+"0306ast.png")
                #plt.close()
                

            if self.done:
                if self.passed:
                    print("show_passed : "+str(reward)+" | obs_lc_num : "+str(self.env.obs_lc_num))
                    #self.env.show(str(data_num)+"_"+str(et)+"_"+str(k)+"_"+str(int(reward*100)/100)+"_"+"0306ast.png")
                    #plt.close()
                    #self.episode_reward += 200
                    self.episode_rewards.append(self.episode_reward)
                    max_reward_list.append(max_reward)
                    #done_lengths.append(40*(k+et*max_steps)/(self.env.total_threshold - self.env.reward0))
                    break
                #self.episode_rewards.append(self.episode_reward)
                #max_reward_list.append(max_reward)
                #if len(self.episode_rewards) % 10 == 0:
                #    print(" episode:", len(self.episode_rewards), ", episode reward:", self.episode_reward)
                
        return reward, (self.episode_rewards, max_reward_list, done_lengths, modified_reward_list), semi_success

def training_plot(path, **kwargs):
    reward_list = kwargs['reward_list']
    max_reward_list = kwargs['max_reward_list']
    modified_reward_list = kwargs['modified_reward_list']
    critic_losses = kwargs['critic_losses']
    done_lengths = kwargs['done_lengths']

    plt.figure(figsize=(20, 7))
    plt.plot(reward_list)
    name = "episode_reward_list" + str(i)
    plt.title(name)
    plt.savefig(path+name+".png")
    plt.close()

    plt.figure(figsize=(20, 7))
    plt.plot(max_reward_list)
    name = "max_reward_list" + str(i)
    plt.title(name)
    plt.savefig(path+name+".png")
    plt.close()

    plt.figure(figsize=(20, 7))
    plt.plot(modified_reward_list)
    name = "modified_reward_list" + str(i)
    plt.title(name)
    right_lim = len(modified_reward_list)
    plt.xlim([max(right_lim-4000, 0), right_lim])
    plt.savefig(path+name+".png")
    plt.close()

    plt.figure(figsize=(20, 7))
    critic_losses = np.array(critic_losses)
    critic_losses[:10] = 0 #avoid plotting first "CRACK" values
    critic_losses = np.log(critic_losses+0.8)
    plt.plot(critic_losses)
    name = "critic_losses" + str(i)
    plt.title(name)
    #plt.ylim([0, 1.5])
    plt.savefig(path+name+".png")
    plt.close()

    plt.figure(figsize=(20, 7))
    plt.plot(done_lengths)
    name = "done_lengths" + str(i)
    plt.title(name)
    plt.grid(True)
    _done_length = np.array(done_lengths)
    _x_arr = np.arange(1, len(_done_length)+1)
    _beta0 = np.sum((_x_arr - np.mean(_x_arr))*(_done_length - np.mean(_done_length))) / np.sum((_x_arr - np.mean(_x_arr))**2)
    _beta1 = np.mean(_done_length) - np.mean(_x_arr) * _beta0
    _y_arr = _beta0 * _x_arr + _beta1
    plt.plot(_y_arr, linestyle='--')
    plt.savefig(path+name+".png")
    plt.close()



l_max = 8
merge_num = 3
N_set = (32, 16)
lightcurve_unit_len = 100
data_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/data/data_total.npz"
model_save_path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/checkpoints/"

"""
dataPP = DataPreProcessing(data_path=data_path)
dataPP.X_total = torch.concat((dataPP.X_total[:, :100], dataPP.X_total[:, -9:]), dim=-1)
dataPP.Y_total = dataPP.Y_total[:, 0:(l_max+1)**2]
dataPP.coef2R(dataPP.Y_total, l_max=l_max, N_set=N_set)
dataPP.merge(merge_num=merge_num, ast_repeat_num=10, lc_len=lightcurve_unit_len, dupl_ratio=0.01)
dataPP.X_total = dataPP.X_total.numpy()
dataPP.Y_total = dataPP.Y_total.numpy()
X_total, _, y_total, _ = dataPP.train_test_split(trainset_ratio=0.1)
X_total = X_total[:100+1, :]"
"""

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

reward_list = []
max_reward_list = []
done_lengths = []
modified_reward_list = []
critic_losses = []

checkpoint_load = False
checkpoint_epoch = 0
prec = 8
reward_domain = [0, 45] #-20, 50
domain_name = "["+str(reward_domain[0])+","+str(reward_domain[1])+"]"
trial = 0
success = 0
agent = None

for i in tqdm(list(range(X_total.shape[0]))*3):
    #obs_lc_num = np.random.randint(merge_num)
    #obs_lc_time = np.argmax(X_total[i, lightcurve_unit_len*(obs_lc_num):lightcurve_unit_len*(obs_lc_num+1)])

    # Chekcpoint
    if i%20 == 0 and i != 0:
        torch.save({
            'epoch':i,
            'model_state_dict':agent.actorNet.state_dict(),
            'optimizer_state_dict':agent.actor_optimizer.state_dict()
        }, model_save_path+"actor"+domain_name+"_"+str(i+checkpoint_epoch)+".pt")

        torch.save({
            'epoch':i,
            'model_state_dict':agent.criticNet.state_dict(),
            'optimizer_state_dict':agent.critic_optimizer.state_dict()
        }, model_save_path+"critic"+domain_name+"_"+str(i+checkpoint_epoch)+".pt")

        with open(model_save_path+"reward_list"+domain_name+".txt", 'a+') as file:
            for item in reward_list:
                file.write(str(item)+",")

        with open(model_save_path+"max_reward_list"+domain_name+".txt", 'a+') as file:
            for item in max_reward_list:
                file.write(str(item)+",")

    if i%20 == 0 and i != 0:
        path = "C:/Users/dlgkr/OneDrive/Desktop/code/astronomy/asteroid_AI/A2C_train/reward_show/"
        training_plot(path, reward_list=reward_list, max_reward_list=max_reward_list, modified_reward_list=modified_reward_list, 
                      critic_losses=critic_losses, done_lengths=done_lengths)
        

    env = AstEnv(X_total[i, :-9*merge_num], X_total[i, -9*merge_num:], merge_num, reward_domain, N_set, lightcurve_unit_len, (True, ell_total[i, :]))
    if env.ell_err:
        continue
    trial += 1
    #state_dim = (env.Ntheta*env.Nphi)//(2*env.ast_obs_unit_step) + 2*(lightcurve_unit_len//(4*env.lc_obs_unit_step))
    state_dim = (env.Ntheta//env.ast_obs_unit_step)*(env.Nphi//env.ast_obs_unit_step) + (lightcurve_unit_len//env.lc_obs_unit_step) + 9*5 + 1
    n_actions = 4

    # config
    if agent == None:
        agent = DDPG(state_dim, n_actions, hidden_dim=512, activation=nn.LeakyReLU, dropout=0.15)
        score = 0.0

        if checkpoint_load:
            actor_checkpoint = torch.load(model_save_path+"actor"+domain_name+"_"+str(checkpoint_epoch)+".pt")
            agent.actorNet.load_state_dict(actor_checkpoint['model_state_dict'])
            agent.actor_optimizer.load_state_dict(actor_checkpoint['optimizer_state_dict'])
            critic_checkpoint = torch.load(model_save_path+"critic"+domain_name+"_"+str(checkpoint_epoch)+".pt")
            agent.criticNet.load_state_dict(critic_checkpoint['model_state_dict'])
            agent.critic_optimizer.load_state_dict(critic_checkpoint['optimizer_state_dict'])

    reward_past = 0
    success_bool = False
    collecting = True if i < COLLECT_EPISODES else False
    runner = Runner(env)
        
    print("_____________________________________________________________________________________________")
    last_reward, reward_lists, semi_success = runner.run(collecting, (max_reward_list, done_lengths, modified_reward_list), success_bool)
    episode_rewards = reward_lists[0]
    max_reward_list = reward_lists[1]
    done_lengths = reward_lists[2]
    modified_reward_list = reward_lists[3]
        
    success_bool = success_bool or semi_success
        
    if success_bool:
        success += 1
    reward_list = reward_list + episode_rewards


plt.plot(reward_list)
plt.show()

plt.plot(max_reward_list)
plt.show()
    
plt.plot(done_lengths)
plt.show()