from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import argparse
from normalization import Normalization, RewardScaling
from replaybuffer import ReplayBuffer
from highway_env_config import config_setting
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch.nn as nn
from torch.distributions import Normal
import json
import pickle
import warnings
warnings.filterwarnings("ignore")


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = torch.tensor(args.max_action)
        self.fc1 = nn.Linear(args.state_dim, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.mean_layer = nn.Linear(args.hidden_size, args.action_dim)
        self.log_std = nn.Parameter(torch.zeros(1, args.action_dim)) 
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        mean = torch.tanh(self.mean_layer(s)) 
        return mean

    def get_dist(self, s):
        mean = self.forward(s)
        log_std = self.log_std.expand_as(mean) 
        std = torch.exp(log_std)  
        dist = Normal(mean, std) 
        return dist
    

class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(args.state_dim, args.hidden_size)
        self.fc2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.fc3 = nn.Linear(args.hidden_size, 1)
        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh] 

    def forward(self, s):
        s = self.activate_func(self.fc1(s))
        s = self.activate_func(self.fc2(s))
        v_s = self.fc3(s)
        return v_s


class PPO_continuous():
    def __init__(self, args):
        self.max_action = torch.tensor(args.max_action)
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch
        self.max_train_steps = args.max_steps*args.max_episodes
        self.lr_a = args.lr_a  
        self.lr_c = args.lr_c  
        self.gamma = args.gamma
        self.lamda = args.lamda  
        self.epsilon = args.epsilon  
        self.K_epochs = args.K_epochs  
        self.entropy_coef = args.entropy_coef 
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm
        self.chkpt_dir = args.save_dir
        self.date = args.date
        
        self.actor = Actor(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample()  
            a = torch.clamp(a, -self.max_action, self.max_action) 
            a_logprob = dist.log_prob(a)  
        return a.data.cpu().numpy().flatten(), a_logprob.data.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, s_, dw, done = replay_buffer.sample() 
        adv = []
        gae = 0
        with torch.no_grad(): 
            vs = self.critic(s)
            vs_ = self.critic(s_)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)
                a_logprob_now = dist_now.log_prob(a[index])
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True)) 

                surr1 = ratios * adv[index]  
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy 
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip: 
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  
            self.lr_decay(total_steps)

    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now
    
    def save_models(self):
        print("---saving models---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{self.date}.pl")
        torch.save(self.actor.state_dict(), agent_actor)

    def load_models(self, date):
        print("---loading model---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{date}.pl")
        self.actor.load_state_dict(torch.load(agent_actor))
    

class ReplayBuffer:
    def __init__(self, args):
        self.state = np.zeros((args.batch_size, args.state_dim))
        self.action = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.reward = np.zeros((args.batch_size, 1))
        self.next_state = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, state, action, a_logprob, reward, next_state, dw, done):
        self.state[self.count] = state
        self.action[self.count] = action
        self.a_logprob[self.count] = a_logprob
        self.reward[self.count] = reward
        self.next_state[self.count] = next_state
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def sample(self):
        state = torch.tensor(self.state, dtype=torch.float)
        a = torch.tensor(self.action, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        reward = torch.tensor(self.reward, dtype=torch.float)
        next_state = torch.tensor(self.next_state, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return state, a, a_logprob, reward, next_state, dw, done

def rotated_axis(x, y, theta):
    x_r = x*np.cos(theta) + y*np.sin(theta)
    y_r = -x*np.sin(theta) + y*np.cos(theta)
    return x_r, y_r

def call_cost(next_state, info):
    xr, yr = rotated_axis(next_state[5::5], next_state[6::5], next_state[4])
    bound_a, bound_b = np.sqrt(abs(info['speed'])) + 10, 3.5
    z = 10*(np.sqrt(xr**2/bound_a**2 + yr**2/bound_b**2) - 1)**2
    mask = (xr**2/bound_a**2 + yr**2/bound_b**2)<=1
    cost_hazard = z*mask
    edge = 0.1*np.clip(next_state[1]*(next_state[1] - 12), a_max=np.inf, a_min=0)
    c1 = 0.1*(info['speed']-26)**2
    c2 = np.sum(cost_hazard)
    c3 = edge
    cost = np.array([c1, c2, c3])
    return cost

def trian(args, agent, env, replay_buffer):
    # save_args(args)
    start_time = time.time()
    writer = SummaryWriter(log_dir=args.log_dir)
    save_dir = os.path.join(args.save_dir, f"state_norm.pkl")
    score_history = []
    best_score = -np.inf
    avg_score = 0
    total_steps = 0 

    state_norm = Normalization(shape=args.state_dim) 
    if args.use_reward_norm: 
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    for episode in range(args.max_episodes):
        state, _ = env.reset()
        state = state.reshape(-1)
        if args.use_state_norm:
            state = state_norm(state)
        if args.use_reward_scaling:
            reward_scaling.reset()
        ep_reward = 0
        CostSpeed, CostCrash, CostEdge = 0, 0, 0
        v_record = []
        for step in range(args.max_steps):
            action, a_logprob = agent.choose_action(state)  
            next_state, reward, done, _, info = env.step(action)
            next_state = next_state.reshape(-1)

            reward += info['speed']/100. if info['speed']<30. else 0.

            ep_reward += reward
            cost = call_cost(next_state, info)
            CostSpeed += cost[0]
            CostCrash += cost[1]
            CostEdge += cost[2]

            if args.use_state_norm:
                next_state = state_norm(next_state)
            if args.use_reward_norm:
                reward = reward_norm(reward)
            elif args.use_reward_scaling:
                reward = reward_scaling(reward)

            if done and step+1 != args.max_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(state, action, a_logprob, reward, next_state, dw, done)
            state = next_state
            total_steps += 1
            v_record.append(info['speed'])
            if replay_buffer.count == args.batch_size:
                print("----Update agent----")
                agent.update(replay_buffer, total_steps)
                replay_buffer.count = 0

            if done:
                break
        v_avg = np.mean(v_record)
        score_history.append(ep_reward)
        avg_score = np.mean(score_history[-50:])
        if avg_score > best_score:
            best_score = avg_score
            if episode > 100:
                os.makedirs(args.save_dir, exist_ok=True)
                agent.save_models()
                save_norm = open(save_dir, 'wb')
                tree_str = pickle.dumps(state_norm)
                save_norm.write(tree_str)
        writer.add_scalar('Reward/reward', ep_reward, episode)
        writer.add_scalar('Reward/avg', avg_score, episode)
        writer.add_scalar('Cost/speed', CostSpeed, episode)
        writer.add_scalar('Cost/crash', CostCrash, episode)
        writer.add_scalar('Cost/Edge', CostEdge, episode)
        writer.add_scalar('Info/step', step+1, episode)
        writer.add_scalar('Info/v_avg', v_avg, episode)
        
        print('episode:',episode+1,
              '\tsteps:',step+1,
              '\tscore:%.1f'%ep_reward,
              '\tavg:%.1f' % avg_score,
              '\tbest:%.1f'%best_score,
              '\tspeed:%.1f'%v_avg,
              '\tEdge:%.1f'%CostEdge,
              '\tCrash:%.1f'%CostCrash,
              '\tSpeed:%.1f'%CostSpeed
              )
    
    end_time = time.time()
    total_seconds = end_time - start_time
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = total_seconds % 60
    print(f"Time: {hours} h {minutes} m {seconds:.2f} s")
    writer.close()
    save_norm.close()
    env.close()

    
def save_args(args):
    os.makedirs(args.log_dir, exist_ok=True)
    json_path = os.path.join(args.log_dir, "args.json")
    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(vars(args), f_json, ensure_ascii=False, indent=4, default=str)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO")
    parser.add_argument("--max_episodes", type=int, default=int(4000), help="Episodes")
    parser.add_argument("--max_steps", type=int, default=int(300), help="Steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch", type=int, default=256, help="Minibatch size")
    parser.add_argument("--hidden_size", type=int, default=256, help="The number of neurons")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="tanh activation function")
    args = parser.parse_args()
    
    args.agent_name = "PPO"
    args.env_name = "highway-v0"
    args.date = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = "agent/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    args.log_dir = "logs/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    
    
    config = config_setting()
    
    env = gym.make(args.env_name, 
                #    render_mode='human',
                   config=config)
    
    args.state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
    args.action_dim = env.action_space.shape[0]
    args.max_action = np.array([3.0, 0.3])
    
    start_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    args.date = datetime.now().strftime("%Y%m%d%H%M")
    
    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    
    trian(args, agent, env, replay_buffer)
    
    print(f"Agent: {args.agent_name}")
    print(f"Start date: {start_date}")
    end_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    print(f"End date: {end_date}")
    