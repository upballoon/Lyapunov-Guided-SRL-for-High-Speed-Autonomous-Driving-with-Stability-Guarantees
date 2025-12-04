from torch.utils.tensorboard import SummaryWriter
import gymnasium as gym
import argparse
from normalization import Normalization, RewardScaling
from highway_env_config import config_setting
import os
import time
from datetime import datetime
import numpy as np
import copy
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
        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch
        self.max_train_steps = args.max_steps*args.max_episodes
        self.lr_a = args.lr_a  
        self.lr_c = args.lr_c  
        self.lr_l = args.lr_l
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
        self.use_lagrangian = args.use_lagrangian
        self.adaptive_lambda = args.adaptive_lambda
        self.cost_dim = args.cost_dim
        self.cost_limit = torch.tensor(args.cost_limit)
        
        self.actor = Actor(args)
        self.critic = Critic(args)

        if self.set_adam_eps:  
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)
        
        if self.use_lagrangian:
            self.critic_cost = []
            self.critic_cost_optimizer = []
            for i in range(self.cost_dim):
                self.critic_cost.append(Critic(args))
                self.critic_cost_optimizer.append(torch.optim.Adam(self.critic_cost[i].parameters(), lr=self.lr_c))
            
            self.log_lambda = []
            self.lam_lag = []
            if self.adaptive_lambda:
                self.lambda_optimizer = []
                for i in range(self.cost_dim):
                    self.log_lambda.append(torch.zeros(1))
                    self.log_lambda[i].requires_grad = True
                    self.lam_lag.append(self.log_lambda[i].exp())
                    self.lambda_optimizer.append(torch.optim.Adam([self.log_lambda[i]], lr=self.lr_l))
            else:
                for i in range(self.cost_dim):
                    self.lam_lag.append(1.0)

    def choose_action(self, s):
        s = torch.unsqueeze(torch.tensor(s, dtype=torch.float), 0)
        with torch.no_grad():
            dist = self.actor.get_dist(s)
            a = dist.sample() 
            a = torch.clamp(a, -1.0, 1.0)  
            a_logprob = dist.log_prob(a) 
        return a.data.cpu().numpy().flatten(), a_logprob.data.cpu().numpy().flatten()

    def update(self, replay_buffer, total_steps):
        s, a, a_logprob, r, c,  s_, dw, done = replay_buffer.sample() 
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
        
        if self.use_lagrangian:
            advantage_cost = []
            v_cost_target = []
            for i in range(self.cost_dim):
                adv_cost = []
                gae_cost = 0
                with torch.no_grad(): 
                    vs_cost = self.critic_cost[i](s)
                    vs_cost_ = self.critic_cost[i](s_)
                    deltas_cost = c[i] + self.gamma * (1.0 - dw) * vs_cost_ - vs_cost
                    for delta_cost, d in zip(reversed(deltas_cost.flatten().numpy()), reversed(done.flatten().numpy())):
                        gae_cost = delta_cost + self.gamma * self.lamda * gae_cost * (1.0 - d)
                        adv_cost.insert(0, gae_cost)
                    adv_cost = torch.tensor(adv_cost, dtype=torch.float).view(-1, 1)
                    v_cost_target.append(adv_cost + vs_cost)
                    if self.use_adv_norm: 
                        advantage_cost.append((adv_cost - adv_cost.mean()) / (adv_cost.std() + 1e-5))

        for _ in range(self.K_epochs):
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                dist_now = self.actor.get_dist(s[index])
                dist_entropy = dist_now.entropy().sum(1, keepdim=True)  
                a_logprob_now = dist_now.log_prob(a[index])
                ratios = torch.exp(a_logprob_now.sum(1, keepdim=True) - a_logprob[index].sum(1, keepdim=True)) 

                surr1 = ratios * adv[index] 
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy 
                
                if self.use_lagrangian:
                    surr_cost = []
                    for i in range(self.cost_dim):
                        surr_cost.append(ratios * advantage_cost[i][index])
                        actor_loss += self.lam_lag[i] * surr_cost[i]
                
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
                
                # Update critic_cost
                for i in range(self.cost_dim):
                    v_s_cost = self.critic_cost[i](s[index])
                    critic_cost_loss = F.mse_loss(v_cost_target[i][index], v_s_cost)
                    self.critic_cost_optimizer[i].zero_grad()
                    critic_cost_loss.backward()
                    if self.use_grad_clip:  
                        torch.nn.utils.clip_grad_norm_(self.critic_cost[i].parameters(), 0.5)
                    self.critic_cost_optimizer[i].step()
                    
                    # Update lambda
                    if self.use_lagrangian:
                        if self.adaptive_lambda:
                            cost_violation = surr_cost[i].detach() - self.cost_limit
                            lagrangian_loss = -torch.mean(self.log_lambda[i].exp() * cost_violation)
                            self.lambda_optimizer[i].zero_grad()
                            lagrangian_loss.backward()
                            self.lambda_optimizer[i].step()
                            self.lam_lag[i] = self.log_lambda[i].exp().detach()
                            with torch.no_grad():
                                self.lam_lag[i].clamp_(0)
                        else:
                            self.lam_lag[i] += self.lr_c * surr_cost[i].detach().mean().item()
                            self.lam_lag[i] = max(self.lam_lag[i], 0.0)

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

    def load_models(self, args):
        print("---loading model---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{args.date}.pl")
        self.actor.load_state_dict(torch.load(agent_actor))
    

class ReplayBuffer:
    def __init__(self, args):
        self.state = np.zeros((args.batch_size, args.state_dim))
        self.action = np.zeros((args.batch_size, args.action_dim))
        self.a_logprob = np.zeros((args.batch_size, args.action_dim))
        self.reward = np.zeros((args.batch_size, 1))
        self.cost = np.zeros((args.batch_size, args.cost_dim))
        self.next_state = np.zeros((args.batch_size, args.state_dim))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    def store(self, state, action, a_logprob, reward, cost, next_state, dw, done):
        self.state[self.count] = state
        self.action[self.count] = action
        self.a_logprob[self.count] = a_logprob
        self.reward[self.count] = reward
        self.cost[self.count] = cost
        self.next_state[self.count] = next_state
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    def sample(self):
        state = torch.tensor(self.state, dtype=torch.float)
        action = torch.tensor(self.action, dtype=torch.float)
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        reward = torch.tensor(self.reward, dtype=torch.float)
        cost = torch.tensor(self.cost, dtype=torch.float)
        next_state = torch.tensor(self.next_state, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return state, action, a_logprob, reward, cost, next_state, dw, done
    

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
    save_args(args)
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
            a = action*args.max_action
            next_state, reward, done, _, info = env.step(a)
            next_state = next_state.reshape(-1)
            reward += info['speed']/100. if info['speed']<30. else 0.
            
            cost = call_cost(next_state, info)
            c = cost  
            ep_reward += reward
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
            
            replay_buffer.store(state, action, a_logprob, reward, c, next_state, dw, done)
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
              '\tSpeed:%.1f'%CostSpeed)

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
    parser = argparse.ArgumentParser("Hyperparameters Setting for PPO-continuous")
    parser.add_argument("--max_episodes", type=int, default=int(4000), help="Episodes")
    parser.add_argument("--max_steps", type=int, default=int(300), help="Steps")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size")
    parser.add_argument("--mini_batch", type=int, default=256, help="Minibatch size")
    parser.add_argument("--hidden_size", type=int, default=256, help="The number of neurons")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--lr_l", type=float, default=3e-4, help="Learning rate of lagrangain")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="Trick 4:reward scaling")
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=False, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=False, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=False, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")
    args = parser.parse_args()
    
    args.use_lagrangian = True
    args.adaptive_lambda = False

    args.agent_name = "PPOLag"
    args.env_name = "highway-v0"
    args.date = datetime.now().strftime("%Y%m%d%H%M")
    args.save_dir = "agent/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    args.log_dir = "logs/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    from highway_env_config import config_setting
    config = config_setting()
    
    env = gym.make(args.env_name, 
                #    render_mode='human',
                   config=config)
    
    args.state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
    args.action_dim = env.action_space.shape[0]
    args.max_action = np.array([3.0, 0.3])
    args.cost_dim = 3
    args.cost_limit = 1e-2  
    
    start_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    args.date = datetime.now().strftime("%Y%m%d%H%M")
     
    replay_buffer = ReplayBuffer(args)
    agent = PPO_continuous(args)
    
    trian(args, agent, env, replay_buffer)
    
    print(f"Agent: {args.agent_name}")
    print(f"Start date: {start_date}")
    end_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    print(f"End date: {end_date}")

    
    