import os
import copy
import time
import argparse
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.set_printoptions(suppress=True, precision=2)
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal
from datetime import datetime
import json
import warnings
warnings.filterwarnings("ignore")
from highway_env_config import config_setting


class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()
        self.max_action = torch.tensor(args.max_action, dtype=torch.float).to(args.device)
        self.l1 = nn.Linear(args.state_dim, args.hidden_size)
        self.l2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.mean_layer = nn.Linear(args.hidden_size, args.action_dim)
        self.log_std_layer = nn.Linear(args.hidden_size, args.action_dim)

    def forward(self, x, deterministic=False, with_logprob=True):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        mean = self.mean_layer(x)
        log_std = self.log_std_layer(x)  
        log_std = torch.clamp(log_std, -20, 2)
        std = torch.exp(log_std)
        dist = Normal(mean, std)  
        if deterministic:  
            action = mean
        else:
            action = dist.rsample()  
        if with_logprob: 
            log_pi = dist.log_prob(action).sum(dim=1, keepdim=True)
            log_pi -= (2 * (np.log(2) - action - F.softplus(-2 * action))).sum(dim=1, keepdim=True)
        else:
            log_pi = None
        action = self.max_action * torch.tanh(action)
        return action, log_pi


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.l1 = nn.Linear(args.state_dim + args.action_dim, args.hidden_size)
        self.l2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.l3 = nn.Linear(args.hidden_size, 1)

        self.l4 = nn.Linear(args.state_dim + args.action_dim, args.hidden_size)
        self.l5 = nn.Linear(args.hidden_size, args.hidden_size)
        self.l6 = nn.Linear(args.hidden_size, 1)

    def forward(self, state, action):
        s_a = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(s_a))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(s_a))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
    
    
class CostNet(nn.Module):  
    def __init__(self, args, activation='tanh'):
        super(CostNet, self).__init__()
        if activation == 'tanh':
            self.activation = F.tanh
        elif activation == 'relu':
            self.activation = F.relu
        else:
            self.activation = F.sigmoid
        self.l1 = nn.Linear(args.state_dim + args.action_dim, args.hidden_size)
        self.l2 = nn.Linear(args.hidden_size, args.hidden_size)
        self.l3 = nn.Linear(args.hidden_size, 1)

        self.l4 = nn.Linear(args.state_dim + args.action_dim, args.hidden_size)
        self.l5 = nn.Linear(args.hidden_size, args.hidden_size)
        self.l6 = nn.Linear(args.hidden_size, 1)

    def forward(self, s, a):
        s_a = torch.cat([s, a], 1)
        q1 = self.activation(self.l1(s_a))
        q1 = self.activation(self.l2(q1))
        q1 = self.l3(q1)
        q1 = q1*q1 
        
        q2 = self.activation(self.l4(s_a))
        q2 = self.activation(self.l5(q2))
        q2 = self.l6(q2)
        q2 = q2*q2 

        return q1, q2


class ReplayBuffer(object):
    def __init__(self, args):
        self.max_size = int(args.batch_size)
        self.count = 0
        self.size = 0
        self.state = np.zeros((self.max_size, args.state_dim))
        self.action = np.zeros((self.max_size, args.action_dim))
        self.reward = np.zeros((self.max_size, 1))
        self.cost = np.zeros((self.max_size, args.cost_dim))
        self.next_state = np.zeros((self.max_size, args.state_dim))
        self.done = np.zeros((self.max_size, 1))

    def store(self, state, action, reward, cost, next_state, done):
        self.state[self.count] = state
        self.action[self.count] = action
        self.reward[self.count] = reward
        self.cost[self.count] = cost
        self.next_state[self.count] = next_state
        self.done[self.count] = done
        self.count = (self.count + 1) % self.max_size  
        self.size = min(self.size + 1, self.max_size) 

    def sample(self, batch_size):
        index = np.random.choice(self.size, size=batch_size)  # Randomly sampling
        batch_s = torch.tensor(self.state[index], dtype=torch.float).to(args.device)
        batch_a = torch.tensor(self.action[index], dtype=torch.float).to(args.device)
        batch_r = torch.tensor(self.reward[index], dtype=torch.float).to(args.device)
        batch_c = torch.tensor(self.cost[index], dtype=torch.float).to(args.device)
        batch_s_ = torch.tensor(self.next_state[index], dtype=torch.float).to(args.device)
        batch_dw = torch.tensor(self.done[index], dtype=torch.float).to(args.device)

        return batch_s, batch_a, batch_r, batch_c, batch_s_, batch_dw


class SAC(object):
    def __init__(self, args):
        self.max_action = args.max_action
        self.action_dim = args.action_dim
        self.hidden_size = args.hidden_size 
        self.mini_batch = args.mini_batch  
        self.gamma = args.gamma 
        self.TAU = args.TAU  
        self.lr = args.lr_a  
        self.chkpt_dir = args.save_dir
        self.date = args.date
        self.adaptive_alpha = True
        self.use_lyapunov = args.use_lyapunov
        self.use_lagrangian = args.use_lagrangian
        self.adaptive_lambda = True
        self.cost_limit = args.cost_limit
        self.cost_dim = args.cost_dim 
        self.device = args.device

        self.actor = Actor(args).to(args.device)
        self.critic = Critic(args).to(args.device)
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        
        if self.adaptive_alpha: 
            self.target_entropy = -args.action_dim
            self.log_alpha = torch.zeros(1).to(args.device) 
            self.log_alpha.requires_grad = True
            self.alpha = self.log_alpha.exp().detach()
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        else:
            self.alpha = 0.2
            
        self.alpha3 = 0.2 
        
        self.critic_lyapunov = CostNet(args).to(args.device)
        self.critic_lyapunov_target = copy.deepcopy(self.critic_lyapunov)
        self.lyapunov_critic_optimizer = torch.optim.Adam(self.critic_lyapunov.parameters(), lr=self.lr)
        
        if self.use_lyapunov:
            if self.adaptive_lambda:
                self.log_lam_lya = torch.zeros(1).to(self.device)
                self.log_lam_lya.requires_grad = True
                self.lam_lya = self.log_lam_lya.exp().detach()
                self.lam_lya_optimizer = torch.optim.Adam([self.log_lam_lya], lr=self.lr)
            else:
                self.lam_lya = 10.0
        
        if self.use_lagrangian:
            self.critic_cost = []
            self.critic_cost_target = []
            self.critic_cost_optimizer = []
            for i in range(self.cost_dim):
                self.critic_cost.append(Critic(args).to(self.device))
                self.critic_cost_target.append(copy.deepcopy(self.critic_cost[i]))
                self.critic_cost_optimizer.append(torch.optim.Adam(self.critic_cost[i].parameters(), lr=self.lr))
            
            self.log_lambda = []
            self.lam_lag = []
            if self.adaptive_lambda:
                self.lambda_optimizer = []
                for i in range(self.cost_dim):
                    self.log_lambda.append(torch.zeros(1).to(self.device))
                    self.log_lambda[i].requires_grad = True
                    self.lam_lag.append(self.log_lambda[i].exp())
                    self.lambda_optimizer.append(torch.optim.Adam([self.log_lambda[i]], lr=self.lr))
            else:
                for i in range(self.cost_dim):
                    self.lam_lag.append(1.0)

    def choose_action(self, state, deterministic=False):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(args.device)
        a, _ = self.actor(state, deterministic, False)
        return a.data.cpu().numpy().flatten()

    def update(self, replay_buffer):
        batch_s, batch_a, batch_r, batch_c, batch_s_, batch_dw = replay_buffer.sample(self.mini_batch) 

        with torch.no_grad():
            batch_a_, log_pi_ = self.actor(batch_s_) 
            target_Q1, target_Q2 = self.critic_target(batch_s_, batch_a_)
            target_Q = batch_r + self.gamma * (1 - batch_dw) * (torch.min(target_Q1, target_Q2) - self.alpha * log_pi_)
            if self.use_lagrangian:
                target_Q_cost = []
                TD_target_cost = []
                for i in range(self.cost_dim):
                    target_Qc1, target_Qc2 = self.critic_cost_target[i](batch_s_, batch_a_)
                    target_Q_cost.append(torch.min(target_Qc1, target_Qc2))
                    TD_target_cost.append(batch_c[:,i+1].reshape([batch_c.shape[0],1]) + self.gamma * (1 - batch_dw) * target_Q_cost[i])
            if self.use_lyapunov:
                target_Lc1, target_Lc2 = self.critic_lyapunov_target(batch_s_, batch_a_)
                TD_target_lya = batch_c[:, 0].unsqueeze(1) + self.gamma * (1 - batch_dw) * torch.min(target_Lc1, target_Lc2)

        current_Q1, current_Q2 = self.critic(batch_s, batch_a)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        if self.use_lyapunov:
            current_Lc1, current_Lc2 = self.critic_lyapunov(batch_s, batch_a)
            lyapunov_loss = F.mse_loss(current_Lc1, TD_target_lya) + F.mse_loss(current_Lc2, TD_target_lya)
            self.lyapunov_critic_optimizer.zero_grad()
            lyapunov_loss.backward()
            self.lyapunov_critic_optimizer.step()
        
        if self.use_lagrangian:
            critic_cost_loss = []
            for i in range(self.cost_dim):
                current_Qc1, current_Qc2 = self.critic_cost[i](batch_s, batch_a)
                critic_cost_loss.append(F.mse_loss(current_Qc1, TD_target_cost[i]) + F.mse_loss(current_Qc2, TD_target_cost[i]))
                self.critic_cost_optimizer[i].zero_grad()
                critic_cost_loss[i].backward()
                self.critic_cost_optimizer[i].step()

        action_new, log_pi = self.actor(batch_s)
        Q1, Q2 = self.critic(batch_s, action_new)
        Q = torch.min(Q1, Q2)
        
        actor_loss = torch.mean(self.alpha * log_pi - Q)
        
        if self.use_lagrangian:
            penalty = []
            for i in range(self.cost_dim):
                qc1, qc2 = self.critic_cost[i](batch_s, action_new)
                penalty.append(torch.min(qc1, qc2) - self.cost_limit)
                actor_loss += torch.mean(self.lam_lag[i] * penalty[i].detach())
            
        if self.use_lyapunov:
            Lc1, Lc2 = self.critic_lyapunov(batch_s, batch_a)
            Lc = torch.min(Lc1, Lc2)
            a_, log_pi_ = self.actor(batch_s_)
            a_ += torch.normal(0, 0.1, size=a_.shape).to(args.device)
            Lc1_, Lc2_ = self.critic_lyapunov(batch_s_, a_)
            Lc_ = torch.min(Lc1_, Lc2_)
            Lyapunov_error = Lc_ - Lc + self.alpha3*batch_c[:,0]
            actor_loss += torch.mean(self.lam_lya * Lyapunov_error)
            
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        if self.adaptive_alpha:
            alpha_loss = -torch.mean(self.log_alpha.exp() * (log_pi + self.target_entropy).detach())
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()
        else:
            self.alpha += self.lr * log_pi.exp().mean().item() - self.target_entropy
        
        if self.use_lyapunov:
            if self.adaptive_lambda:
                lyapunov_loss = -torch.mean(self.log_lam_lya.exp() * Lyapunov_error.detach())
                self.lam_lya_optimizer.zero_grad()
                lyapunov_loss.backward()
                self.lam_lya_optimizer.step()
                self.lam_lya = self.log_lam_lya.exp().detach()
            else:
                self.lam_lya += self.lr * Lyapunov_error.detach().mean().item()
                self.lam_lya = max(self.lam_lya, 0.0)
        
        if self.use_lagrangian:
            if self.adaptive_lambda:
                lagrangian_loss = []
                for i in range(self.cost_dim):
                    lagrangian_loss.append(-torch.mean(self.log_lambda[i].exp() * penalty[i].detach()))
                    self.lambda_optimizer[i].zero_grad()
                    lagrangian_loss[i].backward()
                    self.lambda_optimizer[i].step()
                    self.lam_lag[i] = self.log_lambda[i].exp()
            else:
                for i in range(self.cost_dim):
                    self.lam_lag += self.lr * penalty[i].detach().mean().item()
                    self.lam_lag = max(self.lam_lag, 0.0)

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
        if self.use_lagrangian:
            for i in range(self.cost_dim):
                for param, target_param in zip(self.critic_cost[i].parameters(), self.critic_cost_target[i].parameters()):
                    target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
        if self.use_lyapunov:
            for param, target_param in zip(self.critic_lyapunov.parameters(), self.critic_lyapunov_target.parameters()):
                target_param.data.copy_(self.TAU * param.data + (1 - self.TAU) * target_param.data)
            
    def save_models(self):
        print("---saving models---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{self.date}.pl")
        torch.save(self.actor.state_dict(), agent_actor)

    def load_models(self, date):
        print("---loading model---")
        agent_actor = os.path.join(self.chkpt_dir, f"actor_{date}.pl")
        self.actor.load_state_dict(torch.load(agent_actor))
        

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


def train(args, env, agent, replay_buffer):
    save_args(args)
    start_time = time.time()
    writer = SummaryWriter(log_dir=args.log_dir)
    score_history = []
    best_score = -np.inf
    avg_score = 0
    total_steps = 0
    for episode in range(args.max_episodes):
        state = env.reset()[0]
        state = state.reshape(state.size)
        ep_reward = 0
        CostSpeed, CostCrash, CostEdge = 0, 0, 0
        v_record = []
        for step in range(args.max_steps):
            if total_steps < args.mini_batch:
                action = env.action_space.sample()
            else:
                action = agent.choose_action(state)
            next_state, reward, done, terminated, info = env.step(action)
            reward += info['speed']/100. if info['speed']<30. else 0.
            next_state = next_state.reshape(next_state.size)
            
            cost = call_cost(next_state, info)
            cost_record = cost[0] 
            
            replay_buffer.store(state, action, reward, cost_record, next_state, done) 
            
            ep_reward += reward
            CostSpeed += cost[0]
            CostCrash += cost[1]
            CostEdge += cost[2]
            state = next_state
            
            if total_steps >= args.mini_batch:
                agent.update(replay_buffer)
            total_steps += 1
            v_record.append(info['speed'])
            if done:
                break
        score_history.append(ep_reward)
        avg_score = np.mean(score_history[-50:])
        if avg_score > best_score:
            best_score = avg_score
            if episode > 100:
                folder_path = args.save_dir
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)
                agent.save_models()
        writer.add_scalar('Reward/reward', ep_reward, episode)
        writer.add_scalar('Reward/avg', avg_score, episode)
        writer.add_scalar('Cost/speed', CostSpeed, episode)
        writer.add_scalar('Cost/crash', CostCrash, episode)
        writer.add_scalar('Cost/Edge', CostEdge, episode)
        writer.add_scalar('Info/step', step+1, episode)
        writer.add_scalar('Info/v_avg', np.mean(v_record), episode)
        v_avg = np.mean(v_record)
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
    env.close()
    

def save_args(args):
    os.makedirs(args.log_dir, exist_ok=True)
    json_path = os.path.join(args.log_dir, "args.json")
    with open(json_path, 'w', encoding='utf-8') as f_json:
        json.dump(vars(args), f_json, ensure_ascii=False, indent=4, default=str)
        
        
if __name__ == '__main__':
    start_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    parser = argparse.ArgumentParser("Hyperparameters Setting for SAC-continuous")
    parser.add_argument("--max_episodes", type=int, default=int(4000), help=" Maximum number of training episodes")
    parser.add_argument("--max_steps", type=int, default=300, help="300")
    parser.add_argument("--evaluate_freq", type=float, default=50, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--batch_size", type=int, default=1e6, help="Batch size")
    parser.add_argument("--mini_batch", type=int, default=256, help="mini Batch size")
    parser.add_argument("--hidden_size", type=int, default=256, help="number of hidden layers")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--use_state_norm", type=bool, default=False, help="state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=False, help="reward scaling")
    parser.add_argument("--TAU", type=int, default=0.001, help="huper-parameter")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="set Adam epsilon=1e-5")
    args = parser.parse_args()
    config = config_setting()
    
    args.agent_name = "SACLag"
    args.env_name = "highway-v0"
    
    args.date = datetime.now().strftime("%Y%m%d%H%M")
    args.device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
    args.save_dir = "agent/{}/{}/{}".format(args.env_name, args.agent_name, args.date)
    args.log_dir = 'logs/{}/{}/{}'.format(args.env_name, args.agent_name, args.date)
    
    args.use_lyapunov = True
    args.use_lagrangian = False
    
    print(f"Agent: {args.agent_name}")
    print(f"Start date: {start_date}")
    
    env = gym.make(args.env_name, 
                #    render_mode='human',
                   config=config)
    
    args.state_dim = env.observation_space.shape[0]*env.observation_space.shape[1]
    args.action_dim = env.action_space.shape[0]
    args.cost_dim = 3
    args.cost_limit = 1e-2
    args.max_action = np.array([3.0, 0.3])
    
    agent = SAC(args)
    replay_buffer = ReplayBuffer(args)
    
    train(args, env, agent, replay_buffer)

    end_date = datetime.now().strftime("%Y.%m.%d %H:%M")
    print(f"Agent: {args.agent_name}")
    print(f"Start date: {start_date}")
    print(f"End date: {end_date}")