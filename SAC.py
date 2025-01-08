import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class Net(nn.Module):
    def __init__(self, state_size):
        super(Net, self).__init__()
        self.fc1_A = nn.Linear(state_size[0], 128)
        self.res_blocks_A = nn.ModuleList([self._residual_block(128) for _ in range(3)])
        self.fc2_A = nn.Linear(128, 128)

        self.fc1_B = nn.Linear(state_size[1], 128)
        self.res_blocks_B = nn.ModuleList([self._residual_block(128) for _ in range(3)])
        self.fc2_B = nn.Linear(128, 128)

        self.fc3 = nn.Linear(256, 256)
        self.fc4_mean = nn.Linear(256, 1)
        self.fc4_log_std = nn.Linear(256, 1)

    def _residual_block(self, in_channels):
        return nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels)
        )

    def forward(self, x_A, x_B):
        # 确保展平操作正确地应用于多维张量
        x_A = x_A.view(x_A.size(0), -1)  # Flatten
        x_B = x_B.view(x_B.size(0), -1)  # Flatten

        x_A = self.fc1_A(x_A)
        x_A = F.silu(x_A)
        for block in self.res_blocks_A:
            x_A = x_A + block(x_A)
        x_A = self.fc2_A(x_A)
        x_A = F.silu(x_A)

        x_B = self.fc1_B(x_B)
        x_B = F.silu(x_B)
        for block in self.res_blocks_B:
            x_B = x_B + block(x_B)
        x_B = self.fc2_B(x_B)
        x_B = F.silu(x_B)

        x = torch.cat((x_A, x_B), dim=1)
        x = self.fc3(x)
        x = F.silu(x)
        mean = self.fc4_mean(x)
        log_std = self.fc4_log_std(x)
        log_std = torch.clamp(log_std, min=-20, max=2)  # 限制对数标准差的范围
        return mean, log_std


class Actor(nn.Module):
    def __init__(self, state_size):
        super(Actor, self).__init__()
        self.net = Net(state_size)

    def forward(self, x_A, x_B):
        mean, log_std = self.net(x_A, x_B)
        return mean, log_std


class Critic(nn.Module):
    def __init__(self, state_size, action_size=1):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_size[0] + state_size[1] + action_size, 256)  # 256 是 state_net 的输出维度
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x_A, x_B, action):
        state = torch.cat((x_A, x_B), dim=-1)
        state = state.view(state.size(0), -1)
        action = action.view(action.size(0), -1)
        x = torch.cat((state, action), dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class SACAgent:
    def __init__(self, state_size, action_size=1, alpha=0.2, gamma=0.99, tau=0.005, learning_rate=0.0001):
        self.epsilon = 0.5
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = gamma  # 折扣率
        self.tau = tau  # 软更新参数
        self.alpha = alpha  # 初始温度参数
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.log_alpha = torch.tensor(np.log(alpha)).to(self.device)
        self.log_alpha.requires_grad = True

        # 目标熵
        self.target_entropy = -np.prod(action_size).item()  # 通常设置为 -action_size

        self.actor = Actor(self.state_size).to(self.device)
        self.critic_1 = Critic(self.state_size, self.action_size).to(self.device)
        self.critic_2 = Critic(self.state_size, self.action_size).to(self.device)
        self.target_critic_1 = Critic(self.state_size, self.action_size).to(self.device)
        self.target_critic_2 = Critic(self.state_size, self.action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_1_optimizer = optim.Adam(self.critic_1.parameters(), lr=learning_rate)
        self.critic_2_optimizer = optim.Adam(self.critic_2.parameters(), lr=learning_rate)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

        self.update_target_model()

    def update_target_model(self):
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state, eval_mode=False):

        state = [torch.tensor(s, dtype=torch.float32).to(self.device) for s in state]
        with torch.no_grad():
            A = state[0]
            B = state[1]
            mean, log_std = self.actor(A, B)
            std = torch.exp(log_std)
            normal = torch.distributions.Normal(mean, std)
            z = normal.rsample()  # 使用重参数化技巧
            action = torch.tanh(z)

            if eval_mode:
                return action.cpu().numpy()[0]

            log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
            log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def replay(self, batch_size):
        # 随机采样一个批次
        minibatch = random.sample(self.memory, batch_size)
        state_batch_0, state_batch_1, action_batch, reward_batch, next_state_batch_0, next_state_batch_1 = [], [], [], [], [], []

        for state, action, reward, next_state in minibatch:
            state = [torch.tensor(s, dtype=torch.float32).to(self.device) for s in state]
            next_state = [torch.tensor(s, dtype=torch.float32).to(self.device) for s in next_state]
            reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
            state_batch_0.append(state[0].cpu().numpy())
            state_batch_1.append(state[1].cpu().numpy())
            action_batch.append(action.cpu().numpy())
            reward_batch.append(reward)
            next_state_batch_0.append(next_state[0].cpu().numpy())
            next_state_batch_1.append(next_state[1].cpu().numpy())

        state_batch_0 = np.vstack(state_batch_0)
        state_batch_1 = np.vstack(state_batch_1)
        action_batch = np.vstack(action_batch)
        next_state_batch_0 = np.vstack(next_state_batch_0)
        next_state_batch_1 = np.vstack(next_state_batch_1)

        # Convert to tensors and move to device
        state_batch_0 = torch.tensor(state_batch_0, dtype=torch.float32).to(self.device)
        state_batch_1 = torch.tensor(state_batch_1, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(self.device)
        next_state_batch_0 = torch.tensor(next_state_batch_0, dtype=torch.float32).to(self.device)
        next_state_batch_1 = torch.tensor(next_state_batch_1, dtype=torch.float32).to(self.device)

        reward_batch = torch.stack(reward_batch).unsqueeze(1).to(self.device)

        # 计算下一个状态的动作和对数概率
        with torch.no_grad():
            next_actions, next_log_probs = self.act([next_state_batch_0, next_state_batch_1])
            q1_next = self.target_critic_1(next_state_batch_0, next_state_batch_1, next_actions)
            q2_next = self.target_critic_2(next_state_batch_0, next_state_batch_1, next_actions)
            min_q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            target_q_values = reward_batch + self.gamma * min_q_next

        # 更新Critic
        q1 = self.critic_1(state_batch_0, state_batch_1, action_batch)
        q2 = self.critic_2(state_batch_0, state_batch_1, action_batch)
        critic_1_loss = F.mse_loss(q1, target_q_values)
        critic_2_loss = F.mse_loss(q2, target_q_values)

        self.critic_1_optimizer.zero_grad()
        critic_1_loss.backward()
        self.critic_1_optimizer.step()

        self.critic_2_optimizer.zero_grad()
        critic_2_loss.backward()
        self.critic_2_optimizer.step()

        # 更新Actor
        new_actions, log_probs = self.act([state_batch_0, state_batch_1])
        q1_new = self.critic_1(state_batch_0, state_batch_1, new_actions)
        q2_new = self.critic_2(state_batch_0, state_batch_1, new_actions)
        min_q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - min_q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # 更新α
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = self.log_alpha.exp()

        # 软更新目标网络
        self.update_target_model()

    def load(self, actor_name):
        self.actor.load_state_dict(torch.load(actor_name, map_location=self.device))

    def save(self, actor_name):
        torch.save(self.actor.state_dict(), actor_name)
