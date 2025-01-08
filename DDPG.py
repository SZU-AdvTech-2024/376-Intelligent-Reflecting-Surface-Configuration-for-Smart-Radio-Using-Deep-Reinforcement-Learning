import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class Net(nn.Module):
    def __init__(self, state_size, action_size):
        super(Net, self).__init__()
        self.fc1_A = nn.Linear(state_size[0], 128)
        self.res_blocks_A = nn.ModuleList([self._residual_block(128) for _ in range(3)])
        self.fc2_A = nn.Linear(128, 128)

        self.fc1_B = nn.Linear(state_size[1], 128)
        self.res_blocks_B = nn.ModuleList([self._residual_block(128) for _ in range(3)])
        self.fc2_B = nn.Linear(128, 128)

        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, action_size)

    def _residual_block(self, in_channels):
        return nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_channels)
        )

    def forward(self, x_A, x_B):
        x_A = x_A.view(x_A.size(0), -1)  # Flatten
        x_A = self.fc1_A(x_A)
        x_A = F.silu(x_A)
        for block in self.res_blocks_A:
            x_A = x_A + block(x_A)
        x_A = self.fc2_A(x_A)
        x_A = F.silu(x_A)

        x_B = x_B.view(x_B.size(0), -1)  # Flatten
        x_B = self.fc1_B(x_B)
        x_B = F.silu(x_B)
        for block in self.res_blocks_B:
            x_B = x_B + block(x_B)
        x_B = self.fc2_B(x_B)
        x_B = F.silu(x_B)

        x = torch.cat((x_A, x_B), dim=1)
        x = self.fc3(x)
        x = F.silu(x)
        x = self.fc4(x)
        return x


class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.net = Net(state_size, action_size)

    def forward(self, x_A, x_B):
        x = self.net(x_A, x_B)
        x = F.tanh(x)  # 使用 tanh 激活函数将输出限制在 [-1, 1]
        x = x * 4  # 将输出从 [-1, 1] 映射到 [-4, 4]
        return x


class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()
        self.state_net = Net(state_size, action_size)
        self.action_net = nn.Linear(action_size, action_size)
        self.fc1 = nn.Linear(action_size + action_size, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x_A, x_B, action):
        state_features = self.state_net(x_A, x_B)
        action_features = self.action_net(action)
        x = torch.cat((state_features, action_features), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class DDPGAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # 折扣率
        self.tau = 0.001  # 软更新参数
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate_actor = 0.001
        self.learning_rate_critic = 0.001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.actor = Actor(self.state_size, self.action_size).to(self.device)
        self.target_actor = Actor(self.state_size, self.action_size).to(self.device)
        self.critic = Critic(self.state_size, self.action_size).to(self.device)
        self.target_critic = Critic(self.state_size, self.action_size).to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.learning_rate_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.learning_rate_critic)

        self.update_target_model()

    def update_target_model(self):
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        state = [torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device) for s in state]
        with torch.no_grad():
            action = self.actor(*state).cpu().numpy()[0]

        # Add Gaussian noise to the action
        noise = np.random.normal(0, 0.1, size=self.action_size)
        action += noise
        action = np.clip(action, -3.5, 3.5)  # 确保动作在 [-4, 4] 范围内

        return action, 1

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch_0, state_batch_1, action_batch, reward_batch, next_state_batch_0, next_state_batch_1 = [], [], [], [], [], []

        for state, action, reward, next_state in minibatch:
            state_batch_0.append(state[0])
            state_batch_1.append(state[1])
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch_0.append(next_state[0])
            next_state_batch_1.append(next_state[1])

        state_batch_0 = torch.tensor(np.array(state_batch_0), dtype=torch.float32).to(self.device)
        state_batch_1 = torch.tensor(np.array(state_batch_1), dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(np.array(action_batch), dtype=torch.float32).to(self.device)
        reward_batch = torch.tensor(np.array(reward_batch), dtype=torch.float32).to(self.device)
        next_state_batch_0 = torch.tensor(np.array(next_state_batch_0), dtype=torch.float32).to(self.device)
        next_state_batch_1 = torch.tensor(np.array(next_state_batch_1), dtype=torch.float32).to(self.device)

        # Compute the target Q values
        with torch.no_grad():
            next_actions = self.target_actor(next_state_batch_0, next_state_batch_1)
            next_q_values = self.target_critic(next_state_batch_0, next_state_batch_1, next_actions)
            target_q_values = reward_batch + self.gamma * next_q_values

        # Compute the current Q values
        current_q_values = self.critic(state_batch_0, state_batch_1, action_batch)

        # Compute the critic loss
        critic_loss = F.mse_loss(current_q_values, target_q_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Compute the actor loss
        predicted_actions = self.actor(state_batch_0, state_batch_1)
        actor_loss = -self.critic(state_batch_0, state_batch_1, predicted_actions).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_model()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return critic_loss.item()

    def load(self, actor_name):
        self.actor.load_state_dict(torch.load(actor_name, map_location=self.device))

    def save(self, actor_name):
        torch.save(self.actor.state_dict(), actor_name)
