import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # 折扣率
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.999
        self.learning_rate = 0.001
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)
        self.target_model = self._build_model().to(self.device)
        self.update_target_model()

    def _build_model(self):
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

        return Net(self.state_size, self.action_size)

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def memorize(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            act_type = 'random'
            return random.randrange(self.action_size), act_type
        state = [torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device) for s in state]
        with torch.no_grad():
            act_values = self.model(*state).cpu().numpy()
        act_type = 'RL'
        return np.argmax(act_values[0]), act_type

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        state_batch_0, state_batch_1, action_batch, target_batch = [], [], [], []

        for state, action, reward, next_state in minibatch:
            next_state = [torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device) for s in next_state]
            with torch.no_grad():
                ActionIndex = np.argmax(self.model(*next_state).cpu().numpy()[0])
                target = reward + self.gamma * self.target_model(*next_state).cpu().numpy()[0][ActionIndex]
            state = [torch.tensor(s, dtype=torch.float32).unsqueeze(0).to(self.device) for s in state]
            with torch.no_grad():
                target_f = self.model(*state).cpu().detach().numpy()[0]
            target_f[action] = target
            state_batch_0.append(state[0].cpu().numpy())
            state_batch_1.append(state[1].cpu().numpy())
            action_batch.append(action)
            target_batch.append(target_f)

        InputA_data = np.vstack(state_batch_0)
        InputB_data = np.vstack(state_batch_1)
        action_batch = np.array(action_batch)
        target_batch = np.vstack(target_batch)

        # Convert to tensors and move to device
        InputA_data = torch.tensor(InputA_data, dtype=torch.float32).to(self.device)
        InputB_data = torch.tensor(InputB_data, dtype=torch.float32).to(self.device)
        action_batch = torch.tensor(action_batch, dtype=torch.long).to(self.device)
        target_batch = torch.tensor(target_batch, dtype=torch.float32).to(self.device)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        self.model.train()
        optimizer.zero_grad()
        outputs = self.model(InputA_data, InputB_data)
        actions = action_batch.unsqueeze(1).expand(-1, outputs.size(1))
        loss = criterion(outputs.gather(1, actions), target_batch)
        loss.backward()
        optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

    def load(self, name):
        self.model.load_state_dict(torch.load(name, map_location=self.device))

    def save(self, name):
        torch.save(self.model.state_dict(), name)