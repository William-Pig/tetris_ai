import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DQNCNN(nn.Module):
    def __init__(self, input_shape, num_actions):
        """
        CNN Architecture
        - 3 convolutional layers: 
            - layer 1 sees 3x3 (local)
            - layer 2 sees 5x5 (mid-range), see from layer 1
            - layer 3 sees 7x7 (long-range), in case of big gaps
        - 2 fully connected layers:
            - layer 1 builds hidden features
            - layer 2 gives output based on the hidden features
        """
        super(DQNCNN, self).__init__()
        c, h, w = input_shape  # channels, height of each channel, width of each channel

        self.conv1 = nn.Conv2d(c, 32, kernel_size=3, stride=1, padding=1)  # keep same size
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.gap    = nn.AdaptiveAvgPool2d(1)   # -> [B,64,1,1]
        self.fc1    = nn.Linear(64, 512)  # always 64 in_features
        self.fc2    = nn.Linear(512, num_actions)

    def forward(self, x):
        """Propagates prediction forward in the NN"""
        x = x.float()  # ensure float32
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.gap(x).view(x.size(0), -1)  # flatten, from convolution to 
        x = F.relu(self.fc1(x))
        return self.fc2(x)  # Q-values for all actions


class DQNAgent:
    def __init__(self, state_shape, num_actions, device='cpu'):
        self.state_shape = state_shape  # (# channel, board height, board width) e.g. (15, 20, 10)
        self.num_actions = num_actions  # number of available actions
        self.device = device  # training device, cpu or gpu

        # Hyperparameters
        self.gamma = 0.5
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001

        self.memory = deque(maxlen=10000)  # Replay buffer

        # Models
        self.model = DQNCNN(state_shape, num_actions).to(self.device)  # create the main DQN
        self.target_model = DQNCNN(state_shape, num_actions).to(self.device)  # Frozen DQN, for finding stable target
        self.target_model.load_state_dict(self.model.state_dict())  # Copies weights from main model into frozen model
        self.target_model.eval()  # since we use it for inference
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # Adam optimizer for learning

    def memorize(self, state, action, reward, next_state, done):
        # Detach states when storing. Each state is a torch tensor of shape e.g. (15, 20, 10), see self.state_shape
        self.memory.append((state.clone().detach(), action, reward, next_state.clone().detach(), done))

    def act(self, state, valid_action_ids):
        """Epsilon-greedy action selection using only valid actions"""
        if np.random.rand() < self.epsilon:
            return random.choice(valid_action_ids)

        self.model.eval()  # put to inference as we are inferecing the best action from existing model
        with torch.no_grad():
            state = state.unsqueeze(0).to(self.device)  # Add batch dimension: e.g. (15,20,10) -> (1,15,20,10)
            q_values = self.model(state).squeeze(0)  # prediction step, Shape: (num_actions,)
            valid_qs = q_values[valid_action_ids]  # subset to check only the valid set of actions
            best_idx = torch.argmax(valid_qs).item()  # find the best action
            return valid_action_ids[best_idx]

    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.target_model.load_state_dict(self.model.state_dict())

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return None  # not enough samples yet

        minibatch = random.sample(self.memory, batch_size)

        # organize batch into tensors. B: batch size, C: channel size, H: board height, W: board width
        states = torch.stack([s for (s, _, _, _, _) in minibatch]).to(self.device) # [B, C, H, W]
        actions = torch.tensor([a for (_, a, _, _, _) in minibatch], dtype=torch.long).to(self.device)  # [B]
        rewards = torch.tensor([r for (_, _, r, _, _) in minibatch], dtype=torch.float32).to(self.device)  # [B]
        next_states = torch.stack([s_ for (_, _, _, s_, _) in minibatch]).to(self.device)    # [B, C, H, W]
        dones = torch.tensor([d for (_, _, _, _, d) in minibatch], dtype=torch.float32).to(self.device)   # [B]

        # Predicted Q(s, a)
        self.model.train()
        q_pred = self.model(states)  # [B, num_actions], compute Q-values for all actions in each state
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B], .gather picks Q-value for actual action taken

        # Target Q-values
        with torch.no_grad():
            q_next = self.target_model(next_states)  # [B, num_actions], find all future q-values
            q_next_max = torch.max(q_next, dim=1)[0]  # [B], max Q-value for next state
            q_target = rewards + self.gamma * q_next_max * (1 - dones)  # [B], if done target = rewards

        #  Loss and Backprop
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_pred, q_target)

        # Updates the main model's weights 
        self.optimizer.zero_grad()
        loss.backward()  # back propagation
        self.optimizer.step()

        # Decay epsilon
        self.update_epsilon()

        return loss.item()
