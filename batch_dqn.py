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
    def __init__(self, state_shape, num_actions, device='cpu',
                 alpha=0.001, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_shape = state_shape  # (# channel, board height, board width) e.g. (15, 20, 10)
        self.num_actions = num_actions  # number of available actions
        self.device = device  # training device, cpu or gpu

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=10_000)  # Replay buffer

        # Models
        self.model = DQNCNN(state_shape, num_actions).to(self.device)  # create the main DQN
        self.target_model = DQNCNN(state_shape, num_actions).to(self.device)  # Frozen DQN, for finding stable target
        self.target_model.load_state_dict(self.model.state_dict())  # Copies weights from main model into frozen model
        self.target_model.eval()  # since we use it for inference
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)  # Adam optimizer for learning

    def memorize(self, state, action, reward, next_state, done):
        # Detach states when storing. Each state is a torch tensor of shape e.g. (15, 20, 10), see self.state_shape
        self.memory.append((state.detach().cpu(), action, reward, next_state.detach().cpu(), done))

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

    def replay(self, batch_size):
        """
        Samples mini-batch of past experience from memory
        Use them to train NN by minimizing the squared error between predicted Q-values and target Q-values
        """
        if len(self.memory) < batch_size:
            return None  # not enough samples yet

        minibatch = random.sample(self.memory, batch_size)

        # organize batch into tensors. B: batch size, C: channel size, H: board height, W: board width
        states = torch.stack([s.to(self.device) for (s, _, _, _, _) in minibatch]) # [B, C, H, W]
        actions = torch.tensor([a for (_, a, _, _, _) in minibatch], dtype=torch.long, device=self.device)  # [B]
        rewards = torch.tensor([r for (_, _, r, _, _) in minibatch], dtype=torch.float32, device=self.device)  # [B]
        next_states = torch.stack([s_.to(self.device) for (_, _, _, s_, _) in minibatch])    # [B, C, H, W]
        dones = torch.tensor([d for (_, _, _, _, d) in minibatch], dtype=torch.float32, device=self.device)   # [B]

        # Predicted Q(s, a), from main model
        q_pred = self.model(states)  # [B, num_actions], compute Q-values for all actions in each state
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B], .gather picks Q-value for actual action taken

        # Target Q-values, from frozen target model
        with torch.no_grad():
            q_next = self.target_model(next_states)  # [B, num_actions], find all future q-values
            q_next_max = q_next.max(dim=1)[0]  # [B], max Q-value for next state
            q_target = rewards + self.gamma * q_next_max * (1 - dones)  # [B], if done target = rewards

        #  Loss and Backprop
        loss_fn = nn.MSELoss()
        loss = loss_fn(q_pred, q_target)

        # Updates the main model's weights 
        self.optimizer.zero_grad()
        loss.backward()  # back propagation
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # rescales gradients s.t. global l2-norm<=max_norm 
        self.optimizer.step()

        # Decay epsilon
        self.update_epsilon()

        return loss.item()

    def save(self, path, extra=None):
        """Save the agent, for future training"""
        checkpoint = {
            "model": self.model.state_dict(),  # main model
            "target": self.target_model.state_dict(),  # frozen model
            "optimizer": self.optimizer.state_dict(),
            "epsilon": self.epsilon,  # maintain epsilon for future training
            "memory": list(self.memory),   # convert deque to list
        }
        if extra:  # episode_rewards, episode_num, etc.
            checkpoint.update(extra)

        torch.save(checkpoint, path)

    def load_from_dict(self, checkpoint_dict):
        """Takes in a checkpoint dict, load a trained agent to resume training/do analysis etc"""
        self.model.load_state_dict(checkpoint_dict["model"])
        self.target_model.load_state_dict(checkpoint_dict["target"])
        self.optimizer.load_state_dict(checkpoint_dict["optimizer"])
        self.epsilon = checkpoint_dict["epsilon"]
        self.memory = deque(checkpoint_dict["memory"], maxlen=self.memory.maxlen)
