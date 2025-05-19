import random
from collections import deque

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class FeatureMLP(nn.Module):
    def __init__(self, num_actions, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, num_actions)
        )

    def forward(self, x):  # x shape: [B, F]
        return self.net(x)



class FeatureMLPAgent:
    def __init__(self, num_actions, feature_dim, device='cpu',
                 alpha=0.001, gamma=0.9,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.num_actions = num_actions  # number of available actions
        self.feature_dim = feature_dim  # f
        self.device = device  # training device, cpu or gpu

        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        self.memory = deque(maxlen=30_000)  # Replay buffer

        # Models
        self.model = FeatureMLP(num_actions, feature_dim).to(self.device)  # create the main DQN
        self.target_model = FeatureMLP(num_actions, feature_dim).to(self.device)  # Frozen DQN
        self.target_model.load_state_dict(self.model.state_dict())  # Copies weights from main model into frozen model
        self.target_model.eval()  # since we use it for inference
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.alpha)  # Adam optimizer for learning

    def memorize(self, state, action, reward, next_state, done, next_valid_ids):
        # Detach states when storing
        state = state[1].to(self.device)  # only get feature, not board
        next_state = next_state[1].to(self.device)
        mask = torch.zeros(self.num_actions, dtype=torch.bool)  # mask for valid actions
        mask[next_valid_ids] = True  # (on CPU)
        self.memory.append((state.detach(),
                            action,
                            reward, 
                            next_state.detach(),
                            done,
                            mask
                            ))

    def _ids_to_mask(self, ids_batch):
        """
        Converts every list of IDs into a 0/1 mask tensor once per sample
        ids_batch = list[list[int]], length = B
        """
        B = len(ids_batch)
        mask = torch.zeros((B, self.num_actions),      # bool mask on device
                        dtype=torch.bool,
                        device=self.device)
        for i, ids in enumerate(ids_batch):
            mask[i, ids] = True
        return mask


    def act(self, state, valid_action_ids):
        """Epsilon-greedy action selection using only valid actions"""
        if len(valid_action_ids) == 0:
            raise RuntimeError("No valid actions available")

        state = state[1].to(self.device).unsqueeze(0)

        if np.random.rand() < self.epsilon:
            return random.choice(valid_action_ids)

        self.model.eval()  # put to inference as we are inferecing the best action from existing model
        with torch.no_grad():
            q_values = self.model(state).squeeze(0)  # prediction step, Shape: (num_actions,)
            valid_ids_tensor = torch.tensor(valid_action_ids, device=q_values.device, dtype=torch.long)
            valid_qs = q_values[valid_ids_tensor]  # subset to find only valid actions
            best_idx = torch.argmax(valid_qs).item()
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

        # unpack
        states, actions, rewards, next_states, dones, masks = zip(*minibatch)
        # organize batch into tensors. B: batch size, C: channel size, H: height, W: width, F: hand-crafted features
        states = torch.stack(states).to(self.device) # [B, F]
        actions = torch.tensor(actions,  dtype=torch.long, device=self.device)  # B
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)  # B
        next_states = torch.stack(next_states).to(self.device)  # [B, F]
        dones = torch.tensor(dones,   dtype=torch.float32, device=self.device)  # B
        valid_mask = torch.stack(masks).to(self.device)


        # Predicted Q(s, a), from main model
        q_pred = self.model(states)  # [B, num_actions], compute Q-values for all actions in each state
        q_pred = q_pred.gather(1, actions.unsqueeze(1)).squeeze(1)  # [B], .gather picks Q-value for actual action taken

        # Target Q-values, from frozen target model
        with torch.no_grad():
            # Single DQN, outdated, need to update
            # q_next = self.target_model(next_states)  # [B, num_actions], find all future q-values
            # q_next_max = q_next.max(dim=1)[0]  # [B], max Q-value for next state
            # Double DQN
            q_next_all = self.target_model(next_states)          # [B, |A|]
            q_next_all[~valid_mask] = -1e9                       # effectively −∞
            best_a = torch.argmax(q_next_all, 1, keepdim=True)
            q_next_max = q_next_all.gather(1, best_a).squeeze(1)
            q_target = rewards + self.gamma * q_next_max * (1 - dones)  # [B], if done target = rewards

        #  Loss
        loss = F.mse_loss(q_pred, q_target)

        # Updates the main model's weights 
        self.optimizer.zero_grad()
        loss.backward()  # back propagation
        # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)  # rescales gradients s.t. global l2-norm<=max_norm 
        self.optimizer.step()

        # Decay epsilon, update in train loop, not here
        # self.update_epsilon()

        # GPU usage check
        # print("Feature batch device:", states.device)
        # print("Model device:", next(self.model.parameters()).device)


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
