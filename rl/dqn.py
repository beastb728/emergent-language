import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class QNetwork(nn.Module):

    def __init__(self, state_size, action_size):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            nn.ReLU(),

            nn.Linear(64, 64),
            nn.ReLU(),

            nn.Linear(64, action_size)
        )

    def forward(self, state):
        return self.network(state)


class DQNAgent:

    def __init__(
        self,
        state_size=2,
        action_size=5,
        learning_rate=1e-3,
        gamma=0.99,
    ):

        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma

        self.device = torch.device("cpu")

        self.policy_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net = QNetwork(
            state_size,
            action_size
        ).to(self.device)

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

        self.target_net.eval()

        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=learning_rate
        )

        self.memory = deque(maxlen=10_000)

        self.batch_size = 64

        self.epsilon = 1.0
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.995

    def choose_action(self, state):

        if random.random() < self.epsilon:
            return random.randrange(self.action_size)

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        with torch.no_grad():

            q_values = self.policy_net(
                state_tensor
            )

        return q_values.argmax(dim=1).item()

    def remember(
        self,
        state,
        action,
        reward,
        next_state,
        done
    ):

        self.memory.append(
            (
                state,
                action,
                reward,
                next_state,
                done
            )
        )

    def train_step(self):

        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(
            self.memory,
            self.batch_size
        )

        states, actions, rewards, next_states, dones = zip(
            *batch
        )

        states = torch.tensor(
            np.array(states),
            dtype=torch.float32,
            device=self.device
        )

        actions = torch.tensor(
            actions,
            dtype=torch.long,
            device=self.device
        ).unsqueeze(1)

        rewards = torch.tensor(
            rewards,
            dtype=torch.float32,
            device=self.device
        )

        next_states = torch.tensor(
            np.array(next_states),
            dtype=torch.float32,
            device=self.device
        )

        dones = torch.tensor(
            dones,
            dtype=torch.float32,
            device=self.device
        )

        current_q = self.policy_net(states).gather(
            1,
            actions
        ).squeeze(1)

        with torch.no_grad():

            next_q = self.target_net(
                next_states
            ).max(dim=1)[0]

            target_q = rewards + (
                1 - dones
            ) * self.gamma * next_q

        loss = nn.functional.smooth_l1_loss(
            current_q,
            target_q
        )

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()

        return loss.item()

    def update_target_network(self):

        self.target_net.load_state_dict(
            self.policy_net.state_dict()
        )

    def decay_epsilon(self):

        self.epsilon = max(
            self.epsilon_min,
            self.epsilon * self.epsilon_decay
        )