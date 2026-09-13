import torch
import torch.nn as nn
from torch.distributions import Categorical


class SenderNetwork(nn.Module):

    def __init__(
        self,
        state_size=6,
        message_size=4
    ):

        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),

            nn.Linear(32, 32),
            nn.ReLU(),

            nn.Linear(32, message_size)
        )

    def forward(self, state):

        return self.network(state)


class Sender:

    def __init__(self):

        self.device = torch.device("cpu")

        self.network = SenderNetwork().to(
            self.device
        )

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=1e-3
        )

    def choose_message(self, state):

        state_tensor = torch.tensor(
            state,
            dtype=torch.float32,
            device=self.device
        ).unsqueeze(0)

        logits = self.network(
            state_tensor
        )

        distribution = Categorical(
            logits=logits
        )

        message = distribution.sample()

        log_probability = distribution.log_prob(
            message
        )

        return (
            message.item(),
            log_probability
        )