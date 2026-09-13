import os

import numpy as np
import torch

from environment.communication_world import (
    CommunicationWorld
)

from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES = 5000

TARGET_UPDATE = 25

world = CommunicationWorld()

sender = Sender()

receiver = DQNAgent(
    state_size=6,
    action_size=5
)

os.makedirs(
    "models",
    exist_ok=True
)

success_history = []


for episode in range(
    1,
    EPISODES + 1
):

    world.reset()

    # -------------------------
    # A SENDS MESSAGE
    # -------------------------

    sender_state = world.sender_state()

    message, log_probability = (
        sender.choose_message(sender_state)
    )

    # -------------------------
    # B RECEIVES MESSAGE
    # -------------------------

    state = world.receiver_state(
        message
    )

    total_reward = 0.0

    success = False

    for step in range(
        world.max_steps
    ):

        action = receiver.choose_action(
            state
        )

        reward, done, reached = (
            world.step(action)
        )

        next_state = world.receiver_state(
            message
        )

        receiver.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        receiver.train_step()

        state = next_state

        total_reward += reward

        if reached:

            success = True
            break

        if done:
            break

    # -------------------------
    # TRAIN SENDER
    # -------------------------

    reward_tensor = torch.tensor(
        total_reward,
        dtype=torch.float32
    )

    sender_loss = (
        -log_probability * reward_tensor
    )

    sender.optimizer.zero_grad()

    sender_loss.backward()

    sender.optimizer.step()

    # -------------------------
    # UPDATE RECEIVER
    # -------------------------

    receiver.decay_epsilon()

    if episode % TARGET_UPDATE == 0:

        receiver.update_target_network()

    success_history.append(
        int(success)
    )

    # -------------------------
    # LOGGING
    # -------------------------

    if episode % 100 == 0:

        recent_success = (
            np.mean(
                success_history[-100:]
            )
            * 100
        )

        print(
            f"Episode {episode:5d} | "
            f"Success: {recent_success:5.1f}% | "
            f"Reward: {total_reward:7.2f} | "
            f"Epsilon: {receiver.epsilon:.3f}"
        )

    # -------------------------
    # SAVE
    # -------------------------

    if episode % 500 == 0:

        torch.save(
            sender.network.state_dict(),
            f"models/sender_{episode}.pt"
        )

        torch.save(
            receiver.policy_net.state_dict(),
            f"models/receiver_{episode}.pt"
        )


torch.save(
    sender.network.state_dict(),
    "models/sender_final.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_final.pt"
)

print()
print("Communication training complete.")