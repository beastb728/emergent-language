import numpy as np
import torch

from environment.communication_world import (
    CommunicationWorld
)

from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES = 5000


world = CommunicationWorld()

sender = Sender()

receiver = DQNAgent(
    state_size=6,
    action_size=5
)

receiver.policy_net.load_state_dict(
    torch.load(
        "models/receiver_stage_a.pt",
        map_location="cpu"
    )
)

# Freeze receiver
receiver.epsilon = 0.0

for parameter in receiver.policy_net.parameters():
    parameter.requires_grad = False


success_history = []

print()
print("Stage B: training sender against frozen receiver")
print()


for episode in range(
    1,
    EPISODES + 1
):

    world.reset()

    # --------------------------------
    # A OBSERVES THE WORLD
    # --------------------------------

    sender_state = world.sender_state()

    # --------------------------------
    # A CHOOSES MESSAGE
    # --------------------------------

    message, log_probability = (
        sender.choose_message(sender_state)
    )

    # --------------------------------
    # B RECEIVES MESSAGE
    # --------------------------------

    state = world.receiver_state(
        message
    )

    total_reward = 0.0
    success = False

    # --------------------------------
    # B NAVIGATES
    # --------------------------------

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

        state = next_state

        total_reward += reward

        if reached:

            success = True
            break

        if done:
            break

    # --------------------------------
    # TRAIN A
    # --------------------------------

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

    success_history.append(
        int(success)
    )

    # --------------------------------
    # LOGGING
    # --------------------------------

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
            f"Reward: {total_reward:7.2f}"
        )


torch.save(
    sender.network.state_dict(),
    "models/sender_stage_b.pt"
)

print()
print("Stage B complete.")
print("Saved: models/sender_stage_b.pt")