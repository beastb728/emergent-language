import numpy as np
import torch

from environment.communication_world import (
    CommunicationWorld
)

from rl.dqn import DQNAgent
from rl.random_protocol import RandomFixedProtocol


EPISODES = 3000

TARGET_UPDATE = 25


world = CommunicationWorld()

protocol = RandomFixedProtocol()

receiver = DQNAgent(
    state_size=6,
    action_size=5
)

success_history = []


protocol.show()

print()
print("Training receiver...")
print()


for episode in range(
    1,
    EPISODES + 1
):

    world.reset()

    # --------------------------------
    # FIXED SENDER
    # --------------------------------

    message = protocol.encode(
        world.target_id
    )

    # --------------------------------
    # RECEIVER OBSERVATION
    # --------------------------------

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

    receiver.decay_epsilon()

    if episode % TARGET_UPDATE == 0:

        receiver.update_target_network()

    success_history.append(
        int(success)
    )

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


print()
print("Training complete.")

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_stage_a.pt"
)

print(
    "Saved: models/receiver_stage_a.pt"
)