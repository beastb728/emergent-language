import random
import numpy as np
import torch

from environment.compositional_world import CompositionalWorld
from rl.dqn import DQNAgent


NUM_EPISODES = 5000


# --------------------------------
# Random hidden protocol
# --------------------------------

messages = list(range(16))
random.shuffle(messages)

protocol = {}

message_index = 0

for x_region in range(4):
    for y_region in range(4):

        protocol[
            (x_region, y_region)
        ] = messages[message_index]

        message_index += 1


print()
print("Hidden 16-message protocol:")
print()

for x_region in range(4):

    row = []

    for y_region in range(4):

        message = protocol[
            (x_region, y_region)
        ]

        row.append(
            f"{message:02d}"
        )

    print(
        f"X{x_region}: "
        + "  ".join(row)
    )


world = CompositionalWorld()

receiver = DQNAgent(
    state_size=18,
    action_size=5
)


success_count = 0


for episode in range(NUM_EPISODES):

    world.reset()

    target_key = (
        world.x_region,
        world.y_region
    )

    message = protocol[target_key]

    message_one_hot = np.zeros(
        16,
        dtype=np.float32
    )

    message_one_hot[message] = 1.0

    state = np.concatenate(
        [
            world.agent_b / 10.0,
            message_one_hot
        ]
    ).astype(np.float32)

    done = False

    while not done:

        action = receiver.choose_action(
            state
        )

        reward, done, reached = world.step(
            action
        )

        next_state = np.concatenate(
            [
                world.agent_b / 10.0,
                message_one_hot
            ]
        ).astype(np.float32)

        receiver.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        receiver.train_step()

        state = next_state

    receiver.decay_epsilon()

    if episode % 100 == 0:

        receiver.update_target_network()

    if reached:
        success_count += 1

    if episode % 100 == 0:

        print(
            f"Episode {episode} | "
            f"Success: "
            f"{success_count}% | "
            f"Epsilon: "
            f"{receiver.epsilon:.3f}"
        )

        success_count = 0


torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_compositional_staged.pt"
)

np.save(
    "models/compositional_protocol.npy",
    protocol
)

print()
print("Stage A complete.")
print(
    "Saved models/receiver_compositional_staged.pt"
)
print(
    "Saved models/compositional_protocol.npy"
)