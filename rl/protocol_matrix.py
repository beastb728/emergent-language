import numpy as np
import torch

from environment.communication_world import CommunicationWorld
from rl.dqn import DQNAgent


EPISODES_PER_CELL = 250


world = CommunicationWorld()

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

receiver.epsilon = 0.0


messages = [0, 1, 2, 3]

results = np.zeros((4, 4))


print()
print("Protocol matrix evaluation")
print()
print(
    "Rows = actual target | "
    "Columns = forced message"
)
print()


for target_id in range(4):

    for message in messages:

        successes = 0

        for episode in range(EPISODES_PER_CELL):

            world.reset()

            # Force the target for this evaluation.
            world.target_id = target_id
            world.target = (
                world.target_locations[target_id].copy()
            )

            state = world.receiver_state(message)

            for step in range(world.max_steps):

                action = receiver.choose_action(state)

                reward, done, reached = world.step(action)

                state = world.receiver_state(message)

                if reached:
                    successes += 1
                    break

                if done:
                    break

        success_rate = (
            successes / EPISODES_PER_CELL * 100
        )

        results[target_id, message] = success_rate


print("             Message")
print("Target       00      01      10      11")
print("-----------------------------------------")

for target_id in range(4):

    values = results[target_id]

    print(
        f"T{target_id}       "
        f"{values[0]:5.1f}%  "
        f"{values[1]:5.1f}%  "
        f"{values[2]:5.1f}%  "
        f"{values[3]:5.1f}%"
    )


print()
print("Receiver's best interpretation of each message:")
print()

for message in messages:

    best_target = int(
        np.argmax(results[:, message])
    )

    print(
        f"{format(message, '02b')} -> T{best_target} "
        f"({results[best_target, message]:.1f}%)"
    )