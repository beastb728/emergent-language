import numpy as np
import torch

from environment.communication_world import CommunicationWorld
from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES_PER_CELL = 250


world = CommunicationWorld()

sender = Sender()

sender.network.load_state_dict(
    torch.load(
        "models/sender_joint_final.pt",
        map_location="cpu"
    )
)

receiver = DQNAgent(
    state_size=6,
    action_size=5
)

receiver.policy_net.load_state_dict(
    torch.load(
        "models/receiver_joint_final.pt",
        map_location="cpu"
    )
)

receiver.target_net.load_state_dict(
    receiver.policy_net.state_dict()
)

# Deterministic evaluation.
receiver.epsilon = 0.0


messages = [0, 1, 2, 3]

results = np.zeros((4, 4))


print()
print("Joint protocol evaluation")
print()
print(
    "Rows = actual target | "
    "Columns = forced message"
)
print()


# --------------------------------
# Evaluate receiver semantics
# --------------------------------

for target_id in range(4):

    for message in messages:

        successes = 0

        for episode in range(EPISODES_PER_CELL):

            world.reset()

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

        results[target_id, message] = (
            successes / EPISODES_PER_CELL * 100
        )


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
print("Receiver's learned message meanings:")
print()

for message in messages:

    best_target = int(
        np.argmax(results[:, message])
    )

    print(
        f"{format(message, '02b')} -> T{best_target} "
        f"({results[best_target, message]:.1f}%)"
    )


# --------------------------------
# Evaluate sender consistency
# --------------------------------

sender_counts = {
    target_id: np.zeros(4, dtype=int)
    for target_id in range(4)
}


print()
print("Sender message selection:")
print()


for episode in range(1000):

    world.reset()

    target_id = world.target_id

    state = world.sender_state()

    with torch.no_grad():

        message, _ = sender.choose_message(state)

    sender_counts[target_id][message] += 1


for target_id in range(4):

    counts = sender_counts[target_id]

    best_message = int(
        np.argmax(counts)
    )

    total = int(np.sum(counts))

    consistency = (
        counts[best_message] / total * 100
    )

    print(
        f"T{target_id}: "
        f"message {format(best_message, '02b')} "
        f"| counts {counts.tolist()} "
        f"| consistency {consistency:.1f}%"
    )