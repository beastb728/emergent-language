import numpy as np
import torch

from environment.region_world_v2 import RegionWorldV2
from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES_PER_CELL = 250
SENDER_EPISODES = 1000


world = RegionWorldV2()

sender = Sender()

sender.network.load_state_dict(
    torch.load(
        "models/sender_region_v2.pt",
        map_location="cpu"
    )
)

receiver = DQNAgent(
    state_size=6,
    action_size=5
)

receiver.policy_net.load_state_dict(
    torch.load(
        "models/receiver_region_v2.pt",
        map_location="cpu"
    )
)

receiver.target_net.load_state_dict(
    receiver.policy_net.state_dict()
)

receiver.epsilon = 0.0


# --------------------------------
# Receiver protocol matrix
# --------------------------------

messages = [0, 1, 2, 3]

results = np.zeros((4, 4))


print()
print("V1.7.7: corrected region protocol evaluation")
print()
print(
    "Rows = actual target region | "
    "Columns = forced message"
)
print()


for target_id in range(4):

    for message in messages:

        successes = 0

        for episode in range(EPISODES_PER_CELL):

            world.reset()

            # Force target region.
            world.target_id = target_id

            # Sample a fresh target inside that region.
            world.target = world._sample_target(target_id)

            # Make sure B starts outside the target region.
            while world.in_target_region():
                world.agent_b = np.random.uniform(
                    0.5,
                    9.5,
                    size=2
                ).astype(np.float32)

            state = world.receiver_state(message)

            done = False

            while not done:

                action = receiver.choose_action(state)

                reward, done, reached = world.step(action)

                state = world.receiver_state(message)

                if reached:
                    successes += 1
                    break

        results[target_id, message] = (
            successes / EPISODES_PER_CELL * 100
        )


print("             Message")
print("Region       00      01      10      11")
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


# --------------------------------
# Learned message meanings
# --------------------------------

print()
print("Receiver's learned message meanings:")
print()

for message in messages:

    best_target = int(
        np.argmax(results[:, message])
    )

    print(
        f"{format(message, '02b')} -> "
        f"T{best_target} "
        f"({results[best_target, message]:.1f}%)"
    )


# --------------------------------
# Sender consistency
# --------------------------------

sender_counts = {
    target_id: np.zeros(4, dtype=int)
    for target_id in range(4)
}


for episode in range(SENDER_EPISODES):

    world.reset()

    target_id = world.target_id

    state = world.sender_state()

    message, _ = sender.choose_message(state)

    sender_counts[target_id][message] += 1


print()
print("Sender message selection:")
print()

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


# --------------------------------
# End-to-end sender -> receiver
# --------------------------------

successes = 0

for episode in range(SENDER_EPISODES):

    world.reset()

    state = world.sender_state()

    message, _ = sender.choose_message(state)

    receiver_state = world.receiver_state(message)

    done = False

    while not done:

        action = receiver.choose_action(receiver_state)

        reward, done, reached = world.step(action)

        receiver_state = world.receiver_state(message)

        if reached:
            successes += 1
            break


overall_success = (
    successes / SENDER_EPISODES * 100
)

print()
print(
    f"End-to-end success: "
    f"{overall_success:.1f}% "
    f"({successes}/{SENDER_EPISODES})"
)