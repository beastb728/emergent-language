import numpy as np
import torch

from environment.generalization_world import GeneralizationWorld
from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES = 2000


world = GeneralizationWorld()

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

receiver.epsilon = 0.0


success_count = 0

message_counts = {
    target_id: np.zeros(4, dtype=int)
    for target_id in range(4)
}


print()
print("V1.7 zero-shot generalization evaluation")
print()
print(
    "Testing V1.6.2 models on randomly positioned targets."
)
print()


for episode in range(EPISODES):

    world.reset()

    target_id = world.target_id

    sender_state = world.sender_state()

    with torch.no_grad():

        message, _ = sender.choose_message(
            sender_state
        )

    message_counts[target_id][message] += 1

    receiver_state = world.receiver_state(message)

    success = False

    for step in range(world.max_steps):

        action = receiver.choose_action(
            receiver_state
        )

        reward, done, reached = world.step(
            action
        )

        receiver_state = world.receiver_state(
            message
        )

        if reached:
            success = True
            break

        if done:
            break

    if success:
        success_count += 1


print(
    f"Overall success: "
    f"{success_count / EPISODES * 100:.2f}%"
)

print()
print("Sender message consistency:")
print()

for target_id in range(4):

    counts = message_counts[target_id]

    total = int(np.sum(counts))

    best_message = int(np.argmax(counts))

    consistency = (
        counts[best_message] / total * 100
    )

    print(
        f"T{target_id}: "
        f"message {format(best_message, '02b')} "
        f"| counts {counts.tolist()} "
        f"| consistency {consistency:.1f}%"
    )