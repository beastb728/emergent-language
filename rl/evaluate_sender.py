import numpy as np
import torch

from environment.communication_world import CommunicationWorld
from rl.sender import Sender
from rl.dqn import DQNAgent
from rl.random_protocol import RandomFixedProtocol


EPISODES = 2000


world = CommunicationWorld()

# Recreate the same Stage A protocol.
protocol = RandomFixedProtocol()

sender = Sender()

sender.network.load_state_dict(
    torch.load(
        "models/sender_stage_b.pt",
        map_location="cpu"
    )
)

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

# Count messages selected for each target.
message_counts = {
    target_id: np.zeros(4, dtype=int)
    for target_id in range(4)
}

success_count = 0


print()
print("Stage B evaluation")
print()

print("Hidden protocol used during Stage A:")
protocol.show()

print()


for episode in range(EPISODES):

    world.reset()

    target_id = world.target_id

    state = world.sender_state()

    with torch.no_grad():
        message, _ = sender.choose_message(state)

    message_counts[target_id][message] += 1

    receiver_state = world.receiver_state(message)

    success = False

    for step in range(world.max_steps):

        action = receiver.choose_action(receiver_state)

        reward, done, reached = world.step(action)

        receiver_state = world.receiver_state(message)

        if reached:
            success = True
            break

        if done:
            break

    if success:
        success_count += 1


print(f"Overall success: {success_count / EPISODES * 100:.2f}%")
print()

print("Learned sender communication:")
print()

for target_id in range(4):

    counts = message_counts[target_id]

    most_common_message = int(np.argmax(counts))

    print(
        f"T{target_id}: "
        f"message {format(most_common_message, '02b')} "
        f"| counts {counts.tolist()}"
    )

print()
print("Expected receiver convention:")
print()

for target_id in range(4):
    message = protocol.encode(target_id)

    print(
        f"T{target_id} -> "
        f"{format(message, '02b')}"
    )