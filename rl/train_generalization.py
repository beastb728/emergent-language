import numpy as np
import torch

from environment.generalization_world import GeneralizationWorld
from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES = 5000


world = GeneralizationWorld()

sender = Sender()

receiver = DQNAgent(
    state_size=6,
    action_size=5
)

# Start from the successful V1.6.2 models,
# but allow both agents to adapt to the
# randomized target environment.
sender.network.load_state_dict(
    torch.load(
        "models/sender_joint_final.pt",
        map_location="cpu"
    )
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

receiver.epsilon = 0.10

success_history = []


print()
print("V1.7.3: training on randomized target positions")
print()


for episode in range(1, EPISODES + 1):

    world.reset()

    # -------------------------
    # Sender
    # -------------------------

    sender_state = world.sender_state()

    message, log_probability = (
        sender.choose_message(sender_state)
    )

    # -------------------------
    # Receiver
    # -------------------------

    state = world.receiver_state(message)

    total_reward = 0.0
    success = False

    for step in range(world.max_steps):

        action = receiver.choose_action(state)

        reward, done, reached = world.step(action)

        next_state = world.receiver_state(message)

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
    # Sender update
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

    success_history.append(int(success))

    # -------------------------
    # Receiver update
    # -------------------------

    receiver.decay_epsilon()

    if episode % 100 == 0:

        receiver.update_target_network()

        recent_success = (
            np.mean(success_history[-100:])
            * 100
        )

        print(
            f"Episode {episode:5d} | "
            f"Success: {recent_success:5.1f}% | "
            f"Reward: {total_reward:7.2f} | "
            f"Epsilon: {receiver.epsilon:.3f}"
        )


torch.save(
    sender.network.state_dict(),
    "models/sender_generalization.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_generalization.pt"
)


print()
print("V1.7.3 complete.")
print("Saved:")
print("  models/sender_generalization.pt")
print("  models/receiver_generalization.pt")