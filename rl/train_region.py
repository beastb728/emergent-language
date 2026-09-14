import numpy as np
import torch

from environment.region_world import RegionWorld
from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES = 5000


world = RegionWorld()

sender = Sender()

receiver = DQNAgent(
    state_size=6,
    action_size=5
)

success_history = []


print()
print("V1.7.5: training communication on spatial regions")
print()


for episode in range(1, EPISODES + 1):

    world.reset()

    # -------------------------
    # Sender chooses message
    # -------------------------

    sender_state = world.sender_state()

    message, log_probability = (
        sender.choose_message(sender_state)
    )

    # -------------------------
    # Receiver navigates
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
    # Train sender
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
    # Update receiver
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
    "models/sender_region.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_region.pt"
)


print()
print("V1.7.5 complete.")
print("Saved:")
print("  models/sender_region.pt")
print("  models/receiver_region.pt")