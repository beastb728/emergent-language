import torch

from environment.region_world_v2 import RegionWorldV2
from rl.dqn import DQNAgent
from rl.sender import Sender


NUM_EPISODES = 5000


world = RegionWorldV2()

sender = Sender()

receiver = DQNAgent(
    state_size=6,
    action_size=5
)


success_count = 0


for episode in range(NUM_EPISODES):

    world.reset()

    # -------------------------
    # Sender chooses message
    # -------------------------

    sender_state = world.sender_state()

    message, log_probability = sender.choose_message(
        sender_state
    )

    # -------------------------
    # Receiver navigates
    # -------------------------

    state = world.receiver_state(message)

    episode_reward = 0.0
    done = False

    while not done:

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
        episode_reward += reward

    # -------------------------
    # Sender policy update
    # -------------------------

    loss = -log_probability * episode_reward

    sender.optimizer.zero_grad()
    loss.backward()
    sender.optimizer.step()

    # -------------------------
    # Receiver updates
    # -------------------------

    receiver.decay_epsilon()

    if episode % 100 == 0:
        receiver.update_target_network()

    if reached:
        success_count += 1

    # -------------------------
    # Progress
    # -------------------------

    if episode % 100 == 0:

        success_rate = success_count / 100

        print(
            f"Episode {episode} | "
            f"Success: {success_rate * 100:.1f}% | "
            f"Epsilon: {receiver.epsilon:.3f}"
        )

        success_count = 0


# -------------------------
# Save models
# -------------------------

torch.save(
    sender.network.state_dict(),
    "models/sender_region_v2.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_region_v2.pt"
)

print("\nTraining complete.")

print(
    "Saved models/sender_region_v2.pt"
)

print(
    "Saved models/receiver_region_v2.pt"
)