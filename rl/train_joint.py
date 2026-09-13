import numpy as np
import torch

from environment.communication_world import CommunicationWorld
from rl.dqn import DQNAgent
from rl.sender import Sender


EPISODES = 5000


world = CommunicationWorld()

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

receiver.target_net.load_state_dict(
    receiver.policy_net.state_dict()
)

# Keep some exploration, but do not restart from epsilon=1.
receiver.epsilon = 0.10

success_history = []


print()
print("Stage C: joint sender-receiver learning")
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

    # -------------------------
    # Update receiver
    # -------------------------

    receiver.decay_epsilon()

    if episode % 100 == 0:

        receiver.update_target_network()

        recent_success = (
            np.mean(success_history[-99:] + [int(success)])
            * 100
            if success_history
            else int(success) * 100
        )

        success_history.append(int(success))

        print(
            f"Episode {episode:5d} | "
            f"Success: {recent_success:5.1f}% | "
            f"Reward: {total_reward:7.2f} | "
            f"Epsilon: {receiver.epsilon:.3f}"
        )

    else:
        success_history.append(int(success))


torch.save(
    sender.network.state_dict(),
    "models/sender_joint_final.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_joint_final.pt"
)


print()
print("Stage C complete.")
print("Saved:")
print("  models/sender_joint_final.pt")
print("  models/receiver_joint_final.pt")