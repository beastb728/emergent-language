import torch
import torch.nn as nn
from torch.distributions import Categorical

from environment.compositional_world import CompositionalWorld
from rl.dqn import DQNAgent
from rl.sender import SenderNetwork


NUM_EPISODES = 5000


world = CompositionalWorld()

sender_x = SenderNetwork(
    state_size=6,
    message_size=4
)

sender_y = SenderNetwork(
    state_size=6,
    message_size=4
)

optimizer = torch.optim.Adam(
    list(sender_x.parameters()) +
    list(sender_y.parameters()),
    lr=1e-3
)

receiver = DQNAgent(
    state_size=10,
    action_size=5
)


success_count = 0


for episode in range(NUM_EPISODES):

    world.reset()

    # --------------------------------
    # Sender
    # --------------------------------

    sender_state = torch.tensor(
        world.sender_state(),
        dtype=torch.float32
    ).unsqueeze(0)

    logits_x = sender_x(sender_state)
    logits_y = sender_y(sender_state)

    distribution_x = Categorical(
        logits=logits_x
    )

    distribution_y = Categorical(
        logits=logits_y
    )

    message_x = distribution_x.sample()
    message_y = distribution_y.sample()

    log_prob_x = distribution_x.log_prob(
        message_x
    )

    log_prob_y = distribution_y.log_prob(
        message_y
    )

    message_x = message_x.item()
    message_y = message_y.item()

    # --------------------------------
    # Receiver state
    # --------------------------------

    message_one_hot = torch.zeros(
        8,
        dtype=torch.float32
    )

    message_one_hot[message_x] = 1.0
    message_one_hot[4 + message_y] = 1.0

    state = torch.tensor(
        world.agent_b / 10.0,
        dtype=torch.float32
    )

    state = torch.cat(
        [
            state,
            message_one_hot
        ]
    ).numpy()

    episode_reward = 0.0
    done = False

    # --------------------------------
    # Navigation
    # --------------------------------

    while not done:

        action = receiver.choose_action(state)

        reward, done, reached = world.step(
            action
        )

        next_message_one_hot = message_one_hot

        next_state = torch.tensor(
            world.agent_b / 10.0,
            dtype=torch.float32
        )

        next_state = torch.cat(
            [
                next_state,
                next_message_one_hot
            ]
        ).numpy()

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

    # --------------------------------
    # Sender REINFORCE update
    # --------------------------------

    sender_loss = -(
        log_prob_x + log_prob_y
    ) * episode_reward

    optimizer.zero_grad()
    sender_loss.backward()
    optimizer.step()

    # --------------------------------
    # Receiver updates
    # --------------------------------

    receiver.decay_epsilon()

    if episode % 100 == 0:
        receiver.update_target_network()

    if reached:
        success_count += 1

    # --------------------------------
    # Progress
    # --------------------------------

    if episode % 100 == 0:

        success_rate = (
            success_count / 100
        )

        print(
            f"Episode {episode} | "
            f"Success: {success_rate * 100:.1f}% | "
            f"Epsilon: {receiver.epsilon:.3f}"
        )

        success_count = 0


# --------------------------------
# Save models
# --------------------------------

torch.save(
    sender_x.state_dict(),
    "models/sender_x_compositional.pt"
)

torch.save(
    sender_y.state_dict(),
    "models/sender_y_compositional.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_compositional.pt"
)


print()
print("Training complete.")
print("Saved:")
print("models/sender_x_compositional.pt")
print("models/sender_y_compositional.pt")
print("models/receiver_compositional.pt")