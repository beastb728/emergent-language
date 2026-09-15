import torch
from torch.distributions import Categorical

from environment.compositional_world_v2 import CompositionalWorldV2
from rl.dqn import DQNAgent
from rl.sender import SenderNetwork


NUM_EPISODES = 5000

world = CompositionalWorldV2()

sender_x = SenderNetwork(
    state_size=6,
    message_size=4
)

sender_y = SenderNetwork(
    state_size=6,
    message_size=4
)

optimizer_x = torch.optim.Adam(
    sender_x.parameters(),
    lr=1e-3
)

optimizer_y = torch.optim.Adam(
    sender_y.parameters(),
    lr=1e-3
)

receiver = DQNAgent(
    state_size=10,
    action_size=5
)

success_count = 0


for episode in range(NUM_EPISODES):

    world.reset()

    sender_state = torch.tensor(
        world.sender_state(),
        dtype=torch.float32
    ).unsqueeze(0)

    # X symbol
    logits_x = sender_x(sender_state)
    distribution_x = Categorical(logits=logits_x)
    symbol_x = distribution_x.sample()
    log_prob_x = distribution_x.log_prob(symbol_x)

    # Y symbol
    logits_y = sender_y(sender_state)
    distribution_y = Categorical(logits=logits_y)
    symbol_y = distribution_y.sample()
    log_prob_y = distribution_y.log_prob(symbol_y)

    symbol_x = symbol_x.item()
    symbol_y = symbol_y.item()

    state = world.receiver_state(
        symbol_x,
        symbol_y
    )

    episode_reward = 0.0
    done = False

    while not done:

        action = receiver.choose_action(state)

        reward, done, reached = world.step(action)

        next_state = world.receiver_state(
            symbol_x,
            symbol_y
        )

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

    # Shared reward for both communication channels
    loss_x = -log_prob_x * episode_reward
    loss_y = -log_prob_y * episode_reward

    optimizer_x.zero_grad()
    loss_x.backward()
    optimizer_x.step()

    optimizer_y.zero_grad()
    loss_y.backward()
    optimizer_y.step()

    receiver.decay_epsilon()

    if episode % 100 == 0:
        receiver.update_target_network()

    if reached:
        success_count += 1

    if episode % 100 == 0:

        print(
            f"Episode {episode} | "
            f"Success: {success_count:.1f}% | "
            f"Epsilon: {receiver.epsilon:.3f}"
        )

        success_count = 0


torch.save(
    sender_x.state_dict(),
    "models/sender_compositional_v2_x.pt"
)

torch.save(
    sender_y.state_dict(),
    "models/sender_compositional_v2_y.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_compositional_v2.pt"
)

print()
print("Training complete.")
print("Saved:")
print("models/sender_compositional_v2_x.pt")
print("models/sender_compositional_v2_y.pt")
print("models/receiver_compositional_v2.pt")