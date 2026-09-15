import torch
from torch.distributions import Categorical

from environment.compositional_world import CompositionalWorld
from rl.dqn import DQNAgent
from rl.sender import SenderNetwork


NUM_EPISODES = 5000


world = CompositionalWorld()

sender = SenderNetwork(
    state_size=6,
    message_size=16
)

sender_optimizer = torch.optim.Adam(
    sender.parameters(),
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

    # One categorical distribution over 16 messages.
    logits = sender(sender_state)

    distribution = Categorical(
        logits=logits
    )

    message_id = distribution.sample()

    log_probability = distribution.log_prob(
        message_id
    )

    message_id = message_id.item()

    # Decode 16-way message into two 4-valued symbols.
    message_1 = message_id // 4
    message_2 = message_id % 4

    message_one_hot = torch.zeros(
        8,
        dtype=torch.float32
    )

    message_one_hot[message_1] = 1.0
    message_one_hot[4 + message_2] = 1.0

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

    while not done:

        action = receiver.choose_action(state)

        reward, done, reached = world.step(action)

        next_position = torch.tensor(
            world.agent_b / 10.0,
            dtype=torch.float32
        )

        next_state = torch.cat(
            [
                next_position,
                message_one_hot
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

    # Sender REINFORCE update.
    loss = -log_probability * episode_reward

    sender_optimizer.zero_grad()
    loss.backward()
    sender_optimizer.step()

    receiver.decay_epsilon()

    if episode % 100 == 0:
        receiver.update_target_network()

    if reached:
        success_count += 1

    if episode % 100 == 0:

        success_rate = success_count / 100

        print(
            f"Episode {episode} | "
            f"Success: {success_rate * 100:.1f}% | "
            f"Epsilon: {receiver.epsilon:.3f}"
        )

        success_count = 0


torch.save(
    sender.state_dict(),
    "models/sender_compositional_v2.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_compositional_v2.pt"
)

print()
print("Training complete.")
print("Saved:")
print("models/sender_compositional_v2.pt")
print("models/receiver_compositional_v2.pt")
