import torch
from torch.distributions import Categorical

from environment.compositional_world import CompositionalWorld
from rl.dqn import DQNAgent
from rl.sender import SenderNetwork


NUM_EPISODES = 2000


world = CompositionalWorld()

sender = SenderNetwork(
    state_size=6,
    message_size=16
)

sender.load_state_dict(
    torch.load(
        "models/sender_compositional_supervised.pt",
        map_location="cpu"
    )
)

sender_optimizer = torch.optim.Adam(
    sender.parameters(),
    lr=1e-4
)


receiver = DQNAgent(
    state_size=18,
    action_size=5
)

receiver.policy_net.load_state_dict(
    torch.load(
        "models/receiver_compositional_staged.pt",
        map_location="cpu"
    )
)

receiver.target_net.load_state_dict(
    receiver.policy_net.state_dict()
)

receiver.epsilon = 0.10


success_count = 0


for episode in range(NUM_EPISODES):

    world.reset()

    sender_state = torch.tensor(
        world.sender_state(),
        dtype=torch.float32
    ).unsqueeze(0)

    logits = sender(sender_state)

    distribution = Categorical(
        logits=logits
    )

    message_id = distribution.sample()

    log_probability = distribution.log_prob(
        message_id
    )

    message_id = message_id.item()

    message_1 = message_id // 4
    message_2 = message_id % 4

    message_one_hot = torch.zeros(
        16,
        dtype=torch.float32
    )

    message_one_hot[message_id] = 1.0

    state = torch.cat(
        [
            torch.tensor(
                world.agent_b / 10.0,
                dtype=torch.float32
            ),
            message_one_hot
        ]
    ).numpy()

    episode_reward = 0.0
    done = False

    while not done:

        action = receiver.choose_action(state)

        reward, done, reached = world.step(action)

        next_state = torch.cat(
            [
                torch.tensor(
                    world.agent_b / 10.0,
                    dtype=torch.float32
                ),
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

    sender_loss = (
        -log_probability * episode_reward
    )

    sender_optimizer.zero_grad()
    sender_loss.backward()
    sender_optimizer.step()

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
    sender.state_dict(),
    "models/sender_compositional_finetuned.pt"
)

torch.save(
    receiver.policy_net.state_dict(),
    "models/receiver_compositional_finetuned.pt"
)

print()
print("Fine-tuning complete.")
print("Saved:")
print("models/sender_compositional_finetuned.pt")
print("models/receiver_compositional_finetuned.pt")