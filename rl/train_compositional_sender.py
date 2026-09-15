import numpy as np
import torch
from torch.distributions import Categorical

from environment.compositional_world import CompositionalWorld
from rl.sender import SenderNetwork


NUM_EPISODES = 5000


world = CompositionalWorld()

protocol = np.load(
    "models/compositional_protocol.npy",
    allow_pickle=True
).item()


sender = SenderNetwork(
    state_size=6,
    message_size=16
)

optimizer = torch.optim.Adam(
    sender.parameters(),
    lr=1e-3
)


success_count = 0


for episode in range(NUM_EPISODES):

    world.reset()

    target_key = (
        world.x_region,
        world.y_region
    )

    correct_message = protocol[target_key]

    state = torch.tensor(
        world.sender_state(),
        dtype=torch.float32
    ).unsqueeze(0)

    logits = sender(state)

    distribution = Categorical(
        logits=logits
    )

    message = distribution.sample()

    log_probability = distribution.log_prob(
        message
    )

    # --------------------------------
    # Reward sender for correct symbol
    # --------------------------------

    reward = 1.0 if message.item() == correct_message else 0.0

    loss = -log_probability * reward

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if reward == 1.0:
        success_count += 1

    if episode % 100 == 0:

        accuracy = success_count

        print(
            f"Episode {episode} | "
            f"Sender accuracy: {accuracy:.1f}%"
        )

        success_count = 0


torch.save(
    sender.state_dict(),
    "models/sender_compositional_staged.pt"
)

print()
print("Stage B complete.")
print(
    "Saved models/sender_compositional_staged.pt"
)