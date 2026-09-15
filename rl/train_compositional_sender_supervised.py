import numpy as np
import torch
import torch.nn.functional as F

from environment.compositional_world import CompositionalWorld
from rl.sender import SenderNetwork


NUM_EPISODES = 3000


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


correct = 0


for episode in range(NUM_EPISODES):

    world.reset()

    target_key = (
        world.x_region,
        world.y_region
    )

    target_message = protocol[target_key]

    state = torch.tensor(
        world.sender_state(),
        dtype=torch.float32
    ).unsqueeze(0)

    logits = sender(state)

    target = torch.tensor(
        [target_message],
        dtype=torch.long
    )

    loss = F.cross_entropy(
        logits,
        target
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    prediction = logits.argmax(dim=1).item()

    if prediction == target_message:
        correct += 1

    if episode % 100 == 0:

        accuracy = correct

        print(
            f"Episode {episode} | "
            f"Sender accuracy: {accuracy:.1f}%"
        )

        correct = 0


torch.save(
    sender.state_dict(),
    "models/sender_compositional_supervised.pt"
)

print()
print("Supervised sender training complete.")