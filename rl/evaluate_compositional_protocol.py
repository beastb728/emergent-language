import numpy as np
import torch
from collections import defaultdict

from environment.compositional_world import CompositionalWorld
from rl.sender import SenderNetwork


NUM_EPISODES = 2000

sender = SenderNetwork(state_size=6, message_size=16)
sender.load_state_dict(
    torch.load(
        "models/sender_compositional_finetuned.pt",
        map_location="cpu"
    )
)
sender.eval()

world = CompositionalWorld()

counts = defaultdict(lambda: np.zeros(16, dtype=int))

for _ in range(NUM_EPISODES):
    world.reset()

    state = torch.tensor(
        world.sender_state(),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits = sender(state)
        message = torch.argmax(logits, dim=-1).item()

    x_region = world.x_region
    y_region = world.y_region

    counts[(x_region, y_region)][message] += 1


print()
print("Learned sender protocol")
print("=======================")

for x in range(4):
    row = []

    for y in range(4):
        c = counts[(x, y)]
        message = int(np.argmax(c))
        consistency = 100.0 * c[message] / c.sum()

        row.append(f"{message:02d} ({consistency:.0f}%)")

    print(f"X{x}: " + " | ".join(row))


print()
print("Symbol decomposition")
print("====================")

first_symbol = np.zeros((4, 4), dtype=int)
second_symbol = np.zeros((4, 4), dtype=int)

for x in range(4):
    for y in range(4):
        c = counts[(x, y)]
        message = int(np.argmax(c))

        first_symbol[x, y] = message // 4
        second_symbol[x, y] = message % 4

print("First symbol:")
print(first_symbol)

print()
print("Second symbol:")
print(second_symbol)


print()
print("X-region -> first-symbol consistency")
print("====================================")

for x in range(4):
    symbols = []

    for y in range(4):
        symbols.append(first_symbol[x, y])

    counts_x = np.bincount(symbols, minlength=4)
    consistency = 100.0 * counts_x.max() / len(symbols)

    print(
        f"X{x}: {symbols} | "
        f"consistency = {consistency:.1f}%"
    )


print()
print("Y-region -> second-symbol consistency")
print("=====================================")

for y in range(4):
    symbols = []

    for x in range(4):
        symbols.append(second_symbol[x, y])

    counts_y = np.bincount(symbols, minlength=4)
    consistency = 100.0 * counts_y.max() / len(symbols)

    print(
        f"Y{y}: {symbols} | "
        f"consistency = {consistency:.1f}%"
    )