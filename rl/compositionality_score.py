import numpy as np
import torch

from environment.compositional_world import CompositionalWorld
from rl.sender import SenderNetwork


NUM_EPISODES = 5000


def mutual_information(x, y):
    x = np.asarray(x)
    y = np.asarray(y)

    x_values = np.unique(x)
    y_values = np.unique(y)

    joint = np.zeros((len(x_values), len(y_values)), dtype=np.float64)

    for a, b in zip(x, y):
        i = np.where(x_values == a)[0][0]
        j = np.where(y_values == b)[0][0]
        joint[i, j] += 1

    joint /= joint.sum()

    px = joint.sum(axis=1, keepdims=True)
    py = joint.sum(axis=0, keepdims=True)

    mi = 0.0

    for i in range(joint.shape[0]):
        for j in range(joint.shape[1]):
            if joint[i, j] > 0:
                mi += joint[i, j] * np.log2(
                    joint[i, j] / (px[i, 0] * py[0, j])
                )

    return mi


sender = SenderNetwork(state_size=6, message_size=16)
sender.load_state_dict(
    torch.load(
        "models/sender_compositional_finetuned.pt",
        map_location="cpu"
    )
)
sender.eval()

world = CompositionalWorld()

x_regions = []
y_regions = []
first_symbols = []
second_symbols = []

for _ in range(NUM_EPISODES):
    world.reset()

    state = torch.tensor(
        world.sender_state(),
        dtype=torch.float32
    ).unsqueeze(0)

    with torch.no_grad():
        logits = sender(state)
        message = torch.argmax(logits, dim=-1).item()

    x_regions.append(world.x_region)
    y_regions.append(world.y_region)

    first_symbols.append(message // 4)
    second_symbols.append(message % 4)


x_regions = np.array(x_regions)
y_regions = np.array(y_regions)
first_symbols = np.array(first_symbols)
second_symbols = np.array(second_symbols)


# Desired factorization:
# X -> first symbol
# Y -> second symbol

mi_x_first = mutual_information(x_regions, first_symbols)
mi_y_second = mutual_information(y_regions, second_symbols)

# Cross-information should ideally be smaller:
# X -> second symbol
# Y -> first symbol

mi_x_second = mutual_information(x_regions, second_symbols)
mi_y_first = mutual_information(y_regions, first_symbols)


# Normalize by entropy of the factor.
def entropy(values):
    _, counts = np.unique(values, return_counts=True)
    probabilities = counts / counts.sum()

    return -np.sum(
        probabilities * np.log2(probabilities)
    )


hx = entropy(x_regions)
hy = entropy(y_regions)

normalized_x_first = mi_x_first / hx
normalized_y_second = mi_y_second / hy

normalized_x_second = mi_x_second / hx
normalized_y_first = mi_y_first / hy


# Simple factorization score.
#
# High desired information + low cross information = better.
factorization_score = (
    normalized_x_first
    + normalized_y_second
    - normalized_x_second
    - normalized_y_first
) / 2.0


print()
print("Compositionality analysis")
print("=========================")

print(f"X entropy:                 {hx:.3f}")
print(f"Y entropy:                 {hy:.3f}")

print()
print("Desired information")
print("-------------------")
print(f"MI(X, first symbol):       {mi_x_first:.3f}")
print(f"MI(Y, second symbol):      {mi_y_second:.3f}")

print()
print("Cross information")
print("-----------------")
print(f"MI(X, second symbol):      {mi_x_second:.3f}")
print(f"MI(Y, first symbol):       {mi_y_first:.3f}")

print()
print("Normalized information")
print("----------------------")
print(f"X -> first symbol:         {normalized_x_first:.3f}")
print(f"Y -> second symbol:        {normalized_y_second:.3f}")
print(f"X -> second symbol:        {normalized_x_second:.3f}")
print(f"Y -> first symbol:         {normalized_y_first:.3f}")

print()
print(f"Factorization score:       {factorization_score:.3f}")

print()
if factorization_score >= 0.75:
    print("Strong compositional structure.")
elif factorization_score >= 0.50:
    print("Moderate compositional structure.")
elif factorization_score >= 0.25:
    print("Weak compositional structure.")
else:
    print("No convincing compositional structure.")