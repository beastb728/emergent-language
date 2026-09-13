import numpy as np
import torch

from environment.rl_world import RLWorld
from rl.dqn import DQNAgent


MODEL_PATH = "models/dqn_navigation_final.pt"
EPISODES = 500


world = RLWorld()
agent = DQNAgent()

# Load trained model
agent.policy_net.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location="cpu"
    )
)

# Turn OFF exploration
agent.epsilon = 0.0

successes = 0
steps_taken = []


for episode in range(1, EPISODES + 1):

    state = world.reset()

    for step in range(1, world.max_steps + 1):

        action = agent.choose_action(state)

        next_state, reward, done, reached = (
            world.step(action)
        )

        state = next_state

        if reached:

            successes += 1
            steps_taken.append(step)

            break

        if done:

            break


success_rate = (
    successes / EPISODES
) * 100


if steps_taken:
    average_steps = np.mean(steps_taken)
else:
    average_steps = float("nan")


print()
print("=" * 45)
print("DQN NAVIGATION EVALUATION")
print("=" * 45)
print(f"Episodes:       {EPISODES}")
print(f"Successes:      {successes}")
print(f"Failures:       {EPISODES - successes}")
print(f"Success rate:   {success_rate:.2f}%")
print(f"Average steps:  {average_steps:.2f}")
print("=" * 45)