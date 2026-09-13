import os

import numpy as np

from environment.rl_world import RLWorld
from rl.dqn import DQNAgent


EPISODES = 1000
TARGET_UPDATE = 25

world = RLWorld()

agent = DQNAgent()

os.makedirs("models", exist_ok=True)

success_history = []

for episode in range(1, EPISODES + 1):

    state = world.reset()

    total_reward = 0.0
    success = False

    for step in range(world.max_steps):

        action = agent.choose_action(state)

        next_state, reward, done, reached = (
            world.step(action)
        )

        agent.remember(
            state,
            action,
            reward,
            next_state,
            done
        )

        agent.train_step()

        state = next_state

        total_reward += reward

        if reached:
            success = True
            break

        if done:
            break

    agent.decay_epsilon()

    success_history.append(
        int(success)
    )

    if episode % TARGET_UPDATE == 0:

        agent.update_target_network()

    if episode % 50 == 0:

        recent_success = np.mean(
            success_history[-50:]
        ) * 100

        print(
            f"Episode {episode:4d} | "
            f"Success: {recent_success:5.1f}% | "
            f"Reward: {total_reward:7.2f} | "
            f"Epsilon: {agent.epsilon:.3f}"
        )

    if episode % 250 == 0:

        torch_path = (
            f"models/dqn_navigation_{episode}.pt"
        )

        import torch

        torch.save(
            agent.policy_net.state_dict(),
            torch_path
        )

print("Training complete.")

import torch

torch.save(
    agent.policy_net.state_dict(),
    "models/dqn_navigation_final.pt"
)