import numpy as np

from environment.compositional_world import CompositionalWorld
from rl.dqn import DQNAgent


EPISODES = 2000


world = CompositionalWorld()

receiver = DQNAgent(
    state_size=2,
    action_size=5
)


successes = 0


for episode in range(EPISODES):

    world.reset()

    # Receiver gets ONLY its own position.
    state = (
        world.agent_b / 10.0
    ).astype(np.float32)

    done = False

    while not done:

        action = receiver.choose_action(state)

        reward, done, reached = world.step(action)

        state = (
            world.agent_b / 10.0
        ).astype(np.float32)

        if reached:
            successes += 1
            break


print()
print("V1.8.2: no-communication baseline")
print()
print(
    f"Success: "
    f"{successes / EPISODES * 100:.2f}% "
    f"({successes}/{EPISODES})"
)