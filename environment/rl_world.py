import numpy as np


class RLWorld:

    def __init__(self):

        self.width = 10.0
        self.height = 10.0

        self.agent = None
        self.target = None

        self.max_steps = 50
        self.step_count = 0

    def reset(self):

        self.agent = np.random.uniform(
            0.5,
            9.5,
            size=2
        ).astype(np.float32)

        self.target = np.random.uniform(
            0.5,
            9.5,
            size=2
        ).astype(np.float32)

        self.step_count = 0

        return self.get_state()

    def get_state(self):

        return np.array([
            self.agent[0] / 10.0,
            self.agent[1] / 10.0,
            self.target[0] / 10.0,
            self.target[1] / 10.0,
        ], dtype=np.float32)

    def distance(self):

        return np.linalg.norm(
            self.agent - self.target
        )

    def step(self, action):

        old_distance = self.distance()

        step_size = 0.25

        actions = {
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.0, -1.0]),
            3: np.array([-1.0, 0.0]),
            4: np.array([1.0, 0.0]),
        }

        self.agent += (
            actions[action] * step_size
        )

        self.agent = np.clip(
            self.agent,
            0.0,
            10.0
        )

        new_distance = self.distance()

        reward = (
            old_distance - new_distance
        )

        self.step_count += 1

        reached = new_distance < 0.35

        if reached:

            reward += 10.0

        done = (
            reached
            or self.step_count >= self.max_steps
        )

        return (
            self.get_state(),
            reward,
            done,
            reached
        )