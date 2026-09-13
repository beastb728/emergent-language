import numpy as np


class World:
    def __init__(self, width=10.0, height=10.0):
        self.width = width
        self.height = height

        self.agent_a = None
        self.agent_b = None
        self.target = None

        self.reset()

    def reset(self):
        self.agent_a = self._random_position()
        self.agent_b = self._random_position()
        self.target = self._random_position()

    def _random_position(self):
        return np.random.uniform(
            low=0.5,
            high=9.5,
            size=2
        ).astype(np.float32)