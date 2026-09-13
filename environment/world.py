import numpy as np


class World:
    def __init__(self, width=800, height=600):
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
        margin = 50

        x = np.random.uniform(margin, self.width - margin)
        y = np.random.uniform(margin, self.height - margin)

        return np.array([x, y], dtype=np.float32)