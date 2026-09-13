import numpy as np


class World:
    def __init__(self, width=10.0, height=10.0):
        self.width = width
        self.height = height

        self.agent_a = None
        self.agent_b = None

        self.target_locations = np.array([
            [2.0, 2.0],  # T0
            [8.0, 2.0],  # T1
            [2.0, 8.0],  # T2
            [8.0, 8.0],  # T3
        ], dtype=np.float32)

        self.target_id = None
        self.target = None

        self.reset()

    def reset(self):
        self.agent_a = self._random_position()
        self.agent_b = self._random_position()

        # Choose which target is active
        self.target_id = np.random.randint(0, 4)

        self.target = self.target_locations[self.target_id].copy()

    def _random_position(self):
        return np.random.uniform(
            low=0.5,
            high=9.5,
            size=2
        ).astype(np.float32)