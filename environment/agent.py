import numpy as np


class Agent:
    ACTIONS = {
        0: np.array([0.0, 0.0]),   # stay
        1: np.array([0.0, 1.0]),   # up
        2: np.array([0.0, -1.0]),  # down
        3: np.array([-1.0, 0.0]),  # left
        4: np.array([1.0, 0.0]),   # right
    }

    def __init__(self, position):
        self.position = position.copy()

    def move(self, action, step_size=0.25):
        direction = self.ACTIONS[action]

        self.position += direction * step_size

        self.position[0] = np.clip(
            self.position[0], 0.0, 10.0
        )

        self.position[1] = np.clip(
            self.position[1], 0.0, 10.0
        )