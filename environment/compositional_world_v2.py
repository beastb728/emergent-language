import numpy as np


class CompositionalWorldV2:
    """
    4x4 compositional communication environment.

    Target is defined by two independent factors:

        x_region ∈ {0,1,2,3}
        y_region ∈ {0,1,2,3}

    The sender must communicate using TWO separate symbols:

        symbol_x ∈ {0,1,2,3}
        symbol_y ∈ {0,1,2,3}

    The receiver sees its own position plus both symbols.
    """

    def __init__(self):
        self.width = 10.0
        self.height = 10.0
        self.max_steps = 50
        self.step_count = 0

        self.agent_a = None
        self.agent_b = None
        self.target = None

        self.x_region = None
        self.y_region = None

    def reset(self):
        self.agent_a = np.random.uniform(
            0.5, 9.5, size=2
        ).astype(np.float32)

        self.x_region = np.random.randint(0, 4)
        self.y_region = np.random.randint(0, 4)

        self.target = self._sample_target()

        # Receiver must start outside target cell.
        while True:
            self.agent_b = np.random.uniform(
                0.5, 9.5, size=2
            ).astype(np.float32)

            if not self.in_target_region():
                break

        self.step_count = 0

    def _sample_target(self):
        cell_width = 10.0 / 4.0

        x_low = self.x_region * cell_width + 0.25
        x_high = (self.x_region + 1) * cell_width - 0.25

        y_low = self.y_region * cell_width + 0.25
        y_high = (self.y_region + 1) * cell_width - 0.25

        x = np.random.uniform(x_low, x_high)
        y = np.random.uniform(y_low, y_high)

        return np.array(
            [x, y],
            dtype=np.float32
        )

    def sender_state(self):
        return np.concatenate(
            [
                self.agent_a / 10.0,
                self.agent_b / 10.0,
                self.target / 10.0,
            ]
        ).astype(np.float32)

    def receiver_state(self, symbol_x, symbol_y):
        """
        Receiver gets position + TWO independent symbols.

        State size = 2 + 4 + 4 = 10.
        """

        x_one_hot = np.zeros(4, dtype=np.float32)
        y_one_hot = np.zeros(4, dtype=np.float32)

        x_one_hot[symbol_x] = 1.0
        y_one_hot[symbol_y] = 1.0

        return np.concatenate(
            [
                self.agent_b / 10.0,
                x_one_hot,
                y_one_hot,
            ]
        ).astype(np.float32)

    def in_target_region(self):
        cell_width = 10.0 / 4.0

        x_cell = min(
            int(self.agent_b[0] / cell_width),
            3
        )

        y_cell = min(
            int(self.agent_b[1] / cell_width),
            3
        )

        return (
            x_cell == self.x_region
            and y_cell == self.y_region
        )

    def distance(self):
        return np.linalg.norm(
            self.agent_b - self.target
        )

    def step(self, action):
        old_distance = self.distance()

        actions = {
            0: np.array([0.0, 0.0]),
            1: np.array([0.0, 1.0]),
            2: np.array([0.0, -1.0]),
            3: np.array([-1.0, 0.0]),
            4: np.array([1.0, 0.0]),
        }

        step_size = 0.25

        self.agent_b += actions[action] * step_size

        self.agent_b = np.clip(
            self.agent_b,
            0.0,
            10.0
        )

        new_distance = self.distance()

        reward = old_distance - new_distance

        self.step_count += 1

        reached = self.in_target_region()

        if reached:
            reward += 10.0

        done = (
            reached
            or self.step_count >= self.max_steps
        )

        return reward, done, reached