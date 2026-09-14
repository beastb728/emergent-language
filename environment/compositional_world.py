import numpy as np


class CompositionalWorld:

    def __init__(self):

        self.width = 10.0
        self.height = 10.0

        self.agent_a = None
        self.agent_b = None
        self.target = None

        self.x_region = None
        self.y_region = None

        self.max_steps = 50
        self.step_count = 0

    def reset(self):

        self.agent_a = np.random.uniform(
            0.5,
            9.5,
            size=2
        ).astype(np.float32)

        self.agent_b = np.random.uniform(
            0.5,
            9.5,
            size=2
        ).astype(np.float32)

        self.x_region = np.random.randint(0, 4)
        self.y_region = np.random.randint(0, 4)

        self.target = self._sample_target()

        while self._in_target_region():

            self.agent_b = np.random.uniform(
                0.5,
                9.5,
                size=2
            ).astype(np.float32)

        self.step_count = 0

    def _sample_target(self):

        x_min = self.x_region * 2.5
        x_max = x_min + 2.5

        y_min = self.y_region * 2.5
        y_max = y_min + 2.5

        return np.array(
            [
                np.random.uniform(x_min + 0.25, x_max - 0.25),
                np.random.uniform(y_min + 0.25, y_max - 0.25)
            ],
            dtype=np.float32
        )

    def sender_state(self):

        return np.concatenate(
            [
                self.agent_a / 10.0,
                self.agent_b / 10.0,
                self.target / 10.0
            ]
        ).astype(np.float32)

    def receiver_state(self, message):

        message_one_hot = np.zeros(
            4,
            dtype=np.float32
        )

        message_one_hot[message] = 1.0

        return np.concatenate(
            [
                self.agent_b / 10.0,
                message_one_hot
            ]
        ).astype(np.float32)

    def _in_target_region(self):

        x = self.agent_b[0]
        y = self.agent_b[1]

        x_min = self.x_region * 2.5
        x_max = x_min + 2.5

        y_min = self.y_region * 2.5
        y_max = y_min + 2.5

        return (
            x_min <= x < x_max
            and
            y_min <= y < y_max
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
            4: np.array([1.0, 0.0])
        }

        self.agent_b += (
            actions[action] * 0.25
        )

        self.agent_b = np.clip(
            self.agent_b,
            0.0,
            10.0
        )

        new_distance = self.distance()

        reward = old_distance - new_distance

        reached = self._in_target_region()

        if reached:

            reward += 10.0

        self.step_count += 1

        done = (
            reached
            or
            self.step_count >= self.max_steps
        )

        return reward, done, reached