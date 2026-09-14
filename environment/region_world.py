import numpy as np


class RegionWorld:

    def __init__(self):

        self.width = 10.0
        self.height = 10.0

        self.agent_a = None
        self.agent_b = None

        self.target = None
        self.target_id = None

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

        self.target_id = np.random.randint(0, 4)

        # Random target inside the selected region.
        if self.target_id == 0:
            x = np.random.uniform(0.75, 4.75)
            y = np.random.uniform(0.75, 4.75)

        elif self.target_id == 1:
            x = np.random.uniform(5.25, 9.25)
            y = np.random.uniform(0.75, 4.75)

        elif self.target_id == 2:
            x = np.random.uniform(0.75, 4.75)
            y = np.random.uniform(5.25, 9.25)

        else:
            x = np.random.uniform(5.25, 9.25)
            y = np.random.uniform(5.25, 9.25)

        self.target = np.array(
            [x, y],
            dtype=np.float32
        )

        self.step_count = 0

    def sender_state(self):

        """
        Agent A knows the exact target position.
        """

        return np.concatenate([
            self.agent_a / 10.0,
            self.agent_b / 10.0,
            self.target / 10.0,
        ]).astype(np.float32)

    def receiver_state(self, message):

        """
        Agent B receives its own position and
        the communication symbol.

        B does not receive the target position.
        """

        message_one_hot = np.zeros(
            4,
            dtype=np.float32
        )

        message_one_hot[message] = 1.0

        return np.concatenate([
            self.agent_b / 10.0,
            message_one_hot,
        ]).astype(np.float32)

    def region_center(self, target_id):

        centers = {
            0: np.array([2.75, 2.75]),
            1: np.array([7.25, 2.75]),
            2: np.array([2.75, 7.25]),
            3: np.array([7.25, 7.25]),
        }

        return centers[target_id]

    def in_target_region(self):

        x, y = self.agent_b

        if self.target_id == 0:
            return x < 5.0 and y < 5.0

        if self.target_id == 1:
            return x >= 5.0 and y < 5.0

        if self.target_id == 2:
            return x < 5.0 and y >= 5.0

        return x >= 5.0 and y >= 5.0

    def distance(self):

        center = self.region_center(
            self.target_id
        )

        return np.linalg.norm(
            self.agent_b - center
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

        self.agent_b += (
            actions[action] * step_size
        )

        self.agent_b = np.clip(
            self.agent_b,
            0.0,
            10.0
        )

        new_distance = self.distance()

        reward = (
            old_distance - new_distance
        )

        self.step_count += 1

        reached = self.in_target_region()

        if reached:
            reward += 10.0

        done = (
            reached
            or self.step_count >= self.max_steps
        )

        return reward, done, reached