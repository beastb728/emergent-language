import numpy as np


class CommunicationWorld:

    def __init__(self):

        self.width = 10.0
        self.height = 10.0

        self.target_locations = np.array([
            [2.0, 2.0],  # T0
            [8.0, 2.0],  # T1
            [2.0, 8.0],  # T2
            [8.0, 8.0],  # T3
        ], dtype=np.float32)

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

        self.target = (
            self.target_locations[self.target_id]
            .copy()
        )

        self.step_count = 0

    def sender_state(self):

        """
        Information available to Agent A.

        A knows where the target is.
        """

        return np.concatenate([
            self.agent_a / 10.0,
            self.agent_b / 10.0,
            self.target / 10.0,
        ]).astype(np.float32)

    def receiver_state(self, message):

        """
        Information available to Agent B.

        B does NOT receive the target position.
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

        reached = new_distance < 0.35

        if reached:
            reward += 10.0

        done = (
            reached
            or self.step_count >= self.max_steps
        )

        return reward, done, reached