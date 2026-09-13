import random


class RandomFixedProtocol:

    def __init__(self):

        messages = [0, 1, 2, 3]

        random.shuffle(messages)

        self.mapping = {
            target_id: messages[target_id]
            for target_id in range(4)
        }

    def encode(self, target_id):

        return self.mapping[target_id]

    def show(self):

        print("Hidden communication convention:")

        for target_id in range(4):

            message = self.mapping[target_id]

            print(
                f"T{target_id} -> "
                f"{format(message, '02b')}"
            )