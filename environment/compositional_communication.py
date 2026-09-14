class CompositionalProtocol:

    def __init__(self):
        self.message_size = 4

    def encode(self, x_region, y_region):
        return x_region, y_region

    def decode(self, message_1, message_2):
        return message_1, message_2