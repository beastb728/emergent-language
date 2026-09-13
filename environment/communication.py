class PredefinedProtocol:
    """
    Human-designed communication protocol.

    00 -> T0
    01 -> T1
    10 -> T2
    11 -> T3
    """

    def encode(self, target_id):
        return format(target_id, "02b")

    def decode(self, message):
        return int(message, 2)