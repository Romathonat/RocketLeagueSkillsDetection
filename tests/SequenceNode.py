
class SequenceNode():
    def __init__(self, pattern, parent):
        self.pattern = pattern
        self.parent = parent
        self.quality = 0
        self.number_visit = 0

