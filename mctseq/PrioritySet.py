import heapq

class PrioritySetQuality(object):
    """
    This class is a priority queue, removing duplicates and using node quality
    as reference
    """
    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, node):
        if not node in self.set:
            heapq.heappush(self.heap, (node.quality, node))
            self.set.add(node)

    def get(self):
        quality, node = heapq.heappop(self.heap)
        self.set.remove(node)
        return (quality, node)
