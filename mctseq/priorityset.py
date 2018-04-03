import heapq


class PrioritySetQuality(object):
    """
    This class is a priority queue, removing duplicates and using node wracc
    as the metric to order the priority queue
    """

    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, node):
        if node not in self.set:
            heapq.heappush(self.heap, (node.wracc, node))
            self.set.add(node)

    def get(self):
        quality, node = heapq.heappop(self.heap)
        self.set.remove(node)
        return (quality, node)

    def get_top_k(self, k):
        return heapq.nlargest(k, self.heap)
