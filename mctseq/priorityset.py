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

    def get_top_k(self, k):
        return_list = []
        for i in range(k):
            return_list.append(self.get())

        return return_list
