import heapq
import copy
from seqsamphill.utils import jaccard_measure, is_subsequence, sequence_mutable_to_immutable

THETA = 0.5

def jaccard_measure_misere(sequence1, sequence2, data):
    intersection = 0
    union = 0
    for sequence in data:
        sequence = sequence[1:]
        sequence = sequence_mutable_to_immutable(sequence)
        seq1 = False
        seq2 = False

        if is_subsequence(sequence1, sequence):
            seq1 = True
        if is_subsequence(sequence2, sequence):
            seq2 = True

        if seq1 or seq2:
            union += 1

        if seq1 and seq2:
            intersection += 1

    try:
        return intersection / union
    except ZeroDivisionError:
        return 0


def filter_results_misere(results, data, theta, k):
    """
    Filter redundant elements
    :param results: must be a node
    :param theta:
    :return: filtered list
    """

    results_list = list(results)
    results_list.sort(key=lambda x: x[0], reverse=True)

    filtered_elements = []

    for i, result in enumerate(results_list):
        similar = False

        for filtered_element in filtered_elements:
            if jaccard_measure_misere(result[1],
                                      filtered_element[1], data) > theta:
                similar = True

        if not similar:
            filtered_elements.append(result)

        if len(filtered_elements) > k:
            break

    return filtered_elements



class PrioritySet(object):
    """
    This class is a priority queue, removing duplicates and using node wracc
    as the metric to order the priority queue
    """

    def __init__(self, k=5):
        self.k = k
        self.heap = []
        self.set = set()

    def add(self, sequence, wracc):
        if sequence not in self.set:
            heapq.heappush(self.heap, (wracc, sequence))
            self.set.add(sequence)

    def add_preserve_memory(self, sequence, wracc, data):
        self.add(sequence, wracc)

        # we remove elements that are not in top-k
        self.heap = self.get_top_k(self.k)

        ### UGLY ###
        set_top_k = set()

        for _, seq in self.heap:
            set_top_k.add(seq)

        self.set = set_top_k

    def add_preserve_memory_v2(self, sequence, wracc, data):
        '''
        add sequence to data-structure only if wracc is better than what there is already
        Not optimized (no time)!
        :param sequence:
        :param wracc:
        :return:
        '''
        if len(self.heap) < self.k:
            self.add(sequence, wracc)
            return

        if wracc > min(self.heap, key=lambda x: x[1])[0]:
            heap_copy = copy.deepcopy(self.heap)
            heap_copy.append((wracc, sequence))
            heap_copy = filter_results_misere(heap_copy, data, THETA, self.k)

            sum_wracc = 0
            for wracc_i, seq_i in heap_copy:
                sum_wracc += wracc_i

            mean_add = sum_wracc / len(heap_copy)

            sum_wracc = 0
            for wracc_i, seq_i in self.heap:
                sum_wracc += wracc_i

            mean_actual = sum_wracc / len(self.heap)

            # if no amelioration, do not add
            if mean_actual > mean_add:
               return


            self.add(sequence, wracc)

            # we remove elements that are not in top-k
            self.heap = self.get_top_k(self.k)

            ### UGLY ###
            set_top_k = set()

            for _, seq in self.heap:
                set_top_k.add(seq)

            self.set = set_top_k


    def get(self):
        wracc, sequence = heapq.heappop(self.heap)
        self.set.remove(sequence)
        return (wracc, sequence)

    def get_top_k(self, k):
        data = heapq.nlargest(k, self.heap)
        return data

    def get_top_k_non_redundant(self, data, k):
        self.heap = filter_results_misere(self.heap, data, THETA, k)

        return self.get_top_k(k)


class PrioritySetv2(object):
    # simple because no time
    def __init__(self, k=10):
        self.k = k
        self.heap = []

    def add(self, sequence, wracc):
        if sequence not in self.heap:
            self.heap.append((wracc, sequence))
            self.heap.sort(key=lambda x: x[0], reverse=True)

    def add_non_redondant(self, sequence, wracc, data):
        '''
        add sequence to data-structure only if wracc is better than what there is already
        Not optimized (no time)!
        :param sequence:
        :param wracc:
        :return:
        '''
        #print('adding'+str(sequence))
        if len(self.heap) < self.k:
            self.add(sequence, wracc)
            return

        if wracc > min(self.heap, key=lambda x: x[1])[0]:
            self.add(sequence, wracc)

            # we remove elements that are not in top-k
            self.heap = self.get_top_k(self.k)

            ### UGLY ###
            set_top_k = set()

            for _, seq in self.heap:
                set_top_k.add(seq)

            self.set = set_top_k

    def get(self):
        wracc, sequence = self.heap[0]
        self.heap.pop(0)
        return (wracc, sequence)

    def get_top_k(self, k):
        return self.heap[:k]

    def get_top_k_non_redundant(self, data, k):
        self.heap = filter_results_misere(self.heap, data, THETA, k)
        #TEST
        #similarity_first = jaccard_measure_misere(self.heap[0][1], self.heap[1][1], data)
        #print(similarity_first)

        return self.get_top_k(k)


class PrioritySetUCB(object):
    """
    This class is a priority queue, removing duplicates and using node wracc
    as the metric to order the priority queue
    """
    def __init__(self):
        self.heap = []
        self.set = set()

    def add(self, sequence, tuple):
        '''
        :param sequence:
        :param tuple: (UCB, Ni, WRAcc)
        :return:
        '''
        if sequence not in self.set:
            # we use - sign because heapq return the smalest element
            heapq.heappush(self.heap, (-tuple[0], tuple[1], tuple[2], sequence))
            self.set.add(sequence)

    def pop(self):
        '''
        :return: the max element
        '''
        UCB, Ni, wracc, sequence = heapq.heappop(self.heap)
        self.set.remove(sequence)
        return (-UCB, Ni, wracc, sequence)


