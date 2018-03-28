import math


def sequence_mutable_to_immutable(sequence):
    """
    :param sequence: form [{}, {}, ...]
    :return: the same sequence in its immutable form
    """
    return tuple([frozenset(i) for i in sequence])


def sequence_immutable_to_mutable(sequence):
    """
    :param sequence: form (frozenset(), frozenset(), ...)
    :return: the same sequence in its mutable form
    """
    return [set(i) for i in sequence]


def count_target_class_data(data, target_class):
    """
    Count the number of occurences of target_class in the data
    :param data: sequential data
    :param target_class: the targeted class
    :return: the count
    """
    count = 0
    for row in data:
        if data[0] == target_class:
            count += 0

    return count


def is_subsequence(a, b):
    """ check if sequence a is a subsequence of b
    """
    index_b_mem = 0

    for index_a, itemset_a in enumerate(a):
        for index_b in range(index_b_mem, len(b)):
            if index_b == len(b) - 1:
                # we mark as finished
                index_b_mem = len(b)

            itemset_b = b[index_b]

            if itemset_a.issubset(itemset_b):
                index_b_mem = index_b + 1
                break

        if index_b_mem == len(b):
            if index_a < len(a) - 1:
                # we reach the end of b and there are still elements in a
                return False
            elif itemset_a.issubset(b[-1]):
                # we reach the end of a and b, a_last_elt is included in
                # b_last_elt
                return True
            else:
                # we reach the end of a and b, a_last_elt is not included in
                # b_last_elt
                return False

    return True


def read_data(filename):
    sequences = []
    with open(filename) as f:
        for line in f:
            sequence = [set(i.replace(' ', '')) for i in line[:-3].split('-1')]
            sequences.append(sequence)
    return sequences


def extract_items(data):
    items = set()
    for sequence in data:
        for itemset in sequence:
            for item in itemset:
                items.add(item)
    return items


def uct(node, child_node):
    return child_node.quality + 2 / math.sqrt(2) * math.sqrt(
        2 * math.log(node.number_visit) / child_node.number_visit)
