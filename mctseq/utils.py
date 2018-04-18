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


def immutable_seq(sequence):
    """
    :param sequence: a seq wich is mutable or not
    :return: an immutable seq
    """
    if type(sequence) == list:
        return sequence_mutable_to_immutable(sequence)
    else:
        return sequence


def count_target_class_data(data, target_class):
    """
    Count the number of occurences of target_class in the data
    :param data: sequential data of for [[class, {}, {} ...], [class, {}, {}], ...]
    :param target_class: the targeted class
    :return: the count
    """
    count = 0
    for row in data:
        if row[0] == target_class:
            count += 1

    return count


def k_length(sequence):
    """
    :param sequence: the considered sequence
    :return: the length of the sequence
    """
    return sum([len(i) for i in sequence])


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


def subsequence_indices(a, b):
    """ Return itemset indices of b that itemset of a are included in
        Precondition: a is a subset of b
    """
    index_b_mem = 0
    indices_b = []
    for index_a, itemset_a in enumerate(a):
        for index_b in range(index_b_mem, len(b)):
            if index_b == len(b) - 1:
                # we mark as finished
                index_b_mem = len(b)

            itemset_b = b[index_b]

            if itemset_a.issubset(itemset_b):
                indices_b.append(index_b)
                index_b_mem = index_b + 1
                break

        if index_b_mem == len(b):
            return indices_b

    return indices_b


def read_data(filename):
    sequences = []
    with open(filename) as f:
        for line in f:
            line_split = line.split(',')
            sequence = [line_split[0]]
            sequence += line_split[2].strip()

            sequences.append(sequence)

    for sequence in sequences:
        for itemset_i in range(1, len(sequence)):
            sequence[itemset_i] = set(sequence[itemset_i])

    return sequences


def read_data_r8(filename):
    """

    :param filename:
    :return: [[class, {}, {}, ...], [class, {}, {}, ...]]
    """
    data = []
    count = 0
    with open(filename) as f:
        for line in f:
            count += 1
            line_split = line.split()

            sequence = [line_split[0]]
            for itemset in line_split[1:10]:
                sequence.append({itemset})
            data.append(sequence)

            if count > 100:
                return data
    return data


def extract_items(data):
    """
    :param data: date must be on the form [[class, {}, {}, ...], [class, {}, {}, ...]]
    :return: set of items extracted
    """
    items = set()
    for sequence in data:
        for itemset in sequence[1:]:
            for item in itemset:
                items.add(item)
    return items


def uct(node, child_node):
    return child_node.quality + (2 / math.sqrt(2)) * math.sqrt(
        (2 * math.log(node.number_visit)) / child_node.number_visit)


def following_ones(bitset, bitset_slot_size):
    """
    Transform bitset with 1s following for each 1 encoutered, for
    each bitset_slot.
    :param bitset:
    :param bitset_slot_size: the size of a slot in the bitset
    :return: a bitset (number)
    """
    ones = False

    for i in range(bitset.bit_length(), 0, -1):
        if i % bitset_slot_size == 0:
            ones = False

        if ones:
            bitset = bitset | 2 ** (i - 1)

        if not ones and bitset >> (i - 1) & 1:
            ones = True
            bitset = bitset ^ 2 ** (i - 1)

    return bitset


def generate_bitset(itemset, data, bitset_slot_size):
    """
    Generate the bitset of itemset

    :param itemset: the itemset we want to get the bitset
    :param data: the dataset
    :return: the bitset of itemset
    """

    bitset = 0

    # we compute the extend by scanning the database
    for line in data:
        line = line[1:]
        sequence_bitset = 0
        for itemset_line in line:
            if itemset.issubset(itemset_line):
                bit = 1
            else:
                bit = 0

            sequence_bitset |= bit
            sequence_bitset = sequence_bitset << 1

        # for last element we need to reshift
        sequence_bitset = sequence_bitset >> 1

        # we shift to complete with 0
        sequence_bitset = sequence_bitset << bitset_slot_size - (len(line))

        # we add this bit vector to bitset
        bitset |= sequence_bitset
        bitset = bitset << bitset_slot_size

    # for the last element we need to reshift
    bitset = bitset >> bitset_slot_size

    return bitset


def print_results(results):
    for result in results:
        pattern_display = ''

        for itemset in result[1]:
            pattern_display += repr(set(itemset))

        print('WRAcc: {}, Pattern: {}'.format(result[0], pattern_display))


def print_results_mcts(results):
    for result in results:
        pattern_display = ''

        for itemset in result[1].sequence:
            pattern_display += repr(set(itemset))

        print('WRAcc: {}, Pattern: {}'.format(result[0], pattern_display))
